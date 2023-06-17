use crate::{
    optimised::{
        barret_reduce_coefficients_u128, optimised_add_range_fn, optimised_fma_with_rot,
        optmised_range_fn_fma, sub_from_one,
    },
    pvw::{self, PvwCiphertext, PvwParameters, PvwSecretKey},
    utils::{decrypt_and_print, read_range_coeffs},
};
use bfv::{
    BfvParameters, Ciphertext, Encoding, GaloisKey, Modulus, Plaintext, Poly, RelinearizationKey,
    Representation, SecretKey,
};
use itertools::{izip, Itertools};
use ndarray::{azip, s, Array2, IntoNdProducer};
use rand::{thread_rng, CryptoRng, RngCore};
use rand_chacha::rand_core::le;
use std::{hint, sync::Arc, time::Instant};
use traits::Ntt;

pub fn pre_process_batch<T: Ntt>(
    pvw_params: &Arc<PvwParameters>,
    bfv_params: &Arc<BfvParameters<T>>,
    hints: &[PvwCiphertext],
) -> (Vec<Plaintext<T>>, Vec<Poly<T>>) {
    // can only process as many as polynomial_degree hints in a batch
    debug_assert!(hints.len() <= bfv_params.polynomial_degree);

    let sec_len = pvw_params.n.next_power_of_two();
    let mut hint_a_pts = vec![];
    for i in 0..sec_len {
        let mut m = vec![];
        for j in 0..hints.len() {
            let index = (j + i) % sec_len;
            if index < pvw_params.n {
                m.push(hints[j].a[index]);
            } else {
                m.push(0);
            }
        }
        hint_a_pts.push(Plaintext::encode(&m, &bfv_params, Encoding::simd(0)));
    }

    let mut hint_b_polys = vec![];
    let q_by4 = bfv_params.plaintext_modulus / 4;
    for i in 0..pvw_params.ell {
        let mut m = vec![];
        for j in 0..hints.len() {
            m.push(
                bfv_params
                    .plaintext_modulus_op
                    .sub_mod_fast(hints[j].b[i], q_by4),
            );
        }
        hint_b_polys.push(Plaintext::encode(&m, &bfv_params, Encoding::simd(0)).to_poly());
    }

    // length of plaintexts will be sec_len
    (hint_a_pts, hint_b_polys)
}

/// Encrypts pvw sk under bfv in desired form
pub fn encrypt_pvw_sk<R: CryptoRng + RngCore, T: Ntt>(
    bfv_params: &Arc<BfvParameters<T>>,
    bfv_sk: &SecretKey<T>,
    pvw_sk: &PvwSecretKey,
    rng: &mut R,
) -> Vec<Ciphertext<T>> {
    let sec_len = pvw_sk.par.n.next_power_of_two();
    let degree = bfv_params.polynomial_degree;

    // pvw_sk.key is of dimension ell x n
    let cts = pvw_sk
        .key
        .outer_iter()
        .map(|s| {
            let mut m = vec![];
            for i in 0..degree {
                let index = i % sec_len;
                if index < pvw_sk.par.n {
                    m.push(s[index]);
                } else {
                    m.push(0);
                }
            }
            let pt = Plaintext::encode(&m, bfv_params, Encoding::simd(0));
            let mut ct = bfv_sk.encrypt(&pt, rng);
            ct.change_representation(&Representation::Evaluation);
            ct
        })
        .collect_vec();

    cts
}

// rotate by 1 and perform plaintext mutiplication for each ell
pub fn pvw_decrypt<T: Ntt>(
    pvw_params: &Arc<PvwParameters>,
    hint_a_pts: &[Plaintext<T>],
    hint_b_pts: &[Poly<T>],
    pvw_sk_cts: &Vec<Ciphertext<T>>,
    rtk: &GaloisKey<T>,
    sk: &SecretKey<T>,
) -> Vec<Ciphertext<T>> {
    let sec_len = pvw_params.n.next_power_of_two();
    debug_assert!(hint_a_pts.len() == sec_len);
    debug_assert!(hint_b_pts.len() == pvw_params.ell);
    debug_assert!(pvw_sk_cts.len() == pvw_params.ell);

    // d[j] = s[j][0] * p[0] + s[j][1] * p[1] + ... + s[j][sec_len-1] * p[sec_len-1]
    // where s[j][a] is s[j] rotated to left by 1 `a` times.
    // Each operation is further broken down to: d[j] += s[j][0] * p[0]. Can we take
    // advantage of fused multiplication addition to speed this up? For ex, hexl has
    // an API for FMA which is faster (should be right?) than perfoming vector multiplication
    // and addition in a sequence.
    // There's an additinal optimisation for FMA operation. We can perform FMA in 128 bits without
    // modulur reduction followed by 128 bit barret reduction in the end. Since we will only be adding 512 128 bits values,
    // result will not overflow. Amazing!
    // TODO: Provide and API for FMA (ie Ct + Ct * Pt) in Bfv.
    //
    // Length of `d == ell`.
    // let mut d = vec![];
    let mut sk_a = pvw_sk_cts
        .into_iter()
        .map(|s_ct| optimised_fma_with_rot(s_ct, hint_a_pts, sec_len, rtk, sk))
        .collect_vec();

    // {
    //     let mut rng = thread_rng();
    //     dbg!(sk.measure_noise(&sk_a[0], &mut rng));
    // }

    sk_a.iter_mut().zip(hint_b_pts.iter()).for_each(|(sa, b)| {
        // FIXME: Wo don't need this
        sa.sub_reversed_inplace(b);
    });

    sk_a
}

// TODO: remove clones
pub fn powers_of_x_ct<T: Ntt>(
    x: &Ciphertext<T>,
    rlk: &RelinearizationKey<T>,
    sk: &SecretKey<T>,
) -> Vec<Ciphertext<T>> {
    let mut values = vec![Ciphertext::zero(&x.params(), x.level()); 256];
    let mut calculated = vec![0u64; 256];
    values[0] = x.clone();
    calculated[0] = 1;
    let mut mul_count = 0;

    for i in (2..257).rev() {
        let mut exp = i;
        let mut base_deg = 1;
        let mut res_deg = 0;

        while exp > 0 {
            if exp & 1 == 1 {
                let p_res_deg = res_deg;
                res_deg += base_deg;
                if res_deg != base_deg && calculated[res_deg - 1] == 0 {
                    // let now = Instant::now();
                    let tmp = values[p_res_deg - 1].multiply1(&values[base_deg - 1]);
                    values[res_deg - 1] = rlk.relinearize(&tmp);
                    // println!("Res deg time: {:?}", now.elapsed());
                    calculated[res_deg - 1] = 1;
                    // mul_count += 1;
                }
            }
            exp >>= 1;
            if exp != 0 {
                let p_base_deg = base_deg;
                base_deg *= 2;
                if calculated[base_deg - 1] == 0 {
                    // let now = Instant::now();
                    let tmp = values[p_base_deg - 1].multiply1(&values[p_base_deg - 1]);
                    values[base_deg - 1] = rlk.relinearize(&tmp);

                    {
                        let mut rng = thread_rng();
                        println!(
                            "base_deg {} noise: {}",
                            base_deg,
                            sk.measure_noise(&values[base_deg - 1], &mut rng)
                        );
                    }

                    // println!("Base deg time: {:?}", now.elapsed());
                    calculated[base_deg - 1] = 1;

                    // mul_count += 1;
                }
            }
        }
    }
    // dbg!(mul_count);

    values
}

pub fn range_fn<T: Ntt>(
    ct: &Ciphertext<T>,
    rlk: &RelinearizationKey<T>,
    constants: &Array2<u64>,
    sub_from_one_precompute: &[u64],
    sk: &SecretKey<T>,
) -> Ciphertext<T> {
    // let mut now = Instant::now();
    let mut single_powers = powers_of_x_ct(ct, rlk, sk);
    // println!("single_powers: {:?}", now.elapsed());
    decrypt_and_print(&single_powers[255], sk, "single_powers[255]");

    // now = Instant::now();
    let double_powers = powers_of_x_ct(&single_powers[255], rlk, sk);
    // println!("double_powers: {:?}", now.elapsed());
    decrypt_and_print(&double_powers[255], sk, "double_powers[255]");

    // change to evaluation for plaintext multiplication
    // now = Instant::now();
    single_powers.iter_mut().for_each(|ct| {
        ct.change_representation(&Representation::Evaluation);
    });
    // println!(
    //     "single_powers coefficient to evaluation: {:?}",
    //     now.elapsed()
    // );

    let level = 0;
    let bfv_params = ct.params();
    let q_ctx = bfv_params.ciphertext_ctx_at_level(level);
    let q_size = q_ctx.moduli.len();

    // when i = 0, we skip multiplication and cache the result
    let mut left_over_ct = Ciphertext::zero(&bfv_params, level);
    let mut sum_ct = Ciphertext::zero(&bfv_params, level);

    // now = Instant::now();
    for i in 0..256 {
        let mut res0_u128 = Array2::<u128>::zeros((q_size, ct.params().polynomial_degree));
        let mut res1_u128 = Array2::<u128>::zeros((q_size, ct.params().polynomial_degree));

        // let mut inner_now = Instant::now();
        // Starting from 0th index every alternate constant is 0. Since plintext multiplication by 0 is 0, we don't need to
        // process plaintext multiplications for indices at which constant is 0. Thus, we start from 1st index and process
        // every alternate index.
        for j in (2..257).step_by(2) {
            optmised_range_fn_fma(
                &mut res0_u128,
                &mut res1_u128,
                &single_powers[j - 1],
                constants
                    .slice(s![(i * 256) + (j - 1), ..])
                    .as_slice()
                    .unwrap(),
            );
        }

        let p_res0 = Poly::new(
            barret_reduce_coefficients_u128(&res0_u128, &q_ctx.moduli_ops),
            &q_ctx,
            Representation::Evaluation,
        );
        let p_res1 = Poly::new(
            barret_reduce_coefficients_u128(&res1_u128, &q_ctx.moduli_ops),
            &q_ctx,
            Representation::Evaluation,
        );

        let res_ct = Ciphertext::new(vec![p_res0, p_res1], ct.params(), level);
        // println!("Inner scalar product {i}: {:?}", inner_now.elapsed());
        // decrypt_and_print(&res_ct, sk, &format!("Inner scalar product {i}"));

        // cache i == 0
        if i == 0 {
            left_over_ct = res_ct;
            // convert  ct to coefficient form
            left_over_ct.change_representation(&Representation::Coefficient);
        } else if i == 1 {
            // multiply1_lazy returns in evaluation form
            sum_ct = res_ct.multiply1_lazy(&double_powers[i - 1]);
        } else {
            sum_ct += &res_ct.multiply1_lazy(&double_powers[i - 1]);
        }
    }

    sum_ct.scale_and_round();
    let mut sum_ct = rlk.relinearize(&sum_ct);
    sum_ct += &left_over_ct;
    // println!("Outer summation: {:?}", now.elapsed());
    decrypt_and_print(&sum_ct, sk, "Outer smmation");

    // implement optimised 1 - sum_ct
    sub_from_one(&mut sum_ct, sub_from_one_precompute);
    sum_ct
}

pub fn expand_pertinency_vector() {
    // extract first 32
    // expand 32 into 
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        optimised::sub_from_one_precompute, plaintext::powers_of_x_modulus,
        utils::precompute_range_constants,
    };
    use bfv::BfvParameters;

    #[test]
    fn range_fn_works() {
        let params = Arc::new(BfvParameters::default(15, 1 << 3));
        let ctx = params.ciphertext_ctx_at_level(0);

        let mut rng = thread_rng();
        let constants = precompute_range_constants(&ctx);
        let sub_one_precompute = sub_from_one_precompute(&params, 0);

        let sk = SecretKey::random(&params, &mut rng);
        let mut m = params
            .plaintext_modulus_op
            .random_vec(params.polynomial_degree, &mut rng);
        let pt = Plaintext::encode(&m, &params, Encoding::simd(0));
        let mut ct = sk.encrypt(&pt, &mut rng);

        // gen rlk
        let rlk = RelinearizationKey::new(&params, &sk, 0, &mut rng);

        let now = std::time::Instant::now();
        let ct_res = range_fn(&ct, &rlk, &constants, &sub_one_precompute, &sk);
        let time = now.elapsed();
        dbg!(time);
        dbg!(sk.measure_noise(&ct_res, &mut rng));
        let res = sk.decrypt(&ct_res).decode(Encoding::simd(0));

        izip!(res.iter(), m.iter()).for_each(|(r, e)| {
            if (*e <= 850 || *e >= (65537 - 850)) {
                assert_eq!(*r, 1);
            } else if *r != 0 {
                assert_eq!(*r, 0);
            }
        });
    }

    #[test]
    fn powers_of_x_ct_works() {
        let mut rng = thread_rng();
        let params = Arc::new(BfvParameters::default(15, 1 << 15));
        let sk = SecretKey::random(&params, &mut rng);
        let m = vec![3; params.polynomial_degree];
        let pt = Plaintext::encode(&m, &params, Encoding::simd(0));
        let ct = sk.encrypt(&pt, &mut rng);
        let rlk = RelinearizationKey::new(&params, &sk, 0, &mut rng);

        {
            for _ in 0..1 {
                powers_of_x_ct(&ct, &rlk, &sk);
            }
        }

        let now = std::time::Instant::now();
        let powers_ct = powers_of_x_ct(&ct, &rlk, &sk);
        println!("Time = {:?}", now.elapsed());

        let res_values_mod = powers_of_x_modulus(3, &params.plaintext_modulus_op);

        izip!(powers_ct.iter(), res_values_mod.iter()).for_each(|(pct, v)| {
            dbg!(sk.measure_noise(pct, &mut rng));
            let r = sk.decrypt(pct).decode(Encoding::simd(0));
            r.iter().for_each(|r0| {
                assert!(r0 == v);
            });
        });
    }
}
