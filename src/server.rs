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

pub fn pre_process_batch(
    pvw_params: &Arc<PvwParameters>,
    bfv_params: Arc<BfvParameters>,
    hints: &[PvwCiphertext],
) -> (Vec<Plaintext>, Vec<Poly>) {
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
pub fn encrypt_pvw_sk<R: CryptoRng + RngCore>(
    bfv_params: &Arc<BfvParameters>,
    bfv_sk: &SecretKey,
    pvw_sk: &PvwSecretKey,
    rng: &mut R,
) -> Vec<Ciphertext> {
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
            bfv_sk.encrypt(&pt, rng)
        })
        .collect_vec();

    cts
}

// rotate by 1 and perform plaintext mutiplication for each ell
pub fn pvw_decrypt(
    pvw_params: &Arc<PvwParameters>,
    hint_a_pts: &[Plaintext],
    hint_b_pts: &[Poly],
    pvw_sk_cts: Vec<Ciphertext>,
    rtk: &GaloisKey,
) -> Vec<Ciphertext> {
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
        .map(|s_ct| optimised_fma_with_rot(s_ct, hint_a_pts, sec_len, rtk))
        .collect_vec();

    sk_a.iter_mut().zip(hint_b_pts.iter()).for_each(|(sa, b)| {
        // FIXME: Wo don't need this
        sa.sub_reversed_inplace(b);
    });

    sk_a
}

// TODO: remove clones
pub fn powers_of_x_ct(x: &Ciphertext, rlk: &RelinearizationKey) -> Vec<Ciphertext> {
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
                    let tmp = values[p_res_deg - 1].multiply1(&values[base_deg - 1]);
                    values[res_deg - 1] = rlk.relinearize(&tmp);
                    calculated[res_deg - 1] = 1;
                    mul_count += 1;
                }
            }
            exp >>= 1;
            if exp != 0 {
                let p_base_deg = base_deg;
                base_deg *= 2;
                if calculated[base_deg - 1] == 0 {
                    let tmp = values[p_base_deg - 1].multiply1(&values[p_base_deg - 1]);
                    values[base_deg - 1] = rlk.relinearize(&tmp);
                    calculated[base_deg - 1] = 1;
                    mul_count += 1;
                }
            }
        }
    }
    dbg!(mul_count);

    values
}

pub fn range_fn(
    ct: &Ciphertext,
    rlk: &RelinearizationKey,
    constants: &Array2<u64>,
    sub_from_one_precompute: &[u64],
    sk: &SecretKey,
) -> Ciphertext {
    let mut now = Instant::now();
    let mut single_powers = powers_of_x_ct(ct, rlk);
    println!("single_powers: {:?}", now.elapsed());
    decrypt_and_print(&single_powers[255], sk, "single_powers[255]");

    now = Instant::now();
    let double_powers = powers_of_x_ct(&single_powers[255], rlk);
    println!("double_powers: {:?}", now.elapsed());
    decrypt_and_print(&double_powers[255], sk, "double_powers[255]");

    // change to evaluation for plaintext multiplication
    now = Instant::now();
    single_powers.iter_mut().for_each(|ct| {
        ct.change_representation(&Representation::Evaluation);
    });
    println!(
        "single_powers coefficient to evaluation: {:?}",
        now.elapsed()
    );

    let level = 0;
    let bfv_params = ct.params();
    let q_ctx = bfv_params.ciphertext_ctx_at_level(level);
    let q_size = q_ctx.moduli.len();

    // when i = 0, we skip multiplication and cache the result
    let mut left_over_ct = Ciphertext::zero(&bfv_params, level);
    let mut sum_ct = Ciphertext::zero(&bfv_params, level);

    now = Instant::now();
    for i in 0..256 {
        let mut res0_u128 = Array2::<u128>::zeros((q_size, ct.params().polynomial_degree));
        let mut res1_u128 = Array2::<u128>::zeros((q_size, ct.params().polynomial_degree));

        // let mut inner_now = Instant::now();
        for j in 1..257 {
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
    println!("Outer summation: {:?}", now.elapsed());
    decrypt_and_print(&sum_ct, sk, "Outer smmation");

    // implement optimised 1 - sum_ct
    sub_from_one(&mut sum_ct, sub_from_one_precompute);
    sum_ct
}

#[cfg(test)]
mod tests {
    use std::f32::consts::E;

    use super::*;
    use crate::{optimised::sub_from_one_precompute, utils::precompute_range_constants};
    use bfv::BfvParameters;

    #[test]
    fn range_fn_works() {
        let params = Arc::new(BfvParameters::default(10, 1 << 15));
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
            if *e <= 850 || *e >= (65537 - 850) {
                assert!(*r == 1);
            } else {
                assert!(*r == 0);
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
                powers_of_x_ct(&ct, &rlk);
            }
        }

        let now = std::time::Instant::now();
        let powers = powers_of_x_ct(&ct, &rlk);
        println!("Time = {:?}", now.elapsed());
    }
}
