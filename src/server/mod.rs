use crate::optimised::{coefficient_u128_to_ciphertext, fma_reverse_u128_poly};
use crate::preprocessing::{precompute_expand_32_roll_pt, procompute_expand_roll_pt};
use crate::utils::decrypt_and_print;
use crate::{
    optimised::{barret_reduce_coefficients_u128, optimised_pvw_fma_with_rot, sub_from_one},
    pvw::PvwParameters,
};
use bfv::{
    BfvParameters, Ciphertext, EvaluationKey, Evaluator, GaloisKey, Plaintext, Poly, PolyType,
    RelinearizationKey, Representation, SecretKey,
};
use itertools::{izip, Itertools};
use ndarray::{s, Array2};
use std::{collections::HashMap, sync::Arc, time::Instant};

pub mod phase2;
pub mod range_fn_fma;

// rotate by 1 and perform plaintext mutiplication for each ell
pub fn pvw_decrypt(
    pvw_params: &Arc<PvwParameters>,
    evaluator: &Evaluator,
    hint_a_pts: &[Plaintext],
    hint_b_pts: &[Poly],
    pvw_sk_cts: &[Ciphertext],
    rtg: &GaloisKey,
    sk: &SecretKey,
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
        .map(|s_ct| {
            optimised_pvw_fma_with_rot(evaluator.params(), s_ct, hint_a_pts, sec_len, rtg, sk)
        })
        .collect_vec();

    // {
    //     let mut rng = thread_rng();
    //     dbg!(sk.measure_noise(&sk_a[0], &mut rng));
    // }

    sk_a.iter_mut().zip(hint_b_pts.iter()).for_each(|(sa, b)| {
        // FIXME: Wo don't need this
        evaluator.sub_ciphertext_from_poly_inplace(sa, b);
    });

    sk_a
}

pub fn powers_of_x_ct(
    x: &Ciphertext,
    evaluator: &Evaluator,
    ek: &EvaluationKey,
    sk: &SecretKey,
) -> Vec<Ciphertext> {
    let dummy = Ciphertext::new(vec![], PolyType::Q, 0);
    let mut values = vec![dummy; 255];
    let mut calculated = vec![0u64; 255];
    values[0] = x.clone();
    calculated[0] = 1;
    // let mut mul_count = 0;

    for i in (2..256).rev() {
        let mut exp = i;
        let mut base_deg = 1;
        let mut res_deg = 0;

        while exp > 0 {
            if exp & 1 == 1 {
                let p_res_deg = res_deg;
                res_deg += base_deg;
                if res_deg != base_deg && calculated[res_deg - 1] == 0 {
                    // let now = Instant::now();
                    let tmp = evaluator.mul(&values[p_res_deg - 1], &values[base_deg - 1]);
                    values[res_deg - 1] = evaluator.relinearize(&tmp, ek);
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
                    let tmp = evaluator.mul(&values[p_base_deg - 1], &values[p_base_deg - 1]);
                    values[base_deg - 1] = evaluator.relinearize(&tmp, ek);

                    // unsafe {
                    //     decrypt_and_print(
                    //         &values[base_deg - 1],
                    //         sk,
                    //         &format!("base_deg {base_deg}"),
                    //     )
                    // };

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

pub fn even_powers_of_x_ct(
    x: &Ciphertext,
    evaluator: &Evaluator,
    ek: &EvaluationKey,
    sk: &SecretKey,
) -> Vec<Ciphertext> {
    let dummy = Ciphertext::new(vec![], PolyType::Q, 0);
    let mut values = vec![dummy; 128];
    let mut calculated = vec![0u64; 128];

    // x^2
    let tmp = evaluator.mul(x, x);
    values[0] = evaluator.relinearize(&tmp, ek);
    calculated[0] = 1;
    let mut mul_count = 0;

    for i in (4..257).step_by(2).rev() {
        // LSB of even value is 0. So we can ignore it.
        let mut exp = i >> 1;
        let mut base_deg = 2;
        let mut res_deg = 0;

        while exp > 0 {
            if exp & 1 == 1 {
                let p_res_deg = res_deg;
                res_deg += base_deg;
                if res_deg != base_deg && calculated[res_deg / 2 - 1] == 0 {
                    // let now = Instant::now();
                    let tmp = evaluator.mul(&values[p_res_deg / 2 - 1], &values[base_deg / 2 - 1]);
                    values[res_deg / 2 - 1] = evaluator.relinearize(&tmp, ek);
                    // println!("Res deg time: {:?}", now.elapsed());
                    calculated[res_deg / 2 - 1] = 1;
                    // mul_count += 1;
                }
            }
            exp >>= 1;
            if exp != 0 {
                let p_base_deg = base_deg;
                base_deg *= 2;
                if calculated[base_deg / 2 - 1] == 0 {
                    // let now = Instant::now();
                    let tmp =
                        evaluator.mul(&values[p_base_deg / 2 - 1], &values[p_base_deg / 2 - 1]);
                    values[base_deg / 2 - 1] = evaluator.relinearize(&tmp, ek);

                    // unsafe {
                    //     decrypt_and_print(
                    //         &values[base_deg - 1],
                    //         sk,
                    //         &format!("base_deg {base_deg}"),
                    //     )
                    // };

                    // println!("Base deg time: {:?}", now.elapsed());
                    calculated[base_deg / 2 - 1] = 1;

                    // mul_count += 1;
                }
            }
        }
    }
    // dbg!(mul_count);

    values
}

pub fn range_fn(
    ct: &Ciphertext,
    evaluator: &Evaluator,
    ek: &EvaluationKey,
    constants: &Array2<u64>,
    sub_from_one_precompute: &[u64],
    sk: &SecretKey,
) -> Ciphertext {
    // let mut now = Instant::now();
    let mut single_powers = even_powers_of_x_ct(ct, evaluator, ek, sk);
    // println!("single_powers: {:?}", now.elapsed());
    unsafe { decrypt_and_print(&evaluator, &single_powers[127], sk, "single_powers[127]") }

    // now = Instant::now();
    let double_powers = powers_of_x_ct(&single_powers[127], evaluator, ek, sk);
    // println!("double_powers: {:?}", now.elapsed());
    unsafe { decrypt_and_print(&evaluator, &double_powers[254], sk, "double_powers[254]") };

    // change to evaluation for plaintext multiplication
    // now = Instant::now();
    single_powers.iter_mut().for_each(|ct| {
        evaluator.ciphertext_change_representation(ct, Representation::Evaluation);
    });
    // println!(
    //     "single_powers coefficient to evaluation: {:?}",
    //     now.elapsed()
    // );

    let level = 0;
    let q_ctx = evaluator.params().poly_ctx(&PolyType::Q, level);

    // when i = 0, we skip multiplication and cache the result
    let mut left_over_ct = Ciphertext::new(vec![], PolyType::Q, 0);
    let mut sum_ct = Ciphertext::new(vec![], PolyType::Q, 0);

    // now = Instant::now();
    for i in 0..256 {
        // let mut inner_now = Instant::now();
        #[cfg(target_arch = "x86_64")]
        let res_ct = range_fn_fma::optimised_range_fn_fma_hexl(
            &q_ctx,
            &single_powers,
            &constants,
            256 * i,
            level,
        );

        #[cfg(not(target_arch = "x86"))]
        let res_ct = range_fn_fma::optimised_range_fn_fma_u128(
            &q_ctx,
            evaluator.params(),
            &single_powers,
            &constants,
            256 * i,
            level,
        );
        // println!("Inner scalar product {i}: {:?}", inner_now.elapsed());
        // decrypt_and_print(&res_ct, sk, &format!("Inner scalar product {i}"));

        // let mut second_inner_time = Instant::now();
        // cache i == 0
        if i == 0 {
            left_over_ct = res_ct;
            // convert  ct to coefficient form
            evaluator
                .ciphertext_change_representation(&mut left_over_ct, Representation::Coefficient);
        } else if i == 1 {
            // multiply1_lazy returns in evaluation form
            sum_ct = evaluator.mul_lazy(&res_ct, &double_powers[i - 1]);
        } else {
            evaluator.add_assign(
                &mut sum_ct,
                &evaluator.mul_lazy(&res_ct, &double_powers[i - 1]),
            );
        }
        // println!("Mul_lazy + add {i}: {:?}", second_inner_time.elapsed());
    }

    let sum_ct = evaluator.scale_and_round(&mut sum_ct);

    let mut sum_ct = evaluator.relinearize(&sum_ct, ek);
    evaluator.add_assign(&mut sum_ct, &left_over_ct);
    // println!("Outer summation: {:?}", now.elapsed());
    unsafe { decrypt_and_print(&evaluator, &sum_ct, sk, "Outer summation") }

    // implement optimised 1 - sum_ct
    sub_from_one(evaluator.params(), &mut sum_ct, sub_from_one_precompute);
    sum_ct
}

#[cfg(test)]
mod tests {
    use std::{ascii::escape_default, f32::consts::E};

    use super::*;
    use crate::{
        client::gen_pv_exapnd_rtgs,
        optimised::{coefficient_u128_to_ciphertext, sub_from_one_precompute},
        plaintext::powers_of_x_modulus,
        preprocessing::precompute_indices_pts,
        utils::precompute_range_constants,
    };
    use bfv::{BfvParameters, Encoding};
    use rand::thread_rng;

    #[test]
    fn range_fn_works() {
        let params = BfvParameters::default(15, 1 << 8);
        let level = 0;
        let ctx = params.poly_ctx(&PolyType::Q, level);

        let mut rng = thread_rng();
        let constants = precompute_range_constants(&ctx);
        let sub_one_precompute = sub_from_one_precompute(&params, level);

        let sk = SecretKey::random(params.degree, &mut rng);
        let mut m = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);

        let evaluator = Evaluator::new(params);
        let pt = evaluator.plaintext_encode(&m, Encoding::simd(level));
        let mut ct = evaluator.encrypt(&sk, &pt, &mut rng);

        // gen evaluation key
        let ek = EvaluationKey::new(evaluator.params(), &sk, &[0], &[], &[], &mut rng);

        let now = std::time::Instant::now();
        let ct_res = range_fn(&ct, &evaluator, &ek, &constants, &sub_one_precompute, &sk);
        let time = now.elapsed();

        println!(
            "Time: {:?}, Noise: {}",
            time,
            evaluator.measure_noise(&sk, &ct_res)
        );

        let res = evaluator.plaintext_decode(&evaluator.decrypt(&sk, &ct_res), Encoding::default());

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
        let params = BfvParameters::default(5, 1 << 3);
        let sk = SecretKey::random(params.degree, &mut rng);
        let m = vec![3; params.degree];

        let evaluator = Evaluator::new(params);
        let pt = evaluator.plaintext_encode(&m, Encoding::default());
        let ct = evaluator.encrypt(&sk, &pt, &mut rng);
        let ek = EvaluationKey::new(evaluator.params(), &sk, &[0], &[], &[], &mut rng);

        {
            for _ in 0..1 {
                powers_of_x_ct(&ct, &evaluator, &ek, &sk);
            }
        }

        let now = std::time::Instant::now();
        let powers_ct = powers_of_x_ct(&ct, &evaluator, &ek, &sk);
        println!("Time = {:?}", now.elapsed());

        let res_values_mod = powers_of_x_modulus(3, &evaluator.params().plaintext_modulus_op);

        izip!(powers_ct.iter(), res_values_mod.iter()).for_each(|(pct, v)| {
            dbg!(evaluator.measure_noise(&sk, pct));
            let r = evaluator.plaintext_decode(&evaluator.decrypt(&sk, pct), Encoding::default());
            r.iter().for_each(|r0| {
                assert!(r0 == v);
            });
        });
    }

    #[test]
    fn even_powers_of_x_ct_works() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(5, 1 << 3);
        let sk = SecretKey::random(params.degree, &mut rng);
        let m = vec![3; params.degree];

        let evaluator = Evaluator::new(params);
        let pt = evaluator.plaintext_encode(&m, Encoding::default());
        let ct = evaluator.encrypt(&sk, &pt, &mut rng);
        let ek = EvaluationKey::new(evaluator.params(), &sk, &[0], &[], &[], &mut rng);

        {
            for _ in 0..1 {
                even_powers_of_x_ct(&ct, &evaluator, &ek, &sk);
            }
        }

        let now = std::time::Instant::now();
        let powers_ct = even_powers_of_x_ct(&ct, &evaluator, &ek, &sk);
        println!("Time = {:?}", now.elapsed());

        let res_values_mod = powers_of_x_modulus(3, &evaluator.params().plaintext_modulus_op);

        izip!(res_values_mod.iter().skip(1).step_by(2), powers_ct.iter()).for_each(|(v, v_ct)| {
            dbg!(evaluator.measure_noise(&sk, v_ct));
            let r = evaluator.plaintext_decode(&evaluator.decrypt(&sk, v_ct), Encoding::default());
            r.iter().for_each(|r0| {
                assert!(r0 == v);
            });
        });
    }
}
