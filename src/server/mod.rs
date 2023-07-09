use crate::optimised::{coefficient_u128_to_ciphertext, fma_reverse_u128_poly};
use crate::preprocessing::{precompute_expand_32_roll_pt, procompute_expand_roll_pt};
use crate::server::powers_x::evaluate_powers;
use crate::utils::decrypt_and_print;
use crate::{
    optimised::{barret_reduce_coefficients_u128, optimised_pvw_fma_with_rot, sub_from_one},
    pvw::PvwParameters,
};
use crate::{time_it, LEVELLED};
use bfv::{
    BfvParameters, Ciphertext, EvaluationKey, Evaluator, GaloisKey, Plaintext, Poly, PolyType,
    RelinearizationKey, Representation, SecretKey,
};
use itertools::{izip, Itertools};
use ndarray::{s, Array2};
use powers_x::{even_powers_of_x_ct, powers_of_x_ct};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::{collections::HashMap, sync::Arc, time::Instant};

pub mod phase2;
pub mod powers_x;
pub mod range_fn_fma;

fn ciphertext_square_and_relin(
    evaluator: &Evaluator,
    ek: &EvaluationKey,
    ct: &Ciphertext,
) -> Ciphertext {
    evaluator.relinearize(&evaluator.mul(ct, ct), ek)
}

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

pub fn range_fn(
    ct: &Ciphertext,
    evaluator: &Evaluator,
    ek: &EvaluationKey,
    constants: &Array2<u64>,
    sub_from_one_precompute: &[u64],
    sk: &SecretKey,
) -> Ciphertext {
    let placeholder = Ciphertext::new(vec![], PolyType::Q, 0);

    let cores = 1;

    // Calculate base powers for k_powers. There's no harm in doing this here since
    // evaluate_powers calculates base powers serially as well.
    // The intention with doing this here is to evaluate k_powers and m_powers in parallel.
    // calculate only even powers in range [1,256]
    let mut k_powers = vec![placeholder.clone(); 128];
    // calcuate x^2 separately to make look code simpler
    time_it!("k_powers",
        k_powers[0] = ciphertext_square_and_relin(evaluator, ek, ct);
        for base in [4, 8, 16, 32, 64, 128, 256] {
            k_powers[(base >> 1) - 1] =
                ciphertext_square_and_relin(evaluator, ek, &k_powers[(base >> 2) - 1]);
        }
        evaluate_powers(evaluator, ek, 2, 4, &mut k_powers, true, cores);
        evaluate_powers(evaluator, ek, 4, 8, &mut k_powers, true, cores);
        evaluate_powers(evaluator, ek, 8, 16, &mut k_powers, true, cores);
        evaluate_powers(evaluator, ek, 16, 32, &mut k_powers, true, cores);
        evaluate_powers(evaluator, ek, 32, 64, &mut k_powers, true, cores);
        evaluate_powers(evaluator, ek, 64, 128, &mut k_powers, true, cores);
    );

    time_it!("m_powers",
       // calculate all powers in range [1,255]
        let mut m_powers = vec![placeholder.clone(); 255];
        // since m^1 = x^256, set k[127] at index 0.
        // Although m_powers[0] is equal to k_powers[127] it is neccessary to have two separate copies since we require the same ciphertext
        // into different representations. k_powers[127] must be in `Evaluation` for efficient plaintext multiplication in inner loop and m_powers[0] must
        // `Coefficient` for efficient evaluation of powers and outer loop muliplication corresponding to second iteration.
        m_powers[0] = {
            // We cannot directly call clone, since `mod_down_next` does not free up memory allocated to dropped rows in beggining.
            // Calling clone will clone unecessary values causing unecessary memory allocations. Instead we will have to call
            // `to_owned` on coefficient arrays owned by polynomials inside ciphertext ourselves, to make sure no additional space
            // is occupied by not in use rows.
            let c_vec = k_powers[127]
                .c_ref()
                .iter()
                .map(|p| {
                    let coeffs = p.coefficients.to_owned();
                    Poly::new(coeffs, p.representation.clone())
                })
                .collect_vec();

            Ciphertext::new(c_vec, k_powers[127].poly_type(), k_powers[127].level())
        };
        evaluate_powers(evaluator, ek, 2, 4, &mut m_powers, false, cores);
        evaluate_powers(evaluator, ek, 4, 8, &mut m_powers, false, cores);
        evaluate_powers(evaluator, ek, 8, 16, &mut m_powers, false, cores);
        evaluate_powers(evaluator, ek, 16, 32, &mut m_powers, false, cores);
        evaluate_powers(evaluator, ek, 32, 64, &mut m_powers, false, cores);
        evaluate_powers(evaluator, ek, 64, 128, &mut m_powers, false, cores);
        evaluate_powers(evaluator, ek, 128, 256, &mut m_powers, false, cores);
    );

    // change k_powers to `Evaluation` for efficient plaintext multiplication
    time_it!("k_powers change representation",
         k_powers.iter_mut().for_each(|ct| {
            evaluator.ciphertext_change_representation(ct, Representation::Evaluation);
        });
    );

    let level = 0;
    let q_ctx = evaluator.params().poly_ctx(&PolyType::Q, level);

    // when i = 0, we skip multiplication and cache the result
    let mut left_over_ct = Ciphertext::new(vec![], PolyType::Q, 0);
    let mut sum_ct = Ciphertext::new(vec![], PolyType::Q, 0);

    time_it!(
        "Loops",
        for i in 0..256 {
            // let mut inner_now = Instant::now();
            #[cfg(target_arch = "x86_64")]
            let res_ct = range_fn_fma::optimised_range_fn_fma_hexl(
                &q_ctx,
                &k_powers,
                &constants,
                256 * i,
                level,
            );

            #[cfg(not(target_arch = "x86"))]
            let res_ct = range_fn_fma::optimised_range_fn_fma_u128(
                &q_ctx,
                evaluator.params(),
                &k_powers,
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
                evaluator.ciphertext_change_representation(
                    &mut left_over_ct,
                    Representation::Coefficient,
                );
            } else if i == 1 {
                // multiply1_lazy returns in evaluation form
                sum_ct = evaluator.mul_lazy(&res_ct, &m_powers[i - 1]);
            } else {
                evaluator.add_assign(&mut sum_ct, &evaluator.mul_lazy(&res_ct, &m_powers[i - 1]));
            }
            // println!("Mul_lazy + add {i}: {:?}", second_inner_time.elapsed());
        }

    let sum_ct = evaluator.scale_and_round(&mut sum_ct);

    let mut sum_ct = evaluator.relinearize(&sum_ct, ek);
    evaluator.add_assign(&mut sum_ct, &left_over_ct);
    );

    // // implement optimised 1 - sum_ct
    sub_from_one(evaluator.params(), &mut sum_ct, sub_from_one_precompute);
    sum_ct
}

#[cfg(test)]
mod tests {
    use core::time;
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
        let params = BfvParameters::default(15, 1 << 15);
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

        // limit to single thread
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .unwrap();

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
}
