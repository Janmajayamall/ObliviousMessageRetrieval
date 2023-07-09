use crate::optimised::{coefficient_u128_to_ciphertext, fma_reverse_u128_poly};
use crate::preprocessing::{precompute_expand_32_roll_pt, procompute_expand_roll_pt};
use crate::server::powers_x::evaluate_powers;
use crate::time_it;
use crate::utils::decrypt_and_print;
use crate::{
    optimised::{barret_reduce_coefficients_u128, optimised_pvw_fma_with_rot, sub_from_one},
    pvw::PvwParameters,
};
use bfv::{
    BfvParameters, Ciphertext, EvaluationKey, Evaluator, GaloisKey, Plaintext, Poly, PolyContext,
    PolyType, RelinearizationKey, Representation, SecretKey,
};
use core::time;
use itertools::{izip, Itertools};
use ndarray::{s, Array2};
use powers_x::{even_powers_of_x_ct, powers_of_x_ct};
use rand_chacha::rand_core::le;
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
    // The intention with doing this here is to evaluate k_powers and m_powers in parallel using `join`.
    // calculate only even powers in range [1,256]
    let mut k_powers = vec![placeholder.clone(); 128];
    // calcuate x^2 separately to make code look simpler
    k_powers[0] = ciphertext_square_and_relin(evaluator, ek, ct);
    for base in [4, 8, 16, 32, 64, 128, 256] {
        k_powers[(base >> 1) - 1] =
            ciphertext_square_and_relin(evaluator, ek, &k_powers[(base >> 2) - 1]);
    }

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

    //
    rayon::join(
        || {
            evaluate_powers(evaluator, ek, 2, 4, &mut k_powers, true, cores);
            evaluate_powers(evaluator, ek, 4, 8, &mut k_powers, true, cores);
            evaluate_powers(evaluator, ek, 8, 16, &mut k_powers, true, cores);
            evaluate_powers(evaluator, ek, 16, 32, &mut k_powers, true, cores);
            evaluate_powers(evaluator, ek, 32, 64, &mut k_powers, true, cores);
            evaluate_powers(evaluator, ek, 64, 128, &mut k_powers, true, cores);
        },
        || {
            evaluate_powers(evaluator, ek, 2, 4, &mut m_powers, false, cores);
            evaluate_powers(evaluator, ek, 4, 8, &mut m_powers, false, cores);
            evaluate_powers(evaluator, ek, 8, 16, &mut m_powers, false, cores);
            evaluate_powers(evaluator, ek, 16, 32, &mut m_powers, false, cores);
            evaluate_powers(evaluator, ek, 32, 64, &mut m_powers, false, cores);
            evaluate_powers(evaluator, ek, 64, 128, &mut m_powers, false, cores);
            evaluate_powers(evaluator, ek, 128, 256, &mut m_powers, false, cores);
        },
    );

    // change k_powers to `Evaluation` for efficient plaintext multiplication
    time_it!("k_powers change representation",
         k_powers.iter_mut().for_each(|ct| {
            evaluator.ciphertext_change_representation(ct, Representation::Evaluation);
        });
    );

    let level = 0;
    let q_ctx = evaluator.params().poly_ctx(&PolyType::Q, level);

    let threads = rayon::current_num_threads() as f64;
    // k loop needs to run 255 times. `set_len` fairly distributes 255 iterations among available threads.
    let set_len = (255.0 / threads).ceil() as usize;

    fn process_m_loop(
        evaluator: &Evaluator,
        q_ctx: &PolyContext<'_>,
        level: usize,
        constants: &Array2<u64>,
        k_powers: &[Ciphertext],
        m_powers: &[Ciphertext],
        set_len: usize,
        start: usize,
        end: usize,
    ) -> Ciphertext {
        // process k loop when range is either equal or smaller than set_len
        if end - start <= set_len {
            println!("{start} {end}");

            let mut sum = Ciphertext::new(vec![], PolyType::Q, 0);
            for i in start..end {
                #[cfg(target_arch = "x86_64")]
                let res_ct = range_fn_fma::optimised_range_fn_fma_hexl(
                    &q_ctx,
                    &k_powers,
                    &constants,
                    256 * i,
                    level,
                );

                #[cfg(not(target_arch = "x86_64"))]
                let res_ct = range_fn_fma::optimised_range_fn_fma_u128(
                    &q_ctx,
                    evaluator.params(),
                    &k_powers,
                    &constants,
                    256 * i,
                    level,
                );

                // `res_ct` is in `Evaluation` and `m_powers` is in `Coefficient` representation. This works since
                // `mul_lazy` accepts different representations if lhs is Evaluation and rhs is in Coefficient
                // because the performance is equivalent to when both are in `Coefficient`.
                // It's ok to index by `i - 1` here since `start` is never 0
                let product = evaluator.mul_lazy(&res_ct, &m_powers[i - 1]);

                // Don't add in first iteration when i == start since sum is empty
                if i == start {
                    sum = product;
                } else {
                    evaluator.add_assign(&mut sum, &product);
                }
            }

            sum
        } else {
            let mid = (start + end) / 2;
            let (mut ct0, ct1) = rayon::join(
                || {
                    process_m_loop(
                        evaluator, q_ctx, level, constants, k_powers, m_powers, set_len, start, mid,
                    )
                },
                || {
                    process_m_loop(
                        evaluator, q_ctx, level, constants, k_powers, m_powers, set_len, mid, end,
                    )
                },
            );
            evaluator.add_assign(&mut ct0, &ct1);
            ct0
        }
    }

    time_it!("Loops",
        // calculate degree [1..256], ie the first k loop, seprarately since it does not
        // needs to be multiplied with any m_power and would remain in Q basis.
        let m_0th_loop = {
            #[cfg(target_arch = "x86_64")]
            let mut res_ct =
                range_fn_fma::optimised_range_fn_fma_hexl(&q_ctx, &k_powers, &constants, 0, level);

            #[cfg(not(target_arch = "x86_64"))]
            let mut res_ct = range_fn_fma::optimised_range_fn_fma_u128(
                &q_ctx,
                evaluator.params(),
                &k_powers,
                &constants,
                0,
                level,
            );

            // change representation to Coefficient to stay consistent with output of rest of the k loops
            evaluator.ciphertext_change_representation(&mut res_ct, Representation::Coefficient);
            res_ct
        };

        // process_m_loop processes m_th loop for values in range [start, end)
        let mut sum_ct = process_m_loop(
            evaluator, &q_ctx, level, constants, &k_powers, &m_powers, set_len, 1, 256,
        );

        // `sum_ct` is in PQ basis, instead of usual Q basis. Call `scale_and_round` to scale
        // chiphertext by P/t and switch to Q basis.
        let sum_ct = evaluator.scale_and_round(&mut sum_ct);
        let mut sum_ct = evaluator.relinearize(&sum_ct, ek);

        // add output of first loop, processed separately, to summation of output of rest of the loops
        evaluator.add_assign(&mut sum_ct, &m_0th_loop);
    );

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

    #[test]
    fn trial() {
        fn rec(set_len: usize, start: usize, end: usize) {
            if end - start <= set_len {
                println!("{start} {end}");
                //process
            } else {
                let mid = (start + end) / 2;
                rayon::join(
                    || {
                        rec(set_len, start, mid);
                    },
                    || rec(set_len, mid, end),
                );
            }
        }

        let cores = 32 as f64;
        let set_len = (256.0 / cores).ceil() as usize;

        rec(set_len, 1, 256);
    }
}
