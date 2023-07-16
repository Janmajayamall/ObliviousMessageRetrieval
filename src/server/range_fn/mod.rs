use crate::optimised::sub_from_one;
use crate::server::powers_x::evaluate_powers;
use crate::time_it;
use bfv::{
    Ciphertext, EvaluationKey, Evaluator, GaloisKey, Plaintext, Poly, PolyContext, PolyType,
    RelinearizationKey, Representation, SecretKey,
};
use itertools::Itertools;
use ndarray::Array2;

pub mod range_fn_fma;

/// Helper function that returns sqaure of the `ct`
fn ciphertext_square_and_relin(
    evaluator: &Evaluator,
    ek: &EvaluationKey,
    ct: &Ciphertext,
) -> Ciphertext {
    evaluator.relinearize(&evaluator.mul(ct, ct), ek)
}

/// Helper function to equally distribute 255 m loop iterations between available threads.
/// Functions starts with range [1, 256) recursively divides the range in half until the distance
/// of range is <= set_len, after which processes loop iterations in the range.
/// set_len is equal to `ceil(255/no_of_threads)`. Thus `set_len` defines appropriate number of iterations
/// that must be assigned to each thread.
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
        println!("{start} {end} {}", end - start);

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

            // Don't add in first iteration, that is when i == start, since sum is empty
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
    // Calculate only even powers in range [1,256]
    let mut k_powers = vec![placeholder.clone(); 128];
    // calcuate x^2 separately to simplify the code
    k_powers[0] = ciphertext_square_and_relin(evaluator, ek, ct);
    for base in [4, 8, 16, 32, 64, 128, 256] {
        k_powers[(base >> 1) - 1] =
            ciphertext_square_and_relin(evaluator, ek, &k_powers[(base >> 2) - 1]);
    }

    // calculate all powers in range [1,255]
    let mut m_powers = vec![placeholder.clone(); 255];
    // since m^1 = x^256, set k[127] at index 0.
    // Although m_powers[0] is equal to k_powers[127] it is neccessary to have two separate copies since we require the same ciphertext
    // in different representations. k_powers[127] must be in `Evaluation` representation for efficient plaintext multiplication in inner loop
    //  and m_powers[0] must `Coefficient` representation for efficient evaluation of m_powers and outer loop muliplication in second m loop interation.
    m_powers[0] = {
        // We cannot directly call clone, since `mod_down_next` does not free up memory allocated to already dropped rows of fresh ciphertext.
        // Calling clone will clone unecessary values causing unecessary memory allocations. Instead we will have to call
        // `to_owned` on coefficient arrays owned by polynomials inside ciphertext, to make sure no additional space
        // is occupied by not in use rows.
        let c_vec = k_powers[127]
            .c_ref()
            .iter()
            .map(|p| {
                let coeffs = p.coefficients().to_owned();
                Poly::new(coeffs, p.representation().clone())
            })
            .collect_vec();

        Ciphertext::new(c_vec, k_powers[127].poly_type(), k_powers[127].level())
    };

    rayon::join(
        || {
            // For k_powers we only need to calculate even powers in the range [1,256]. Recall that
            // all even numbers in a given range can be obtained by multiplying 2 by all numbers in half of the range.
            // For example, all even numbers in range [1,256] can be obtained by multiplying all values in
            // range [1,128] by 2. Thus to calculate only even powers in range [0, 256] we calculate all powers
            // of `a` in range [1,128] where `a=x^2`.
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
    k_powers.iter_mut().for_each(|ct| {
        evaluator.ciphertext_change_representation(ct, Representation::Evaluation);
    });

    let level = 0;
    let q_ctx = evaluator.params().poly_ctx(&PolyType::Q, level);

    let threads = rayon::current_num_threads() as f64;
    // k loop needs to run 255 times. `set_len` fairly distributes 255 iterations among available threads.
    let set_len = (255.0 / threads).ceil() as usize;

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

        // change representation to Coefficient to stay consistent with output of rest of the m loops
        evaluator.ciphertext_change_representation(&mut res_ct, Representation::Coefficient);
        res_ct
    };

    // process_m_loop processes m^th loop for values in range [start, end)
    let mut sum_ct = process_m_loop(
        evaluator, &q_ctx, level, constants, &k_powers, &m_powers, set_len, 1, 256,
    );

    // `sum_ct` is in PQ basis, instead of usual Q basis. Call `scale_and_round` to scale
    // chiphertext by P/t and switch to Q basis.
    let sum_ct = evaluator.scale_and_round(&mut sum_ct);
    let mut sum_ct = evaluator.relinearize(&sum_ct, ek);

    // add output of first loop, processed separately, to summation of output of rest of the loops
    evaluator.add_assign(&mut sum_ct, &m_0th_loop);

    sub_from_one(evaluator.params(), &mut sum_ct, sub_from_one_precompute);
    sum_ct
}

/// Helper function to call range_fn 4 times with multiple threads.
/// Creates and installs thread pools for each range_fn such that
/// global thread pool is distributed equally among the 4 range_fn calls.
pub fn range_fn_4_times(
    decrypted_cts: &[Ciphertext],
    evaluator: &Evaluator,
    ek: &EvaluationKey,
    constants: &Array2<u64>,
    sub_from_one_precompute: &[u64],
    sk: &SecretKey,
) -> ((Ciphertext, Ciphertext), (Ciphertext, Ciphertext)) {
    let threads = (rayon::current_num_threads() as f64 / 4.0).ceil() as usize;

    macro_rules! install_pool_and_call_range_fn {
        ($index:literal) => {{
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();

            pool.install(|| {
                // println!("Running {} with {threads} threads...", $index);
                range_fn(
                    &decrypted_cts[$index],
                    evaluator,
                    ek,
                    constants,
                    sub_from_one_precompute,
                    sk,
                )
            })
        }};
    }
    let ranged_cts = rayon::join(
        || {
            rayon::join(
                || install_pool_and_call_range_fn!(0),
                || install_pool_and_call_range_fn!(1),
            )
        },
        || {
            rayon::join(
                || install_pool_and_call_range_fn!(2),
                || install_pool_and_call_range_fn!(3),
            )
        },
    );

    ranged_cts
}

#[cfg(test)]
mod tests {
    use core::time;
    use itertools::izip;
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
        let params = BfvParameters::default(15, 1 << 3);
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
            .num_threads(10)
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
