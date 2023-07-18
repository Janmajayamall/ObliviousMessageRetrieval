use crate::server::powers_x::evaluate_powers;
use crate::{level_down, time_it};
use crate::{optimised::sub_from_one, print_noise};
use bfv::{
    Ciphertext, EvaluationKey, Evaluator, GaloisKey, Plaintext, Poly, PolyContext, PolyType,
    RelinearizationKey, Representation, SecretKey,
};
use itertools::Itertools;
use ndarray::Array2;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

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

    // Calculate base powers for k_powers. There's no harm in doing this here since
    // evaluate_powers calculates base powers serially as well.
    // The intention with doing this here is to evaluate k_powers and m_powers in parallel using `join`.
    // Calculate only even powers in range [1,256]
    let mut k_powers = vec![placeholder.clone(); 128];
    // calcuate x^2 separately to simplify the code
    k_powers[0] = ciphertext_square_and_relin(evaluator, ek, ct);

    level_down!( evaluator.mod_down_next(&mut k_powers[0]););

    print_noise!(
        println!(
            "k_powers base 2 noise: {}",
            evaluator.measure_noise(sk, &k_powers[0])
        );
    );

    for base in [4, 8, 16, 32, 64, 128, 256] {
        k_powers[(base >> 1) - 1] =
            ciphertext_square_and_relin(evaluator, ek, &k_powers[(base >> 2) - 1]);

        level_down!(if base == 8 || base == 32 || base == 64 || base == 256 {
            evaluator.mod_down_next(&mut k_powers[(base >> 1) - 1]);
        });

        print_noise!(
            println!(
                "k_powers base {base} noise: {}",
                evaluator.measure_noise(sk, &k_powers[(base >> 1) - 1])
            );
        );
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
            evaluate_powers(evaluator, ek, 2, 4, &mut k_powers, true, sk);
            evaluate_powers(evaluator, ek, 4, 8, &mut k_powers, true, sk);
            evaluate_powers(evaluator, ek, 8, 16, &mut k_powers, true, sk);
            evaluate_powers(evaluator, ek, 16, 32, &mut k_powers, true, sk);
            evaluate_powers(evaluator, ek, 32, 64, &mut k_powers, true, sk);
            evaluate_powers(evaluator, ek, 64, 128, &mut k_powers, true, sk);
        },
        || {
            evaluate_powers(evaluator, ek, 2, 4, &mut m_powers, false, sk);
            evaluate_powers(evaluator, ek, 4, 8, &mut m_powers, false, sk);
            evaluate_powers(evaluator, ek, 8, 16, &mut m_powers, false, sk);
            evaluate_powers(evaluator, ek, 16, 32, &mut m_powers, false, sk);
            evaluate_powers(evaluator, ek, 32, 64, &mut m_powers, false, sk);
            evaluate_powers(evaluator, ek, 64, 128, &mut m_powers, false, sk);
            evaluate_powers(evaluator, ek, 128, 256, &mut m_powers, false, sk);
        },
    );

    // don't re-declare match level inside `level_down` since the macro creates a new scope
    let mut match_level = 0;
    level_down!(
        // lose another level from m_powers to reduce noise for range (128,255]. Then
        // match level of all ciphertexts in m_powers and k_powers
        match_level = m_powers.last().unwrap().level() + 1;
        assert!(match_level == 10);

        k_powers.par_iter_mut().for_each(|mut c| {
            // mod down before changing representation
            evaluator.mod_down_level(&mut c, match_level);

            // change k_powers to `Evaluation` for efficient plaintext multiplication
            evaluator.ciphertext_change_representation(&mut c, Representation::Evaluation);
        });

        m_powers.par_iter_mut().for_each(|mut c| {
            // mod down before changing representation
            evaluator.mod_down_level(&mut c, match_level);
        });
    );

    #[cfg(not(feature = "level"))]
    {
        k_powers.par_iter_mut().for_each(|mut c| {
            // change k_powers to `Evaluation` for efficient plaintext multiplication
            evaluator.ciphertext_change_representation(&mut c, Representation::Evaluation);
        });
    }

    let level = match_level;

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
/// Assumes that total_threads is power of 2.
pub fn range_fn_4_times(
    decrypted_cts: &[Ciphertext],
    evaluator: &Evaluator,
    ek: &EvaluationKey,
    constants: &Array2<u64>,
    sub_from_one_precompute: &[u64],
    sk: &SecretKey,
) -> ((Ciphertext, Ciphertext), (Ciphertext, Ciphertext)) {
    // We assume that total no of threads is power of two
    let total_threads = rayon::current_num_threads();

    macro_rules! call_range {
        ($index:literal) => {
            range_fn(
                &decrypted_cts[$index],
                evaluator,
                ek,
                constants,
                sub_from_one_precompute,
                sk,
            )
        };
    }

    if total_threads == 1 {
        // there's only 1 thread. Process all 4 calls serially.
        let ranged_cts = (
            (call_range!(0), call_range!(1)),
            (call_range!(2), call_range!(3)),
        );
        return ranged_cts;
    } else {
        // Divide all threads among 4 range_fn calls. Assuming total_threads is power of 2, `threads_by_4`
        // will be >= 1 whenever total_threads != 2. When thread_by_4 >= 1 we can easily assign `threads_by_4` threads
        // to each of the 4 range_fn calls and process them in parallel. When thread_by_4 == 0 (ie total_thread = 2)
        // we need to group two range_fn calls together and assign a single thread to each of the two group.
        let threads_by_4 = total_threads / 4;
        // As explained, threads to install will be 1 if there are only 2 threads in total and 4 range_fn calls
        // are grouped into 2 of 2. Otherwise install as many as `threads_by_4` threads.
        let threads = if threads_by_4 == 0 { 1 } else { threads_by_4 };

        macro_rules! install_pool_and_call_range_fn {
            ($block:tt) => {{
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();

                pool.install(|| {
                    // println!("Running {} with {threads} threads...", $index);
                    $block
                })
            }};
        }

        let ranged_cts = if threads_by_4 == 0 {
            // case when total_threads = 2
            rayon::join(
                || install_pool_and_call_range_fn!((call_range!(0), call_range!(1))),
                || install_pool_and_call_range_fn!((call_range!(2), call_range!(3))),
            )
        } else {
            // case when total_threads >= 4
            rayon::join(
                || {
                    rayon::join(
                        || install_pool_and_call_range_fn!({ call_range!(0) }),
                        || install_pool_and_call_range_fn!({ call_range!(1) }),
                    )
                },
                || {
                    rayon::join(
                        || install_pool_and_call_range_fn!({ call_range!(2) }),
                        || install_pool_and_call_range_fn!({ call_range!(3) }),
                    )
                },
            )
        };

        return ranged_cts;
    }
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
        utils::{generate_bfv_parameters, precompute_range_constants},
    };
    use bfv::{BfvParameters, Encoding};
    use rand::thread_rng;

    #[test]
    fn range_fn_works() {
        let params = generate_bfv_parameters();

        #[cfg(feature = "level")]
        let precompute_level = 10;

        #[cfg(not(feature = "level"))]
        let precompute_level = 0;

        let ctx = params.poly_ctx(&PolyType::Q, precompute_level);

        let mut rng = thread_rng();
        let constants = precompute_range_constants(&ctx);
        let sub_one_precompute = sub_from_one_precompute(&params, precompute_level);

        let sk = SecretKey::random(params.degree, &mut rng);
        let mut m = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);

        let evaluator = Evaluator::new(params);
        let pt = evaluator.plaintext_encode(&m, Encoding::simd(0));
        let mut ct = evaluator.encrypt(&sk, &pt, &mut rng);

        evaluator.ciphertext_change_representation(&mut ct, Representation::Evaluation);

        unsafe { evaluator.add_noise(&mut ct, 40) };
        dbg!(evaluator.measure_noise(&sk, &ct));

        // gen evaluation key
        let ek = EvaluationKey::new(
            evaluator.params(),
            &sk,
            &(0..12).into_iter().collect_vec(),
            &[],
            &[],
            &mut rng,
        );

        // limit to single thread
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
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
