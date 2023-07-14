use bfv::{
    BfvParameters, Ciphertext, EvaluationKey, Evaluator, GaloisKey, Plaintext, Poly, PolyType,
    RelinearizationKey, Representation, SecretKey,
};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

/// Calculates all powers of `x` for range [1,256) using binary exponentiation
///
/// Note: not used anywhere. Here just for testing.
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

                    calculated[base_deg - 1] = 1;

                    // mul_count += 1;
                }
            }
        }
    }
    // dbg!(mul_count);

    values
}

/// Calcultes power of x for range [start, end) in parallel. Assumes that all
/// powers in range [0, start) are pre-calculated and stored in `calculated`.
/// Assumes that `start` and `end=start*2` are power of 2.
///
/// Set `bases` to true if `base` power for the range has already calculated and
/// stored in `calculated`.
///
/// If base is not calculated, then function performs end-start ciphertext multiplications.
/// Otherwise performs end-start-1 ciphertext multiplications.
///
/// To understand the function, consider the base power as `x^start`. We assume that
/// `start` is always a power of 2. All values in range [start, end) can be calculated
/// by adding base with some value in range [0,start). For example, if start is 128
/// (b10000000) then to calculate 250 (b 1 1111010), we write it as sum of 128 +
/// 122 (b 0 1111010). Thus, to calculate x^250 by multiply `x^128 * x^122`.
pub fn evaluate_powers(
    evaluator: &Evaluator,
    ek: &EvaluationKey,
    start: usize,
    end: usize,
    calculated: &mut [Ciphertext],
    bases: bool,
    cores: usize,
) {
    // start is always a power of two. Hence, start-1 equals the mask to extract log2(start) bits
    let mask = start - 1;

    // To calculate values in range (start, start*2) we require value for start, that is the base
    if !bases {
        let tmp = evaluator.mul(&calculated[start / 2 - 1], &calculated[start / 2 - 1]);
        calculated[start - 1] = evaluator.relinearize(&tmp, ek);
    }

    // To avoid running into borrow issues later, split `calculated` at `start+1` where the second chunk will
    // be mutated
    let (done, pending) = calculated.split_at_mut(start - 1 + 1);

    // Since we only need to calculate for range [start+1, end-1], slice off the rest
    let pending = &mut pending[..(end - 1 - start)];
    let size = (pending.len() as f64 / cores as f64).ceil() as usize;

    pending.par_iter_mut().enumerate().for_each(|(index, v)| {
        // calculate real_index for power to figure out which other power to multiply base with.
        // For ex, if real_index = 5 (ie x^5) then we must multiply x^4 with x^1 since x^4 is the base
        // and 1 equals to `5 & mask` (ie 101 & 011 = 1).
        let real_index = index + start + 1;
        let tmp = evaluator.mul(done.last().unwrap(), &done[(real_index & mask) - 1]);
        *v = evaluator.relinearize(&tmp, ek);
    });

    // let pending_chunks = pending.par_chunks_mut(size);
    // pending_chunks
    //     .enumerate()
    //     .into_par_iter()
    //     .for_each(|(index, chunk)| {
    //         chunk.iter_mut().enumerate().for_each(|(chunk_index, v)| {
    //             // calculate real_index of the value to figure out the mask
    //             let real_index = size * index + chunk_index + start + 1;
    //             let tmp = evaluator.mul(done.last().unwrap(), &done[(real_index & mask) - 1]);
    //             *v = evaluator.relinearize(&tmp, ek);
    //         });
    //     });
}

mod tests {
    use crate::plaintext::powers_of_x_modulus;

    use super::*;
    use bfv::{BfvParameters, Encoding};
    use itertools::izip;
    use rand::thread_rng;

    #[test]
    fn test_evaluate_powers() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(15, 1 << 15);
        let sk = SecretKey::random(params.degree, &mut rng);
        let m = vec![3; params.degree];

        let evaluator = Evaluator::new(params);
        let pt = evaluator.plaintext_encode(&m, Encoding::default());
        let ct = evaluator.encrypt(&sk, &pt, &mut rng);
        let ek = EvaluationKey::new(evaluator.params(), &sk, &[0], &[], &[], &mut rng);
        let dummy = Ciphertext::new(vec![], PolyType::Q, 0);

        let cores = 4;

        // warm up
        {
            let mut calculated = vec![dummy.clone(); 255];
            calculated[0] = ct.clone();
            for _ in 0..10 {
                evaluate_powers(&evaluator, &ek, 2, 4, &mut calculated, false, cores);
            }
        }

        let now = std::time::Instant::now();
        let mut calculated = vec![dummy.clone(); 255];
        calculated[0] = ct;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();

        pool.install(|| {
            evaluate_powers(&evaluator, &ek, 2, 4, &mut calculated, false, cores);
            evaluate_powers(&evaluator, &ek, 4, 8, &mut calculated, false, cores);
            evaluate_powers(&evaluator, &ek, 8, 16, &mut calculated, false, cores);
            evaluate_powers(&evaluator, &ek, 16, 32, &mut calculated, false, cores);
            evaluate_powers(&evaluator, &ek, 32, 64, &mut calculated, false, cores);
            evaluate_powers(&evaluator, &ek, 64, 128, &mut calculated, false, cores);
            evaluate_powers(&evaluator, &ek, 128, 256, &mut calculated, false, cores);
        });
        println!("Time: {:?}", now.elapsed());

        let res_values_mod = powers_of_x_modulus(3, &evaluator.params().plaintext_modulus_op, 255);
        izip!(calculated.iter(), res_values_mod.iter()).for_each(|(pct, v)| {
            // dbg!(evaluator.measure_noise(&sk, pct));
            let r = evaluator.plaintext_decode(&evaluator.decrypt(&sk, pct), Encoding::default());
            r.iter().for_each(|r0| {
                assert!(r0 == v);
            });
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

        // {
        //     for _ in 0..1 {
        //         powers_of_x_ct(&ct, &evaluator, &ek, &sk);
        //     }
        // }

        let now = std::time::Instant::now();
        let powers_ct = powers_of_x_ct(&ct, &evaluator, &ek, &sk);
        println!("Time = {:?}", now.elapsed());

        let res_values_mod = powers_of_x_modulus(3, &evaluator.params().plaintext_modulus_op, 255);

        izip!(powers_ct.iter(), res_values_mod.iter()).for_each(|(pct, v)| {
            dbg!(evaluator.measure_noise(&sk, pct));
            let r = evaluator.plaintext_decode(&evaluator.decrypt(&sk, pct), Encoding::default());
            r.iter().for_each(|r0| {
                assert!(r0 == v);
            });
        });
    }
}
