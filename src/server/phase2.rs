use crate::optimised::{coefficient_u128_to_ciphertext, fma_reverse_u128_poly};
use crate::preprocessing::{
    precompute_expand_32_roll_pt, procompute_expand_roll_pt, read_indices_poly,
};
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
use rand_chacha::rand_core::le;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSlice;
use std::{collections::HashMap, sync::Arc, time::Instant};

pub fn fma_poly(
    ek: &EvaluationKey,
    evaluator: &Evaluator,
    cts: &[Ciphertext],
    polys: &[Poly],
    sk: &SecretKey,
    level: usize,
) -> (Array2<u128>, Array2<u128>) {
    let coeff_shape = cts
        .first()
        .unwrap()
        .c_ref()
        .first()
        .unwrap()
        .coefficients()
        .shape();

    let mut res0 = Array2::<u128>::zeros((coeff_shape[0], coeff_shape[1]));
    let mut res1 = Array2::<u128>::zeros((coeff_shape[0], coeff_shape[1]));

    izip!(cts.iter(), polys.iter()).for_each(|(o, i)| {
        // indices
        fma_reverse_u128_poly(&mut res0, &o.c_ref()[0], i);
        fma_reverse_u128_poly(&mut res1, &o.c_ref()[1], i);
    });

    (res0, res1)
}

pub fn add_u128_array(a: &mut Array2<u128>, b: &Array2<u128>) {
    izip!(a.outer_iter_mut(), b.outer_iter(),).for_each(|(mut ac, bc)| {
        izip!(ac.iter_mut(), bc.iter()).for_each(|(v0, v1)| {
            *v0 += *v1;
        });
    });
}

pub fn pv_expand_batch(
    ek: &EvaluationKey,
    evaluator: &Evaluator,
    pts_4_roll: &[Plaintext],
    pts_1_roll: &[Plaintext],
    pv_ct: &Ciphertext,
    pts_32: &[Plaintext],
    sk: &SecretKey,
) -> Vec<Ciphertext> {
    let now = std::time::Instant::now();

    let degree = evaluator.params().degree as isize;
    let mut ones = vec![];
    pts_32.iter().enumerate().for_each(|(batch_index, pt_32)| {
        // dbg!(pv_ct.level(), pt_32.encoding.as_ref().unwrap().level);
        let mut r32_ct = evaluator.mul_poly(pv_ct, pt_32.poly_ntt_ref());

        // populate 32 across all lanes
        let mut i = 32;
        while i < (degree / 2) {
            // rot_count += 1;
            let tmp = evaluator.rotate(&r32_ct, i, ek);
            evaluator.add_assign(&mut r32_ct, &tmp);
            i *= 2;
        }
        let tmp = evaluator.rotate(&r32_ct, 2 * degree - 1, ek);
        evaluator.add_assign(&mut r32_ct, &tmp);

        // extract first 4
        let mut fours = vec![];
        for i in 0..8 {
            fours.push(evaluator.mul_poly(&r32_ct, pts_4_roll[i].poly_ntt_ref()));
        }

        // expand fours
        let mut i = 4;
        while i < 32 {
            for j in 0..8 {
                // rot_count += 1;
                let tmp = evaluator.rotate(&mut fours[j], i, ek);
                evaluator.add_assign(&mut fours[j], &tmp);
            }
            i *= 2;
        }

        // extract ones
        for i in 0..8 {
            let four = &fours[i];
            for j in 0..4 {
                ones.push(evaluator.mul_poly(four, pts_1_roll[j].poly_ntt_ref()));
            }
        }

        // expand ones
        let mut i = 1;
        while i < 4 {
            for j in (batch_index * 32)..(batch_index + 1) * 32 {
                // rot_count += 1;
                let tmp = evaluator.rotate(&ones[j], i, ek);
                evaluator.add_assign(&mut ones[j], &tmp);
            }
            i *= 2;
        }
        // dbg!(ones.first().unwrap().c_ref()[0].coefficients.shape()[0]);
    });

    println!(
        "Pv expand took for batch_size {}: {:?} level{}",
        pts_32.len(),
        now.elapsed(),
        ones.first().unwrap().c_ref()[0].coefficients().shape()[0]
    );

    ones
}

/// Processes a batch of 32 slots
pub fn process_pv_batch(
    evaluator: &Evaluator,
    ek: &EvaluationKey,
    pv_ct: &Ciphertext,
    pts_4_roll: &[Plaintext],
    pts_1_roll: &[Plaintext],
    pts_32_batch: &[Plaintext],
    sk: &SecretKey,
    batch_index: usize,
    batch_size: usize,
    level: usize,
) -> (Array2<u128>, Array2<u128>) {
    println!("Processing batch {batch_index} with size {batch_size}");

    // expand batch cts
    let expanded_cts = pv_expand_batch(
        ek,
        evaluator,
        &pts_4_roll,
        &pts_1_roll,
        &pv_ct,
        pts_32_batch,
        &sk,
    );

    assert!(expanded_cts.len() == batch_size * 32);

    let start = batch_index * 32 * batch_size;
    let end = start + batch_size * 32;
    println!("Reading indices poly for range {start} to {end}...");
    let indices_poly = read_indices_poly(evaluator, level, start, end);
    println!(
        "Read indices poly of len {} from {start} to {end}...",
        indices_poly.len()
    );

    fma_poly(ek, evaluator, &expanded_cts, &indices_poly, sk, level)
}

pub fn phase2(
    evaluator: &Evaluator,
    pv_ct: &Ciphertext,
    ek: &EvaluationKey,
    sk: &SecretKey,
    level: usize,
) -> Ciphertext {
    let ctx = evaluator.params().poly_ctx(&PolyType::Q, level);
    let degree = ctx.degree();
    let moduli_count = ctx.moduli_count();

    // extract first 32
    let pts_32 = precompute_expand_32_roll_pt(degree, evaluator, level);
    // pt_4_roll must be 2d vector that extracts 1st 4, 2nd 4, 3rd 4, and 4th 4.
    let pts_4_roll = procompute_expand_roll_pt(32, 4, degree, evaluator, level);
    // pt_1_roll must be 2d vector that extracts 1st 1, 2nd 1, 3rd 1, and 4th 1.
    let pts_1_roll = procompute_expand_roll_pt(4, 1, degree, evaluator, level);

    // let mut rot_count = 0;
    dbg!(rayon::current_num_threads());

    let now = std::time::Instant::now();

    let num_threads = rayon::current_num_threads();
    let batch_size = pts_32.len() / num_threads;

    let mut result = vec![];
    pts_32
        .par_chunks(batch_size)
        .enumerate()
        .map(|(batch_index, pts_32_batch)| {
            let now = std::time::Instant::now();

            let tmp = process_pv_batch(
                evaluator,
                ek,
                pv_ct,
                &pts_4_roll,
                &pts_1_roll,
                pts_32_batch,
                sk,
                batch_index,
                batch_size,
                level,
            );

            println!("Batch {batch_index} time: {:?}", now.elapsed());

            tmp
        })
        .collect_into_vec(&mut result);

    // TODO: remove clones
    let mut first0 = result.first().unwrap().0.clone();
    let mut first1 = result.first().unwrap().1.clone();
    result.iter().skip(1).for_each(|tup| {
        add_u128_array(&mut first0, &tup.0);
        add_u128_array(&mut first1, &tup.1);
    });
    let res = coefficient_u128_to_ciphertext(evaluator.params(), &first0, &first1, level);

    println!("Phase 2 Time: {:?}", now.elapsed());
    // println!("Rot count: {rot_count}");
    res
    // expand 32 into
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
    fn test_phase2() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(15, 1 << 15);
        let sk = SecretKey::random(params.degree, &mut rng);

        let m = (0..params.degree)
            .into_iter()
            .map(|index| index as u64)
            .collect_vec();
        let evaluator = Evaluator::new(params);

        let mut level = 0;

        let pt = evaluator.plaintext_encode(&m, Encoding::simd(level));
        let mut ct = evaluator.encrypt(&sk, &pt, &mut rng);

        level = 12;
        evaluator.mod_down_level(&mut ct, level);

        evaluator.ciphertext_change_representation(&mut ct, Representation::Evaluation);

        // Generator rotation keys
        let mut ek = gen_pv_exapnd_rtgs(evaluator.params(), &sk, level);
        let res = phase2(&evaluator, &ct, &ek, &sk, level);

        dbg!(evaluator.measure_noise(&sk, &res));
    }

    #[test]
    fn test_pv_expand_batch() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(15, 1 << 15);
        let sk = SecretKey::random(params.degree, &mut rng);
        let degree = params.degree;
        let m = vec![3; params.degree];

        let level = 12;

        let evaluator = Evaluator::new(params);
        let pt = evaluator.plaintext_encode(&m, Encoding::default());
        let mut ct = evaluator.encrypt(&sk, &pt, &mut rng);
        evaluator.mod_down_level(&mut ct, level);
        evaluator.ciphertext_change_representation(&mut ct, Representation::Evaluation);

        let pts_32 = precompute_expand_32_roll_pt(degree, &evaluator, level);
        // pt_4_roll must be 2d vector that extracts 1st 4, 2nd 4, 3rd 4, and 4th 4.
        let pts_4_roll = procompute_expand_roll_pt(32, 4, degree, &evaluator, level);
        // pt_1_roll must be 2d vector that extracts 1st 1, 2nd 1, 3rd 1, and 4th 1.
        let pts_1_roll = procompute_expand_roll_pt(4, 1, degree, &evaluator, level);

        // restrict to single batch
        let pts_32 = (0..128)
            .into_iter()
            .map(|i| pts_32[i].clone())
            .collect_vec();

        let ek = gen_pv_exapnd_rtgs(evaluator.params(), &sk, level);
        let ones = pv_expand_batch(&ek, &evaluator, &pts_4_roll, &pts_1_roll, &ct, &pts_32, &sk);

        ones.iter().for_each(|ct| {
            dbg!(evaluator.measure_noise(&sk, ct));
            let rm = evaluator.plaintext_decode(&evaluator.decrypt(&sk, ct), Encoding::default());
            // dbg!(&rm);
            assert!(rm == m);
        });
    }

    #[test]
    fn test_process_pv_batch() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(3, 1 << 15);
        let sk = SecretKey::random(params.degree, &mut rng);
        let degree = params.degree;
        let m = vec![3; params.degree];

        let ek = gen_pv_exapnd_rtgs(&params, &sk, 0);

        let evaluator = Evaluator::new(params);
        let pt = evaluator.plaintext_encode(&m, Encoding::default());
        let mut ct = evaluator.encrypt(&sk, &pt, &mut rng);
        evaluator.ciphertext_change_representation(&mut ct, Representation::Evaluation);

        let level = 0;

        let pts_32 = precompute_expand_32_roll_pt(degree, &evaluator, level);
        // pt_4_roll must be 2d vector that extracts 1st 4, 2nd 4, 3rd 4, and 4th 4.
        let pts_4_roll = procompute_expand_roll_pt(32, 4, degree, &evaluator, level);
        // pt_1_roll must be 2d vector that extracts 1st 1, 2nd 1, 3rd 1, and 4th 1.
        let pts_1_roll = procompute_expand_roll_pt(4, 1, degree, &evaluator, level);

        let level = 0;

        // restrict to single batch
        let batch_size = 1;
        let batc_index = 0;
        let pts_32_batch = (0..batch_size)
            .into_iter()
            .map(|i| pts_32[i].clone())
            .collect_vec();

        let now = std::time::Instant::now();
        let (indices_res0, indices_res1) = process_pv_batch(
            &evaluator,
            &ek,
            &ct,
            &pts_4_roll,
            &pts_1_roll,
            &pts_32_batch,
            &sk,
            batc_index,
            batch_size,
            level,
        );
        println!("Time: {:?}", now.elapsed());

        // TODO test FMA of indices and weights
        let indices_ct =
            coefficient_u128_to_ciphertext(evaluator.params(), &indices_res0, &indices_res1, level);
        // let c0 = ;

        dbg!(evaluator.measure_noise(&sk, &indices_ct));
    }

    #[test]
    pub fn dummy_rot_count() {
        let degree = 1 << 15;
        let mut rot_count = 0;

        for index in 0..(degree / 32) {
            // populate 32 across all lanes
            let mut i = 32;
            while i < (degree / 2) {
                // expansion
                rot_count += 1;
                i *= 2;
            }
            rot_count += 1;

            // expand fours
            let mut i = 4;
            while i < 32 {
                for j in 0..8 {
                    rot_count += 1;
                }
                i *= 2;
            }

            for i in 0..8 {
                let mut j = 1;
                while j < 4 {
                    for k in 0..4 {
                        rot_count += 1;
                    }
                    j *= 2;
                }
            }
        }

        println!("Rot count: {rot_count}");
    }
}
