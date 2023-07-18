use std::f32::consts::E;

use crate::optimised::{coefficient_u128_to_ciphertext, fma_reverse_u128_poly};
use crate::preprocessing::{
    compute_weight_pts, precompute_expand_32_roll_pt, procompute_expand_roll_pt, read_indices_poly,
};

use crate::{
    optimised::{barret_reduce_coefficients_u128, optimised_pvw_fma_with_rot, sub_from_one},
    pvw::PvwParameters,
};
use crate::{print_noise, BUCKET_SIZE, GAMMA, MESSAGE_BYTES};
use bfv::{
    BfvParameters, Ciphertext, EvaluationKey, Evaluator, GaloisKey, Plaintext, Poly, PolyType,
    RelinearizationKey, Representation, SecretKey,
};
use itertools::{izip, Itertools};
use ndarray::{s, Array2};
use num_traits::ToPrimitive;
use rand_chacha::rand_core::le;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSlice;

pub fn phase2_precomputes(
    evaluator: &Evaluator,
    degree: usize,
    level: usize,
) -> (Vec<Plaintext>, Vec<Plaintext>, Vec<Plaintext>) {
    // extract first 32
    let pts_32 = precompute_expand_32_roll_pt(degree, evaluator, level);
    // pt_4_roll must be 2d vector that extracts 1st 4, 2nd 4, 3rd 4, and 4th 4.
    let pts_4_roll = procompute_expand_roll_pt(32, 4, degree, evaluator, level);
    // pt_1_roll must be 2d vector that extracts 1st 1, 2nd 1, 3rd 1, and 4th 1.
    let pts_1_roll = procompute_expand_roll_pt(4, 1, degree, evaluator, level);

    (pts_32, pts_4_roll, pts_1_roll)
}

/// Calculates and returns dot product of cts and polys.
///
/// res0 += cts[i].c0 * poly
/// res1 += cts[i].c1 * poly
///
/// Assumes that each qi in moduli chain is <= 50 bits. Leverages this to
/// replace modulur vec multiplication with simply vector multiplication
/// and delays modular reduction until end.
///
/// Returns result without modulur reduction.
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

/// a += b
pub fn add_u128_array(a: &mut Array2<u128>, b: &Array2<u128>) {
    izip!(a.outer_iter_mut(), b.outer_iter(),).for_each(|(mut ac, bc)| {
        izip!(ac.iter_mut(), bc.iter()).for_each(|(v0, v1)| {
            *v0 += *v1;
        });
    });
}

/// A single set of 32 lanes is expanded into 32 ciphertexts where each lane in new ciphertext at index i is equal to corresponding lane at index i in
/// original ciphertext.
///
/// pv_ct is expanded in set of 32 lanes. Hence each set outputs 32 ciphertext. For a batch of size `batch_size` function returns
/// `32*batch_size` ciphertexts.
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

        // extract sets of 4
        let mut fours = vec![];
        for i in 0..8 {
            fours.push(evaluator.mul_poly(&r32_ct, pts_4_roll[i].poly_ntt_ref()));
        }

        // expand each set of 4 across all lanes
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

        // expand ones across all lanes
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

    print_noise!(
        println!("pv_expand_batch ones[0] noise: {}", evaluator.measure_noise(sk, ones.first().unwrap()));
        println!("pv_expand_batch ones[-1] noise: {}", evaluator.measure_noise(sk, ones.last().unwrap()));
    );

    println!(
        "Pv expand took for batch_size {}: {:?};",
        pts_32.len(),
        now.elapsed(),
    );

    ones
}

/// Expands the batch of lanes in pv_ct into individual ciphertexts. Then multiplies each ciphertext with corresponding
/// index plaintext and weight plaintext and then adds all products into 2 ciphertexts, 1 for indices and 1 for weights.
pub fn process_pv_batch(
    evaluator: &Evaluator,
    ek: &EvaluationKey,
    pv_ct: &Ciphertext,
    pts_4_roll: &[Plaintext],
    pts_1_roll: &[Plaintext],
    pts_32_batch: &[Plaintext],
    payloads: &[Vec<u64>],
    start: usize,
    end: usize,
    level: usize,
    buckets: &[Vec<u64>],
    weights: &[Vec<u64>],
    sk: &SecretKey,
) -> ((Array2<u128>, Array2<u128>), (Array2<u128>, Array2<u128>)) {
    println!("Processing batches: {start} - {end}");

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

    assert!(expanded_cts.len() == (end - start) * 32);

    let start_lane = start * 32;
    let end_lane = end * 32;

    println!("Reading indices poly for lanes {start_lane} to {end_lane}...");
    // Note that indices_poly current computes the poly at runtime. However these polynomials do not change
    // across multiple runs. Hence can be pre-computed at stored, replacing expensive NTTs with read operations.
    let indices_poly = read_indices_poly(evaluator, level, start_lane, end_lane);

    // Like indices_poly weight_polys can also be pre-computed and stored but with few caveats.
    // 1. Pre-computation is limited to each batch. That means for every new batch of 32768 messages we require
    // new set of weight_polys. This is because same `seed` used for assigning each index to different buckets
    // cannot be used across different batches. Otherwise, the notion that message board is contructed randomly
    // will be violated since the server will have to make the seed public for previous batch and an attacker
    // can send messages to message board such that system of linear equations are unsolvable.
    // 2. Since pre-computation is limited to batch of 32768 we cannot process phase2 across multiple
    // batches. I think any practical deployment of OMR will require separating phase 1 and phase 2 and allowing
    // users to request messages in a dynamic range. For example, user can request message in range 60,000 and 100,000.
    // Hence, at least in any real life scenario pre-computing weight_polys will be useless.
    let weights_poly = compute_weight_pts(
        evaluator,
        level,
        payloads,
        start_lane,
        end_lane,
        BUCKET_SIZE,
        buckets,
        weights,
        GAMMA,
    );

    let indices_res = fma_poly(ek, evaluator, &expanded_cts, &indices_poly, sk, level);
    let weights_res = fma_poly(ek, evaluator, &expanded_cts, &weights_poly, sk, level);

    (indices_res, weights_res)
}

/// Divides pts_32 among available threads equally and calls `process_pv_batch` for each chunk (ie batch).
///
/// Warning: available threads must be power of 2 for maximum core usage.
fn helper(
    start: usize,
    end: usize,
    set_len: usize,
    pts_32: &[Plaintext],
    evaluator: &Evaluator,
    ek: &EvaluationKey,
    pv_ct: &Ciphertext,
    pts_4_roll: &[Plaintext],
    pts_1_roll: &[Plaintext],
    payloads: &[Vec<u64>],
    level: usize,
    buckets: &[Vec<u64>],
    weights: &[Vec<u64>],
    sk: &SecretKey,
) -> ((Array2<u128>, Array2<u128>), (Array2<u128>, Array2<u128>)) {
    if (end - start) <= set_len {
        let res = process_pv_batch(
            evaluator,
            ek,
            pv_ct,
            &pts_4_roll,
            &pts_1_roll,
            &pts_32[start..end],
            payloads,
            start,
            end,
            level,
            buckets,
            weights,
            sk,
        );
        res
    } else {
        let mid = (start + end) / 2;
        let (mut r0, r1) = rayon::join(
            || {
                helper(
                    start, mid, set_len, pts_32, evaluator, ek, pv_ct, pts_4_roll, pts_1_roll,
                    payloads, level, buckets, weights, sk,
                )
            },
            || {
                helper(
                    mid, end, set_len, pts_32, evaluator, ek, pv_ct, pts_4_roll, pts_1_roll,
                    payloads, level, buckets, weights, sk,
                )
            },
        );

        // add indices polys
        // c0
        add_u128_array(&mut r0.0 .0, &r1.0 .0);
        // c1
        add_u128_array(&mut r0.0 .1, &r1.0 .1);

        // add weights polys
        // c0
        add_u128_array(&mut r0.1 .0, &r1.1 .0);
        // c1
        add_u128_array(&mut r0.1 .1, &r1.1 .1);

        r0
    }
}

pub fn phase2(
    evaluator: &Evaluator,
    pv_ct: &Ciphertext,
    ek: &EvaluationKey,
    level: usize,
    pts_32: &[Plaintext],
    pts_4_roll: &[Plaintext],
    pts_1_roll: &[Plaintext],
    payloads: &[Vec<u64>],
    buckets: &[Vec<u64>],
    weights: &[Vec<u64>],
    sk: &SecretKey,
) -> (Ciphertext, Ciphertext) {
    // pertinency vector ciphertext must be in Evaluation representation
    assert!(pv_ct.c_ref()[0].representation() == &Representation::Evaluation);

    let num_threads = rayon::current_num_threads() as f64;
    let set_len = (pts_32.len() as f64 / num_threads)
        .ceil()
        .to_usize()
        .unwrap();

    let (indices_poly_u128, weights_poly_u128) = helper(
        0,
        pts_32.len(),
        set_len,
        pts_32,
        evaluator,
        ek,
        pv_ct,
        pts_4_roll,
        pts_1_roll,
        payloads,
        level,
        buckets,
        weights,
        sk,
    );

    let indices_ct = coefficient_u128_to_ciphertext(
        evaluator.params(),
        &indices_poly_u128.0,
        &indices_poly_u128.1,
        level,
    );

    let weights_ct = coefficient_u128_to_ciphertext(
        evaluator.params(),
        &weights_poly_u128.0,
        &weights_poly_u128.1,
        level,
    );

    (indices_ct, weights_ct)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{
        client::gen_pv_exapnd_rtgs,
        optimised::{coefficient_u128_to_ciphertext, sub_from_one_precompute},
        plaintext::powers_of_x_modulus,
        preprocessing::{assign_buckets_and_weights, precompute_indices_pts},
        utils::{generate_bfv_parameters, generate_random_payloads, precompute_range_constants},
        K,
    };
    use bfv::{BfvParameters, Encoding};
    use rand::thread_rng;

    #[test]
    fn test_phase2() {
        let mut rng = thread_rng();
        let params = generate_bfv_parameters();
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
        let mut ek = gen_pv_exapnd_rtgs(evaluator.params(), &sk, level, &mut rng);

        // pre-computes
        let (pts_32_batch, pts_4_roll, pts_1_roll) =
            phase2_precomputes(&evaluator, evaluator.params().degree, level);
        let (_, buckets, weights) = assign_buckets_and_weights(
            K * 2,
            GAMMA,
            evaluator.params().plaintext_modulus,
            evaluator.params().degree,
            &mut rng,
        );
        let payloads = generate_random_payloads(evaluator.params().degree);

        // restrict to single batch
        let pts_32_batch = pts_32_batch[..4].to_vec();

        let (indices_ct, weights_ct) = phase2(
            &evaluator,
            &ct,
            &ek,
            level,
            &pts_32_batch,
            &pts_4_roll,
            &pts_1_roll,
            &payloads,
            &buckets,
            &weights,
            &sk,
        );

        dbg!(evaluator.measure_noise(&sk, &indices_ct));
        dbg!(evaluator.measure_noise(&sk, &weights_ct));
    }

    #[test]
    fn test_pv_expand_batch() {
        let mut rng = thread_rng();
        let params = generate_bfv_parameters();
        let sk = SecretKey::random(params.degree, &mut rng);
        let degree = params.degree;
        let m = vec![3; params.degree];

        let level = 12;

        let evaluator = Evaluator::new(params);
        let pt = evaluator.plaintext_encode(&m, Encoding::default());
        let mut ct = evaluator.encrypt(&sk, &pt, &mut rng);
        evaluator.mod_down_level(&mut ct, level);
        evaluator.ciphertext_change_representation(&mut ct, Representation::Evaluation);

        let (pts_32_batch, pts_4_roll, pts_1_roll) =
            phase2_precomputes(&evaluator, evaluator.params().degree, level);

        // restrict to single batch
        let pts_32_batch = pts_32_batch[..1].to_vec();

        let ek = gen_pv_exapnd_rtgs(evaluator.params(), &sk, level, &mut rng);
        let ones = pv_expand_batch(
            &ek,
            &evaluator,
            &pts_4_roll,
            &pts_1_roll,
            &ct,
            &pts_32_batch,
            &sk,
        );

        assert!(pts_32_batch.len() * 32 == ones.len());

        ones.iter().for_each(|ct| {
            dbg!(evaluator.measure_noise(&sk, ct));
            let rm = evaluator.plaintext_decode(&evaluator.decrypt(&sk, ct), Encoding::default());
            // dbg!(&rm);
            assert!(rm == m);
        });
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

    fn reverse() {}
}
