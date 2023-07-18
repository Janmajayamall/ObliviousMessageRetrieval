use core::panic;
use std::{f32::consts::PI, sync::Arc};

use crate::pvw::{PvwCiphertext, PvwParameters, PvwSecretKey};
use bfv::{
    rot_to_galois_element, BfvParameters, Ciphertext, Encoding, EvaluationKey, Evaluator,
    GaloisKey, Modulus, Plaintext, Representation, SecretKey,
};
use itertools::Itertools;
use rand::{thread_rng, CryptoRng, RngCore};
use rayon::vec;

/// Encrypts pvw sk under bfv in desired form
pub fn encrypt_pvw_sk<R: CryptoRng + RngCore>(
    evaluator: &Evaluator,
    bfv_sk: &SecretKey,
    pvw_sk: &PvwSecretKey,
    rng: &mut R,
) -> Vec<Ciphertext> {
    let sec_len = pvw_sk.par.n.next_power_of_two();
    let degree = evaluator.params().degree;

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

            let pt = evaluator.plaintext_encode(&m, Encoding::simd(0));
            let mut ct = evaluator.encrypt(bfv_sk, &pt, rng);
            evaluator.ciphertext_change_representation(&mut ct, Representation::Evaluation);
            ct
        })
        .collect_vec();

    cts
}

pub fn evaluation_key<R: CryptoRng + RngCore>(
    params: &BfvParameters,
    sk: &SecretKey,
    rng: &mut R,
) -> EvaluationKey {
    #[cfg(feature = "level")]
    let rlk_levels = (0..12).into_iter().collect_vec();

    #[cfg(not(feature = "level"))]
    let rlk_levels = vec![0];

    let level = 12;
    let (mut rtg_indices, mut rtg_levels) = get_pv_expand_rtgs_vecs(level, params.degree);

    // pvw rot key
    rtg_indices.push(1);
    rtg_levels.push(0);

    EvaluationKey::new(params, sk, &rlk_levels, &rtg_levels, &rtg_indices, rng)
}

pub fn gen_pv_exapnd_rtgs<R: CryptoRng + RngCore>(
    params: &BfvParameters,
    sk: &SecretKey,
    level: usize,
    rng: &mut R,
) -> EvaluationKey {
    // create galois keys
    let (rtg_indices, rtg_levels) = get_pv_expand_rtgs_vecs(level, params.degree);
    EvaluationKey::new(params, sk, &[], &rtg_levels, &rtg_indices, rng)
}

pub fn get_pv_expand_rtgs_vecs(level: usize, degree: usize) -> (Vec<isize>, Vec<usize>) {
    let mut rtg_indices = vec![];
    let mut rtg_levels = vec![];
    // keys for 32 expand
    let mut i = 32;
    while i < degree as isize {
        rtg_indices.push(i);
        rtg_levels.push(level);
        i *= 2;
    }
    // row swap
    rtg_indices.push(2 * degree as isize - 1);
    rtg_levels.push(level);

    // keys for 4 expand
    let mut i = 4;
    while i < 32 {
        rtg_indices.push(i);
        rtg_levels.push(level);
        i *= 2;
    }

    // keys for 1 expand
    let mut i = 1;
    while i < 4 {
        rtg_indices.push(i);
        rtg_levels.push(level);
        i *= 2;
    }

    (rtg_indices, rtg_levels)
}

pub fn pv_decompress(evaluator: &Evaluator, indices_ct: &Ciphertext, sk: &SecretKey) -> Vec<u64> {
    let pv = evaluator.plaintext_decode(&evaluator.decrypt(sk, indices_ct), Encoding::default());
    let mut pv_decompressed = vec![];

    assert!(pv.len() == 32768);

    pv.iter().for_each(|value| {
        let mut value = *value;
        for _ in 0..16 {
            pv_decompressed.push(value & 1);
            value >>= 1;
        }
    });

    pv_decompressed
}

pub fn construct_lhs(
    pv: &[u64],
    buckets: &[Vec<u64>],
    weights: &[Vec<u64>],
    k: usize,
    gamma: usize,
    set_size: usize,
) -> Vec<Vec<u64>> {
    let mut lhs = vec![vec![0u64; k]; k * 2];
    let mut curr_col = 0;
    for i in 0..set_size {
        let value = pv[i];
        assert!(value <= 1);

        if value == 1 {
            if curr_col == k {
                panic!("Overflow!");
            }

            let row_buckets = &buckets[i];
            let row_weights = &weights[i];

            for j in 0..gamma {
                let b = row_buckets[j];
                let w = row_weights[j];
                lhs[b as usize][curr_col] = w;
            }
            curr_col += 1;
        }
    }
    lhs
}

pub fn construct_rhs(weights_vec: &[u64], bucket_size: usize) -> Vec<Vec<u64>> {
    weights_vec
        .chunks_exact(bucket_size)
        .map(|c| c.to_vec())
        .collect_vec()
}

pub fn scale_vec(scale_by: u64, a: &[u64], modq: &Modulus) -> Vec<u64> {
    a.iter()
        .map(|v| modq.mul_mod_fast(scale_by, *v))
        .collect_vec()
}

pub fn print_matrix(m: &Vec<Vec<u64>>, row: usize, col: usize) {
    println!("### Matrix ###");
    println!("rows: {row}; cols: {col}");
    for i in 0..row {
        println!("{:?}", &m[i][..col]);
    }
}

pub fn solve_equations(
    mut lhs: Vec<Vec<u64>>,
    mut rhs: Vec<Vec<u64>>,
    k: usize,
    modq: u64,
) -> Vec<Vec<u64>> {
    // max no of vars
    let cols = k;
    // no of equations
    let rows = k * 2;

    let modq = Modulus::new(modq);
    let mut pivot_rows = vec![-1; cols];
    for pi in 0..cols {
        for eq in 0..rows {
            // find the pivot
            if !pivot_rows.contains(&(eq as isize)) {
                if (pivot_rows[pi] != -1 && lhs[pivot_rows[pi] as usize][pi] < lhs[eq][pi])
                    || (pivot_rows[pi] == -1 && lhs[eq][pi] != 0)
                {
                    pivot_rows[pi] = eq as isize;
                }
            }
        }

        if pivot_rows[pi] == -1 {
            break;
        }

        let pivot_row = pivot_rows[pi] as usize;
        let pivot_value = lhs[pivot_row][pi];
        for r in 0..rows {
            if r != pivot_row as usize {
                let value = lhs[r][pi];

                if value == 0 {
                    continue;
                }

                // scale `r`th row by `pivot_value/value` and then subtract `pivot_row` from `r`th to cancel `r`th
                // row's coefficient at `pi`th column.
                let scale_by = modq.mul_mod_fast(pivot_value, modq.inv(value));
                let mut scaled_lhs = scale_vec(scale_by, &lhs[r], &modq);
                modq.sub_mod_fast_vec(&mut scaled_lhs, &lhs[pivot_row]);
                lhs[r] = scaled_lhs;

                let mut scaled_rhs = scale_vec(scale_by, &rhs[r], &modq);
                modq.sub_mod_fast_vec(&mut scaled_rhs, &rhs[pivot_row]);
                rhs[r] = scaled_rhs;
            }
        }
    }
    let mut messages = vec![];
    for pi in 0..cols {
        if pivot_rows[pi] != -1 {
            let row = pivot_rows[pi] as usize;
            let col = pi as usize;
            let value = lhs[row][col];
            let value_inv = modq.inv(value);
            let m = rhs[row]
                .iter()
                .map(|v| modq.mul_mod_fast(value_inv, *v))
                .collect_vec();
            messages.push(m);
        }
    }

    messages
}

#[cfg(test)]
mod tests {
    use bfv::Modulus;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use crate::{
        client::construct_rhs, preprocessing::assign_buckets_and_weights,
        utils::generate_random_payloads, BUCKET_SIZE, GAMMA, K,
    };

    use super::{construct_lhs, solve_equations};

    #[test]
    fn test_solve_linear_equations() {
        let mut rng = ChaCha8Rng::from_seed([2; 32]);
        let qmod = Modulus::new(65537);
        let set_size = 32768;
        let bucket_size = BUCKET_SIZE as u64;
        let (seed, buckets, weights) =
            assign_buckets_and_weights(K * 2, GAMMA, qmod.modulus(), set_size, &mut rng);

        // Randomly generates pertinency vector with K pertinent indices
        let mut pv = vec![0u64; set_size];
        let mut pertinent_indices = vec![];
        while pertinent_indices.len() != 64 {
            let index = rng.gen::<usize>() % set_size;
            if !pertinent_indices.contains(&index) {
                pertinent_indices.push(index);
                pv[index] = 1;
            }
        }
        pertinent_indices.sort();

        // randomly generate corresponding data
        let payloads = generate_random_payloads(set_size);

        // calculate weight vector
        let mut weight_vec = vec![0u64; set_size];
        for lane_index in 0..set_size {
            // ignore the ones not pertinent. We can do this here, but not when pv is ciphertext
            if pv[lane_index] == 1 {
                let row_bucket = &buckets[lane_index];
                let row_weights = &weights[lane_index];

                for j in 0..GAMMA {
                    let bucket_index = row_bucket[j];
                    let bucket_weight = row_weights[j];
                    let bucket_offset = (bucket_index * bucket_size) as usize;
                    for i in 0..bucket_size as usize {
                        weight_vec[bucket_offset + i] = qmod.add_mod(
                            weight_vec[bucket_offset + i],
                            qmod.mul_mod_fast(bucket_weight, payloads[lane_index][i] as u64),
                        );
                    }
                }
            }
        }

        let lhs = construct_lhs(&pv, &buckets, &weights, K, GAMMA, set_size);
        let rhs = construct_rhs(&weight_vec, bucket_size as usize);

        let result = solve_equations(lhs, rhs, K, qmod.modulus());
        let mut expected = vec![];
        pertinent_indices.iter().for_each(|index| {
            expected.push(payloads[*index].clone());
        });
        assert_eq!(result, expected);
    }
}
