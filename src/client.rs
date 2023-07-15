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

pub fn gen_pv_exapnd_rtgs(params: &BfvParameters, sk: &SecretKey, level: usize) -> EvaluationKey {
    // create galois keys
    let mut rng = thread_rng();

    let mut rtg_indices = vec![];
    let mut rtg_levels = vec![];
    // keys for 32 expand
    let mut i = 32;
    while i < params.degree as isize {
        rtg_indices.push(i);
        rtg_levels.push(level);
        i *= 2;
    }
    // row swap
    rtg_indices.push(2 * params.degree as isize - 1);
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

    EvaluationKey::new(params, sk, &[], &rtg_levels, &rtg_indices, &mut rng)
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
    buckets: Vec<Vec<u64>>,
    weights: Vec<Vec<u64>>,
    k: usize,
    gamma: usize,
    set_size: usize,
) {
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
}

pub fn scale_vec(scale_by: u64, a: &[u64], modq: &Modulus) -> Vec<u64> {
    a.iter()
        .map(|v| modq.mul_mod_fast(scale_by, *v))
        .collect_vec()
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

    let mut pivot_indices = vec![-1; cols];
    for pi in 0..cols {
        for eq in 0..rows {
            // find the pivot
            if !pivot_indices.contains(&(pi as isize)) {
                if pivot_indices[pi] != -1 && lhs[pivot_indices[pi] as usize][pi] < lhs[eq][pi] {
                    pivot_indices[pi] = eq as isize;
                } else if (pivot_indices[pi] == -1 && lhs[eq][pi] != 0) {
                    pivot_indices[pi] = eq as isize;
                }
            }
        }

        if pivot_indices[pi] == -1 {
            break;
        }

        let pivot_index = pivot_indices[pi];
        let pivot_value = lhs[pivot_index as usize][pi];
        let pivot_row_lhs = lhs[pivot_index as usize].clone();
        let pivot_row_rhs = rhs[pivot_index as usize].clone();
        for eq in 0..rows {
            if eq != pivot_index as usize {
                let value = lhs[eq][pi];

                let scale_by = modq.mul_mod_fast(pivot_value, modq.inv(value));
                let mut scaled_lhs = scale_vec(scale_by, &lhs[eq], &modq);
                modq.sub_mod_fast_vec(&mut scaled_lhs, &pivot_row_lhs);
                lhs[eq] = scaled_lhs;

                let mut scaled_rhs = scale_vec(scale_by, &rhs[eq], &modq);
                modq.sub_mod_fast_vec(&mut scaled_rhs, &pivot_row_rhs);
                rhs[eq] = scaled_rhs;
            }
        }
    }

    let mut messages = vec![];
    for pi in 0..cols {
        if pivot_indices[pi] != -1 {
            let value = lhs[pivot_indices[pi] as usize][pi as usize];
            let value_inv = modq.inv(value);
            let m = rhs[pivot_indices[pi] as usize]
                .iter()
                .map(|v| modq.mul_mod_fast(value_inv, *v))
                .collect_vec();
            messages.push(m);
        }
    }

    messages
}
