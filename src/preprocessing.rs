use crate::pvw::{PvwCiphertext, PvwParameters};
use bfv::{BfvParameters, Encoding, Evaluator, Plaintext, Poly, Representation};
use std::sync::Arc;

pub fn pre_process_batch(
    pvw_params: &Arc<PvwParameters>,
    evaluator: &Evaluator,
    hints: &[PvwCiphertext],
) -> (Vec<Plaintext>, Vec<Poly>) {
    // can only process as many as polynomial_degree hints in a batch
    debug_assert!(hints.len() <= evaluator.params().degree);

    let sec_len = pvw_params.n.next_power_of_two();
    let mut hint_a_pts = vec![];
    for i in 0..sec_len {
        let mut m = vec![];
        for j in 0..hints.len() {
            let index = (j + i) % sec_len;
            if index < pvw_params.n {
                m.push(hints[j].a[index]);
            } else {
                m.push(0);
            }
        }
        hint_a_pts.push(evaluator.plaintext_encode(&m, Encoding::default()));
    }

    let mut hint_b_polys = vec![];
    let q_by4 = evaluator.params().plaintext_modulus / 4;
    for i in 0..pvw_params.ell {
        let mut m = vec![];
        for j in 0..hints.len() {
            m.push(
                evaluator
                    .params()
                    .plaintext_modulus_op
                    .sub_mod_fast(hints[j].b[i], q_by4),
            );
        }
        hint_b_polys.push(
            evaluator
                .plaintext_encode(&m, Encoding::default())
                .to_poly(evaluator.params(), Representation::Evaluation),
        );
    }

    // length of plaintexts will be sec_len
    (hint_a_pts, hint_b_polys)
}

/// Returns plaintexts to extract blocks of extract_size.
///
/// `block_size` is the size of block replicated consecutively on exisitng ciphertext.
/// `extract_size` is the size of block that is extracted from the existing block.
/// For example, if block_size if 32. The ciphertext text is of form [0,1,2,3,..31,0,1,2..31,....],
/// that is [0,1,2,3...31] is replicated across the ciphertext. If extract_size is 4, the function
/// will return 8 (32/4=8) plaintexts to extract 1st 4 from each block, 2nd 4, 3rd 4,...8th 4.
pub fn procompute_expand_roll_pt(
    block_size: usize,
    extract_size: usize,
    degree: usize,
    evaluator: &Evaluator,
) -> Vec<Plaintext> {
    let parts = block_size / extract_size;
    let mut pts = vec![];
    for part in 0..parts {
        let mut m = vec![];
        let lower = part * extract_size;
        let higher = (part + 1) * extract_size;
        for i in 0..degree {
            if i % block_size >= lower && i % block_size < higher {
                m.push(1);
            } else {
                m.push(0);
            }
        }
        pts.push(evaluator.plaintext_encode(&m, Encoding::default()));
    }
    pts
}

pub fn precompute_expand_32_roll_pt(degree: usize, evaluator: &Evaluator) -> Vec<Plaintext> {
    assert!(degree >= 32);

    let mut pts = vec![];
    for i in 0..(degree / 32) {
        let mut m = vec![0; degree];
        for j in (32 * i)..(32 * (i + 1)) {
            m[j] = 1u64;
        }
        pts.push(evaluator.plaintext_encode(&m, Encoding::default()));
    }

    pts
}

pub fn precompute_indices_pts(evaluator: &Evaluator, level: usize, max: usize) -> Vec<Poly> {
    let params = evaluator.params();
    let degree = params.degree;
    let bit_space = 64 - params.plaintext_modulus.leading_zeros() - 1;
    assert!(bit_space == 16);
    (0..std::cmp::min(degree, max))
        .into_iter()
        .map(|i| {
            let mut m = vec![0u64; degree];
            let col = i % 16;
            let row = i / 16;
            m[row] = 1 << col;
            evaluator
                .plaintext_encode(&m, Encoding::default())
                .poly_ntt_ref()
                .clone()
        })
        .collect()
}

pub fn precompute_weight_pts(evaluator: &Evaluator, level: usize, max: usize) -> Vec<Poly> {
    let params = evaluator.params();
    let degree = params.degree;
    let bit_space = 64 - params.plaintext_modulus.leading_zeros() - 1;
    assert!(bit_space == 16);
    (0..std::cmp::min(degree, max))
        .into_iter()
        .map(|i| {
            let mut m = vec![0u64; degree];
            let col = i % 16;
            let row = i / 16;
            m[row] = 1 << col;
            evaluator
                .plaintext_encode(&m, Encoding::default())
                .poly_ntt_ref()
                .clone()
        })
        .collect()
}
