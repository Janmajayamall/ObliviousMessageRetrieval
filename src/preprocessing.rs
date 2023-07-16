use crate::pvw::{PvwCiphertext, PvwParameters};
use bfv::{BfvParameters, Encoding, Evaluator, Modulus, Plaintext, Poly, Representation};
use rand::{distributions::Uniform, CryptoRng, Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;

pub fn pre_process_batch(
    pvw_params: &Arc<PvwParameters>,
    evaluator: &Evaluator,
    hints: &[PvwCiphertext],
) -> (Vec<Plaintext>, Vec<Poly>) {
    // can only process as many as polynomial_degree hints in a batch
    assert!(hints.len() <= evaluator.params().degree);

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
/// `block_size` is the size of block replicated consecutively in lanes of ciphertext.
/// `extract_size` is the size of chunk that will extracted from the a block.
/// For example, if block_size is 32. The ciphertext lanes look like [0,1,2,3,..31,0,1,2..31,....],
/// that is block [0,1,2,3...31] is repeated across all lanes. If extract_size is 4, the function
/// will return 8 (32/4=8) plaintexts to extract 1st 4 from each block, 2nd 4, 3rd 4,...8th 4.
pub fn procompute_expand_roll_pt(
    block_size: usize,
    extract_size: usize,
    degree: usize,
    evaluator: &Evaluator,
    level: usize,
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
        pts.push(evaluator.plaintext_encode(&m, Encoding::simd(level)));
    }
    pts
}

pub fn precompute_expand_32_roll_pt(
    degree: usize,
    evaluator: &Evaluator,
    level: usize,
) -> Vec<Plaintext> {
    assert!(degree >= 32);

    let mut pts = vec![];
    for i in 0..(degree / 32) {
        let mut m = vec![0; degree];
        for j in (32 * i)..(32 * (i + 1)) {
            m[j] = 1u64;
        }
        pts.push(evaluator.plaintext_encode(&m, Encoding::simd(level)));
    }

    pts
}

pub fn read_indices_poly(evaluator: &Evaluator, level: usize, min: usize, max: usize) -> Vec<Poly> {
    precompute_indices_pts(evaluator, level, min, max)
}

pub fn precompute_indices_pts(
    evaluator: &Evaluator,
    level: usize,
    min: usize,
    max: usize,
) -> Vec<Poly> {
    let params = evaluator.params();
    let degree = params.degree;
    let bit_space = 64 - params.plaintext_modulus.leading_zeros() - 1;
    assert!(bit_space == 16);
    (min..std::cmp::min(degree, max))
        .into_iter()
        .map(|i| {
            let mut m = vec![0u64; degree];
            let col = i % 16;
            let row = i / 16;
            m[row] = 1 << col;
            evaluator
                .plaintext_encode(&m, Encoding::simd(level))
                .move_poly_ntt()
        })
        .collect()
}

pub fn compute_weight_pts(
    evaluator: &Evaluator,
    level: usize,
    payloads: &[Vec<u64>],
    min: usize,
    max: usize,
    bucket_size: usize,
    buckets: &[Vec<u64>],
    weights: &[Vec<u64>],
    gamma: usize,
) -> Vec<Poly> {
    let modq = Modulus::new(65537);

    let params = evaluator.params();
    let degree = params.degree;
    let bit_space = 64 - params.plaintext_modulus.leading_zeros() - 1;
    assert!(bit_space == 16);
    (min..std::cmp::min(degree, max))
        .into_iter()
        .map(|i| {
            let payload = &payloads[i];

            let row_buckets = &buckets[i];
            let row_weights = &weights[i];

            // prepare the bucket
            let mut m = vec![0u64; degree];
            for j in 0..gamma {
                // get j^th bucket and corresponding weight
                let bucket_index = row_buckets[j];
                let bucket_weight = row_weights[j];

                let bucket_offset = bucket_size * (bucket_index as usize);

                for j in 0..bucket_size {
                    m[j + bucket_offset] = modq.mul_mod_fast(bucket_weight, payload[j]);
                }
            }

            evaluator
                .plaintext_encode(&m, Encoding::simd(level))
                .move_poly_ntt()
        })
        .collect()
}

/// Seeds a new prng and assigns `gamma` buckets and weights to
/// each row in set.
///
/// Note: We use the same prng for sampling buckets and weights. This
/// implies that the order of sampling bucket and weight in sequence for each row
/// must be followed across all clients. Moreover, both bucket and weight should
/// have same type, ie u64. Care should be taken to not change bucket to
/// usize otherwise values produced on runtime environments with native
/// word size 32 (for example, WASM) will be different.
pub fn assign_buckets_and_weights<R: CryptoRng + RngCore>(
    no_of_buckets: usize,
    gamma: usize,
    q_mod: u64,
    set_size: usize,
    rng: &mut R,
) -> ([u8; 32], Vec<Vec<u64>>, Vec<Vec<u64>>) {
    // create a seeded prng
    let mut seed = <ChaCha8Rng as SeedableRng>::Seed::default();
    rng.fill_bytes(&mut seed);

    let mut buckets = Vec::with_capacity(set_size);
    let mut weights = Vec::with_capacity(set_size);

    let dist = Uniform::new(0, no_of_buckets as u64);
    let weight_dist = Uniform::new(1u64, q_mod);

    for _ in 0..set_size {
        let mut row_buckets = Vec::with_capacity(gamma);
        let mut row_weights = Vec::with_capacity(gamma);

        while row_buckets.len() != gamma {
            // random bucket
            let bucket = rng.sample(dist);

            // avoid duplicate buckets
            if !row_buckets.contains(&bucket) {
                row_buckets.push(bucket);

                // Assign weight
                // Weight cannot be zero
                let weight = rng.sample(weight_dist);
                row_weights.push(weight);
            }
        }

        buckets.push(row_buckets);
        weights.push(row_weights);
    }

    (seed, buckets, weights)
}
