use crate::pvw::{PvwCiphertext, PvwParameters, PvwSecretKey};
use bfv::{
    rot_to_galois_element, BfvParameters, Ciphertext, Encoding, EvaluationKey, Evaluator,
    GaloisKey, Plaintext, Representation, SecretKey,
};
use itertools::Itertools;
use rand::{thread_rng, CryptoRng, RngCore};

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

pub fn gen_pv_exapnd_rtgs(params: &BfvParameters, sk: &SecretKey) -> EvaluationKey {
    // create galois keys
    let mut rng = thread_rng();

    let level = 0;
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
