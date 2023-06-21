use crate::pvw::{PvwCiphertext, PvwParameters, PvwSecretKey};
use bfv::{
    utils::rot_to_galois_element, BfvParameters, Ciphertext, Encoding, GaloisKey, Plaintext,
    Representation, SecretKey,
};
use itertools::Itertools;
use rand::{thread_rng, CryptoRng, RngCore};
use std::{collections::HashMap, sync::Arc};
use traits::Ntt;

/// Encrypts pvw sk under bfv in desired form
pub fn encrypt_pvw_sk<R: CryptoRng + RngCore, T: Ntt>(
    bfv_params: &Arc<BfvParameters<T>>,
    bfv_sk: &SecretKey<T>,
    pvw_sk: &PvwSecretKey,
    rng: &mut R,
) -> Vec<Ciphertext<T>> {
    let sec_len = pvw_sk.par.n.next_power_of_two();
    let degree = bfv_params.polynomial_degree;

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
            let pt = Plaintext::encode(&m, bfv_params, Encoding::simd(0));
            let mut ct = bfv_sk.encrypt(&pt, rng);
            ct.change_representation(&Representation::Evaluation);
            ct
        })
        .collect_vec();

    cts
}

pub fn gen_pv_exapnd_rtgs<T: Ntt>(
    params: &Arc<BfvParameters<T>>,
    sk: &SecretKey<T>,
) -> HashMap<usize, GaloisKey<T>> {
    // create galois keys
    let mut rng = thread_rng();
    let mut rtks = HashMap::new();
    let ct_ctx = params.ciphertext_ctx_at_level(0);

    // keys for 32 expand
    let mut i = 32;
    while i < params.polynomial_degree {
        let exponent = rot_to_galois_element(i as isize, params.polynomial_degree);
        let key = GaloisKey::new(exponent, &ct_ctx, &sk, &mut rng);
        rtks.insert(i, key);
        i *= 2;
    }
    // row swap
    rtks.insert(
        2 * params.polynomial_degree - 1,
        GaloisKey::new(2 * params.polynomial_degree - 1, &ct_ctx, &sk, &mut rng),
    );

    // keys for 4 expand
    let mut i = 4;
    while i < 32 {
        let exponent = rot_to_galois_element(i as isize, params.polynomial_degree);
        let key = GaloisKey::new(exponent, &ct_ctx, &sk, &mut rng);
        rtks.insert(i, key);
        i *= 2;
    }

    // keys for 1 expand
    let mut i = 1;
    while i < 4 {
        let exponent = rot_to_galois_element(i as isize, params.polynomial_degree);
        let key = GaloisKey::new(exponent, &ct_ctx, &sk, &mut rng);
        rtks.insert(i, key);
        i *= 2;
    }

    rtks
}
