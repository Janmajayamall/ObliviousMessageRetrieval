use crate::optimised::{coefficient_u128_to_ciphertext, fma_reverse_u128_poly};
use crate::preprocessing::{precompute_expand_32_roll_pt, procompute_expand_roll_pt};
use crate::server::powers_x::evaluate_powers;
use crate::time_it;
use crate::utils::decrypt_and_print;
use crate::{
    optimised::{barret_reduce_coefficients_u128, sub_from_one},
    pvw::PvwParameters,
};
use bfv::{
    BfvParameters, Ciphertext, EvaluationKey, Evaluator, GaloisKey, Plaintext, Poly, PolyContext,
    PolyType, RelinearizationKey, Representation, SecretKey,
};
use core::time;
use itertools::{izip, Itertools};
use ndarray::{s, Array2};
use rand_chacha::rand_core::le;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::{collections::HashMap, sync::Arc, time::Instant};

// TODO: add multithreading for PVW

/// Rotates `s` for 512 times. After i^th rotation multiplies the result with plaintext at i^th index in hint_a_pts and adds the
/// result to final sum. Function takes advantage of the assumption that modulus is smaller than 50 bits to speed up fused mutliplication
/// and additions using 128 bit arithmetic, that is without modulur reduction. Result is reduced only once in the end using 128 bit barrett reduction
pub fn optimised_pvw_fma_with_rot(
    params: &BfvParameters,
    s: &Ciphertext,
    hint_a_pts: &[Plaintext],
    sec_len: usize,
    rtg: &GaloisKey,
    sk: &SecretKey,
) -> Ciphertext {
    debug_assert!(sec_len <= 512);

    let shape = s.c_ref()[0].coefficients.shape();
    let mut d_u128 = ndarray::Array2::<u128>::zeros((shape[0], shape[1]));
    let mut d1_u128 = ndarray::Array2::<u128>::zeros((shape[0], shape[1]));

    // To repeatedly rotate `s` and set output to `s`, `s` must be mutable, however the function
    // only takes `s` as a reference. Changing to mutable reference is unecessary after realising that
    // `rotate` also takes `s` as a reference. Hence, we process the first iteration outside the loop.
    fma_reverse_u128_poly(&mut d_u128, &s.c_ref()[0], hint_a_pts[0].poly_ntt_ref());
    fma_reverse_u128_poly(&mut d1_u128, &s.c_ref()[1], hint_a_pts[0].poly_ntt_ref());
    let mut s = rtg.rotate(s, params);
    for i in 1..sec_len {
        fma_reverse_u128_poly(&mut d_u128, &s.c_ref()[0], hint_a_pts[i].poly_ntt_ref());
        fma_reverse_u128_poly(&mut d1_u128, &s.c_ref()[1], hint_a_pts[i].poly_ntt_ref());
        s = rtg.rotate(&s, params);
    }
    coefficient_u128_to_ciphertext(params, &d_u128, &d1_u128, s.level())
}

pub fn pvw_decrypt(
    pvw_params: &Arc<PvwParameters>,
    evaluator: &Evaluator,
    hint_a_pts: &[Plaintext],
    hint_b_pts: &[Poly],
    pvw_sk_cts: &[Ciphertext],
    rtg: &GaloisKey,
    sk: &SecretKey,
) -> Vec<Ciphertext> {
    let sec_len = pvw_params.n.next_power_of_two();
    debug_assert!(hint_a_pts.len() == sec_len);
    debug_assert!(hint_b_pts.len() == pvw_params.ell);
    debug_assert!(pvw_sk_cts.len() == pvw_params.ell);

    let threads = rayon::current_num_threads();

    let mut sk_a = pvw_sk_cts
        .into_iter()
        .map(|s_ct| {
            optimised_pvw_fma_with_rot(evaluator.params(), s_ct, hint_a_pts, sec_len, rtg, sk)
        })
        .collect_vec();

    sk_a.iter_mut().zip(hint_b_pts.iter()).for_each(|(sa, b)| {
        // FIXME: Wo don't need this
        evaluator.sub_ciphertext_from_poly_inplace(sa, b);
    });

    sk_a
}
