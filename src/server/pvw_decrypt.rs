use crate::optimised::{add_u128, coefficient_u128_to_ciphertext, fma_reverse_u128_poly};
use crate::preprocessing::{precompute_expand_32_roll_pt, procompute_expand_roll_pt};
use crate::server::powers_x::evaluate_powers;
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
use ndarray::{s, Array, Array2};
use rand_chacha::rand_core::le;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
};

/// Pre-compute rotations of `sk_cts`s such that pvw_decrypt can leverage all avaialble cores.
pub fn pvw_setup(
    evaluator: &Evaluator,
    ek: &EvaluationKey,
    pvw_sk_cts: &[Ciphertext],
) -> Vec<Vec<Ciphertext>> {
    // assumes that threads % 4=0
    let threads = rayon::current_num_threads();
    assert!(threads % 4 == 0);
    // pvw decrypt is called 4 times with different ciphertexts. Thus, we first assign total no. of threads
    // equally among all 4 calls before dividing them further for rotations within each call.
    let threads_by_4 = threads as f64 / 4.0;
    // We need to distribute the task of 512 rotations among all threads available to a single pvw_decrypt call equally. For ex, if threads_by_4 = 8
    // then each thread will perform `512/8=64` rotations. However, rotations will be distributed unequally if 512%threads_by_4 != 0
    // and the time will be set by the last thread to which maximum number of rotations are allocated. For ex, let thread_by_4 = 11.
    // Since 512%11 != 0, 10 threads will be allocated 46 rotations whereas the last thread will have to do 512-(10*46) = 52 rotations,
    // thus defining the total time taken.
    let rots_per_thread = (512.0 / threads_by_4).floor() as usize;

    dbg!(threads_by_4);
    dbg!(rots_per_thread);

    let mut checkpoint_cts = vec![vec![]; 4];
    let mut cts = pvw_sk_cts.to_vec();

    // verify all cts are in Evaluation representation for efficient rotations and plaintext multiplication in pvw_decrypt
    cts.iter()
        .for_each(|c| assert!(c.c_ref()[0].representation() == &Representation::Evaluation));

    for j in 0..threads_by_4 as usize {
        // Checkpoints for each thread are cts rotated by j*rots_per_thread.
        for i in 0..4 {
            checkpoint_cts[i].push(cts[i].clone());
        }

        // No rotations are needed anymore once checkpoints for last thread have been stored.
        if j != (threads_by_4 as usize) - 1 {
            // rotate the ciphertexts till next checkpoint
            cts.par_iter_mut().for_each(|ct| {
                for _ in 0..rots_per_thread {
                    *ct = evaluator.rotate(&ct, 1, ek);
                }
            });
        }
    }

    checkpoint_cts
}

/// Rotates `s` for `sec_len` times. After i^th rotation multiplies the result with plaintext at i^th index in hint_a_pts and adds the
/// result to final sum. Function takes advantage of the assumption that modulus is smaller than 50 bits to speed up fused mutliplication
/// and additions using 128 bit arithmetic, that is without modulur reduction. Returns coefficients of ciphertext polynomials without modular
/// reduction.
pub fn optimised_pvw_fma_with_rot(
    params: &BfvParameters,
    s: &Ciphertext,
    hint_a_pts: &[Plaintext],
    sec_len: usize,
    rtg: &GaloisKey,
    sk: &SecretKey,
) -> (Array2<u128>, Array2<u128>) {
    debug_assert!(sec_len <= 512);

    let shape = s.c_ref()[0].coefficients().shape();
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
    (d_u128, d1_u128)
}

/// Calls optimised_pvw_fma_with_rot and reduces the 128bit coefficients of ciphertext polynomials using 128 bit barrett reduction
/// and returns the ciphertext.
pub fn optimised_pvw_fma_with_rot_and_reduction(
    params: &BfvParameters,
    s: &Ciphertext,
    hint_a_pts: &[Plaintext],
    sec_len: usize,
    rtg: &GaloisKey,
    sk: &SecretKey,
) -> Ciphertext {
    let (d_u128, d1_u128) = optimised_pvw_fma_with_rot(params, s, hint_a_pts, sec_len, rtg, sk);
    coefficient_u128_to_ciphertext(params, &d_u128, &d1_u128, s.level())
}

/// pvw_decrypt can only use 4 cores at once.
pub fn pvw_decrypt(
    pvw_params: &PvwParameters,
    evaluator: &Evaluator,
    hint_a_pts: &[Plaintext],
    hint_b_pts: &[Poly],
    pvw_sk_cts: &[Ciphertext],
    rtg: &GaloisKey,
    sk: &SecretKey,
) -> Vec<Ciphertext> {
    let sec_len = pvw_params.n.next_power_of_two();
    assert!(hint_a_pts.len() == sec_len);
    assert!(hint_b_pts.len() == pvw_params.ell);
    assert!(pvw_sk_cts.len() == pvw_params.ell);

    let mut sk_a = vec![];
    pvw_sk_cts
        .into_par_iter()
        .map(|s_ct| {
            // s_ct must be in Evaluation for efficient rotations and plaintext multiplication
            assert!(s_ct.c_ref()[0].representation() == &Representation::Evaluation);
            optimised_pvw_fma_with_rot_and_reduction(
                evaluator.params(),
                s_ct,
                hint_a_pts,
                sec_len,
                rtg,
                sk,
            )
        })
        .collect_into_vec(&mut sk_a);

    sk_a.iter_mut().zip(hint_b_pts.iter()).for_each(|(sa, b)| {
        evaluator.sub_ciphertext_from_poly_inplace(sa, b);
    });

    sk_a
}

fn add_array_u128(a: &mut Array2<u128>, b: &Array2<u128>) {
    izip!(a.outer_iter_mut(), b.outer_iter()).for_each(|(mut a0, b0)| {
        izip!(a0.iter_mut(), b0.iter()).for_each(|(r, s)| {
            *r += *s;
        });
    });
}

/// Assigns precomputed sk_cts and corresponding hint_a_pts to available threads.
///
/// Recursively calls itself until it narrows down to a single start index. After which it calls
/// `optimised_pvw_fma_with_rot` for sk_ct with corresponding slice of hint_a_pts. Correspondence is
/// determined by the index of sk_ct, which infact means the number of times it has been rotated during pre-computation. For ex,
/// the sk_ct at position 1 must be matched with slice chunk of hint_a_pts offset by `rots_per_thread*1` since
/// first `rots_per_thread` pts are for sk_ct at position 0.
///
/// Takes care of the case when available threads (ie threads_by_4) does not divide 512 (ie total no. of rotations).
/// For ex, when threads_by_4 = 11, it means there will be 45 rotations on first 10 threads and 62 rotations
/// on the last thread. (Note: no need to handle case threads_by_4%2 !=0 anymore)
fn thread_helper(
    size: usize,
    params: &BfvParameters,
    sk_cts: &[Ciphertext],
    pts: &[Plaintext],
    rtg: &GaloisKey,
    sk: &SecretKey,
) -> (Array2<u128>, Array2<u128>) {
    if sk_cts.len() == 1 {
        let s = &sk_cts[0];
        println!(" len: {}", pts.len());
        optimised_pvw_fma_with_rot(params, s, pts, pts.len(), rtg, sk)
    } else {
        let mid = sk_cts.len() / 2;

        let (mut r0, r1) = rayon::join(
            || thread_helper(size, params, &sk_cts[..mid], &pts[..mid * size], rtg, sk),
            || thread_helper(size, params, &sk_cts[mid..], &pts[mid * size..], rtg, sk),
        );

        add_array_u128(&mut r0.0, &r1.0);
        add_array_u128(&mut r0.1, &r1.1);

        r0
    }
}

pub fn pvw_decrypt_precomputed(
    pvw_params: &PvwParameters,
    evaluator: &Evaluator,
    hint_a_pts: &[Plaintext],
    hint_b_pts: &[Poly],
    precomputed_pvw_sk_cts: &[Vec<Ciphertext>],
    rtg: &GaloisKey,
    sk: &SecretKey,
) -> Vec<Ciphertext> {
    let sec_len = pvw_params.n.next_power_of_two();
    assert!(pvw_params.ell == 4);
    assert!(hint_a_pts.len() == sec_len);
    assert!(hint_b_pts.len() == pvw_params.ell);
    assert!(precomputed_pvw_sk_cts.len() == pvw_params.ell);

    let threads = rayon::current_num_threads();
    assert!(threads % 4 == 0);
    let threads_by_4 = threads / 4;
    // Validate that number of checkpoints are equals disbuted among all threads_by_4.
    // This means precomputed rotations of sk_ct must be equal to threads_by_4.
    assert!(precomputed_pvw_sk_cts[0].len() == threads_by_4);
    assert!(precomputed_pvw_sk_cts[1].len() == threads_by_4);
    assert!(precomputed_pvw_sk_cts[2].len() == threads_by_4);
    assert!(precomputed_pvw_sk_cts[3].len() == threads_by_4);

    let rots_per_thread = (512.0 / threads_by_4 as f64).floor() as usize;

    let mut sk_a = vec![];
    (0..4)
        .into_par_iter()
        .map(|index| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads_by_4)
                .build()
                .unwrap();
            pool.install(|| {
                let (d0, d1) = thread_helper(
                    rots_per_thread,
                    evaluator.params(),
                    &precomputed_pvw_sk_cts[index],
                    &hint_a_pts,
                    rtg,
                    sk,
                );
                coefficient_u128_to_ciphertext(evaluator.params(), &d0, &d1, 0)
            })
        })
        .collect_into_vec(&mut sk_a);

    sk_a.iter_mut().zip(hint_b_pts.iter()).for_each(|(sa, b)| {
        // b - s0
        evaluator.sub_ciphertext_from_poly_inplace(sa, b);
    });

    sk_a
}

#[cfg(test)]
mod tests {

    fn helper(v: &[u64], pts: &[u64], size: usize) {
        if v.len() == 1 {
            // let sk_ct = &v[start];
            // let chunk_pts = &pts[start * size..];

            println!("{}", pts.len());
        } else {
            let mid = v.len() / 2;
            helper(&v[..mid], &pts[..size * mid], size);
            helper(&v[mid..], &pts[size * mid..], size);
        }
    }

    #[test]
    fn caller() {
        let v = [0; 5];
        let pts = [0; 512];
        helper(&v, &pts, 102);
    }
}
