use crate::{
    optimised::{optimised_add_range_fn, optimised_fma_with_rot, optmised_range_fn_fma},
    pvw::{self, PvwCiphertext, PvwParameters},
};
use bfv::{
    BfvParameters, Ciphertext, Encoding, GaloisKey, Modulus, Plaintext, Poly, RelinearizationKey,
    Representation, SecretKey,
};
use itertools::{izip, Itertools};
use ndarray::{azip, s, Array2, IntoNdProducer};
use rand::thread_rng;
use rand_chacha::rand_core::le;
use std::{hint, sync::Arc, task::Poll};

pub fn pre_process_batch(
    pvw_params: &Arc<PvwParameters>,
    bfv_params: Arc<BfvParameters>,
    hints: &[PvwCiphertext],
) -> (Vec<Plaintext>, Vec<Poly>) {
    // can only process as many as polynomial_degree hints in a batch
    debug_assert!(hints.len() <= bfv_params.polynomial_degree);

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
        hint_a_pts.push(Plaintext::encode(&m, &bfv_params, Encoding::simd(0)));
    }

    let mut hint_b_polys = vec![];
    let q_by4 = bfv_params.plaintext_modulus / 4;
    for i in 0..pvw_params.ell {
        let mut m = vec![];
        for j in 0..hints.len() {
            m.push(
                bfv_params
                    .plaintext_modulus_op
                    .sub_mod_fast(hints[j].b[i], q_by4),
            );
        }
        hint_b_polys.push(Plaintext::encode(&m, &bfv_params, Encoding::simd(0)).to_poly());
    }

    // length of plaintexts will be sec_len
    (hint_a_pts, hint_b_polys)
}

// rotate by 1 and perform plaintext mutiplication for each ell
pub fn pvw_decrypt(
    pvw_params: &Arc<PvwParameters>,
    hint_a_pts: &[Plaintext],
    hint_b_pts: &[Poly],
    pvw_sk_cts: Vec<Ciphertext>,
    rtk: &GaloisKey,
) -> Vec<Ciphertext> {
    let sec_len = pvw_params.n.next_power_of_two();
    debug_assert!(hint_a_pts.len() == sec_len);
    debug_assert!(hint_b_pts.len() == pvw_params.ell);
    debug_assert!(pvw_sk_cts.len() == pvw_params.ell);

    // d[j] = s[j][0] * p[0] + s[j][1] * p[1] + ... + s[j][sec_len-1] * p[sec_len-1]
    // where s[j][a] is s[j] rotated to left by 1 `a` times.
    // Each operation is further broken down to: d[j] += s[j][0] * p[0]. Can we take
    // advantage of fused multiplication addition to speed this up? For ex, hexl has
    // an API for FMA which is faster (should be right?) than perfoming vector multiplication
    // and addition in a sequence.
    // There's an additinal optimisation for FMA operation. We can perform FMA in 128 bits without
    // modulur reduction followed by 128 bit barret reduction in the end. Since we will only be adding 512 128 bits values,
    // result will not overflow. Amazing!
    // TODO: Provide and API for FMA (ie Ct + Ct * Pt) in Bfv.
    //
    // Length of `d == ell`.
    // let mut d = vec![];
    let mut sk_a = pvw_sk_cts
        .into_iter()
        .map(|s_ct| optimised_fma_with_rot(s_ct, hint_a_pts, sec_len, rtk))
        .collect_vec();

    sk_a.iter_mut().zip(hint_b_pts.iter()).for_each(|(sa, b)| {
        sa.sub_reversed_inplace(b);
    });

    sk_a
}

// TODO: remove clones
pub fn powers_of_x_ct(x: &Ciphertext, rlk: &RelinearizationKey) -> Vec<Ciphertext> {
    let mut values = vec![Ciphertext::zero(&x.params(), x.level()); 256];
    let mut calculated = vec![0u64; 256];
    let mut mul_count = 0;
    for i in (2..257).rev() {
        let mut exp = i;
        let mut base = x.clone();
        let mut res = Ciphertext::zero(&x.params(), x.level());
        let mut base_deg = 1;
        let mut res_deg = 0;

        while exp > 0 {
            if exp & 1 == 1 {
                res_deg += base_deg;

                if calculated[res_deg - 1] == 1 {
                    res = values[res_deg - 1].clone();
                } else {
                    // Covers the case when res is Ciphertext::zero
                    if res_deg == base_deg {
                        res = base.clone();
                    } else {
                        mul_count += 1;
                        let tmp = res.multiply1(&base);
                        res = rlk.relinearize(&tmp);
                    }
                    values[res_deg - 1] = res.clone();
                    calculated[res_deg - 1] = 1;
                }
            }
            exp >>= 1;
            if exp != 0 {
                base_deg *= 2;

                if calculated[base_deg - 1] == 1 {
                    base = values[base_deg - 1].clone();
                } else {
                    mul_count += 1;
                    let tmp = base.multiply1(&base);
                    values[base_deg - 1] = rlk.relinearize(&tmp);
                    base = values[base_deg - 1].clone();
                    calculated[base_deg - 1] = 1;
                }
            }
        }
    }
    // dbg!(mul_count);
    values
}

pub fn range_fn(ct: &Ciphertext, rlk: &RelinearizationKey, constants: &Array2<u64>) {
    let mut single_powers = powers_of_x_ct(ct, rlk);
    let double_powers = powers_of_x_ct(&single_powers[255], rlk);

    // change to evaluation for plaintext multiplication
    single_powers.iter_mut().for_each(|ct| {
        ct.change_representation(&Representation::Evaluation);
    });

    let level = 0;
    let bfv_params = ct.params();
    let q_ctx = bfv_params.ciphertext_ctx_at_level(level);
    let qp_ctx = bfv_params.extension_poly_contexts[level].clone();
    let q_size = q_ctx.moduli.len();
    let qp_size = qp_ctx.moduli.len();

    // when i = 0, we skip multiplication and cache the result
    let mut left_over_ct = Ciphertext::zero(&bfv_params, level);

    let mut sum_res0_u128 = Array2::<u128>::zeros((qp_size, ct.params().polynomial_degree));
    let mut sum_res1_u128 = Array2::<u128>::zeros((qp_size, ct.params().polynomial_degree));
    let mut sum_res2_u128 = Array2::<u128>::zeros((qp_size, ct.params().polynomial_degree));

    for i in 0..256 {
        let mut res0_u128 = Array2::<u128>::zeros((q_size, ct.params().polynomial_degree));
        let mut res1_u128 = Array2::<u128>::zeros((q_size, ct.params().polynomial_degree));
        for j in 1..257 {
            optmised_range_fn_fma(
                &mut res0_u128,
                &mut res1_u128,
                &single_powers[j - 1],
                constants
                    .slice(s![(i * 256) + (j - 1), ..])
                    .as_slice()
                    .unwrap(),
            );
        }

        let mut res0 = Array2::<u64>::zeros((q_size, ct.params().polynomial_degree));
        let mut res1 = Array2::<u64>::zeros((q_size, ct.params().polynomial_degree));

        azip!(
            res0.outer_iter_mut(),
            res1.outer_iter_mut(),
            res0_u128.outer_iter(),
            res1_u128.outer_iter(),
            q_ctx.moduli_ops.into_producer()
        )
        .for_each(|mut r0, mut r1, r0_u128, r1_u128, modqi| {
            r0.as_slice_mut()
                .unwrap()
                .copy_from_slice(&modqi.barret_reduction_u128_vec(r0_u128.as_slice().unwrap()));
            r1.as_slice_mut()
                .unwrap()
                .copy_from_slice(&modqi.barret_reduction_u128_vec(r1_u128.as_slice().unwrap()));
        });

        let p0 = Poly::new(res0, &q_ctx, Representation::Evaluation);
        let p1 = Poly::new(res1, &q_ctx, Representation::Evaluation);

        let res_ct = Ciphertext::new(vec![p0, p1], ct.params(), level);

        // cache i == 0
        if i == 0 {
            left_over_ct = res_ct;
            // convert  ct to coefficient form
            left_over_ct.change_representation(&Representation::Coefficient);
        } else {
            // multiply1_lazy returns in evaluation form
            let product = res_ct.multiply1_lazy(&double_powers[i - 1]);
            optimised_add_range_fn(&mut sum_res0_u128, &product.c_ref()[0]);
            optimised_add_range_fn(&mut sum_res1_u128, &product.c_ref()[1]);
            optimised_add_range_fn(&mut sum_res2_u128, &product.c_ref()[2]);
        }
    }

    let mut sum_res0 = Poly::new(
        barret_reduce_tmp(&sum_res0_u128, &qp_ctx.moduli_ops),
        &qp_ctx,
        Representation::Evaluation,
    );
    let mut sum_res1 = Poly::new(
        barret_reduce_tmp(&sum_res1_u128, &qp_ctx.moduli_ops),
        &qp_ctx,
        Representation::Evaluation,
    );
    let mut sum_res2 = Poly::new(
        barret_reduce_tmp(&sum_res2_u128, &qp_ctx.moduli_ops),
        &qp_ctx,
        Representation::Evaluation,
    );

    let mut sum_ct = Ciphertext::new(vec![sum_res0, sum_res1, sum_res2], bfv_params, level);

    sum_ct.scale_and_round();
    sum_ct += &left_over_ct;

    // implement optimised of 1 - sum_ct
}

fn barret_reduce_tmp(r_u128: &Array2<u128>, modq: &[Modulus]) -> Array2<u64> {
    let v = r_u128
        .outer_iter()
        .zip(modq.iter())
        .flat_map(|(r0_u128, modqi)| modqi.barret_reduction_u128_vec(r0_u128.as_slice().unwrap()))
        .collect_vec();

    Array2::from_shape_vec((r_u128.shape()[0], r_u128.shape()[1]), v).unwrap()
}
