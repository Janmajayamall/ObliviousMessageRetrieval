use crate::pvw::{PvwCiphertext, PvwParameters};
use bfv::{
    BfvParameters, Ciphertext, Encoding, GaloisKey, Modulus, Plaintext, Poly, RelinearizationKey,
    Representation, SecretKey,
};
use itertools::{izip, Itertools};
use ndarray::{azip, s, Array2, IntoNdProducer};
use num_bigint_dig::{BigUint, ModInverse, ToBigUint};
use num_traits::ToPrimitive;
use rand::thread_rng;
use std::{hint, sync::Arc, task::Poll};

/// Barrett reduction of coefficients in u128 to u64
pub fn barret_reduce_coefficients_u128(r_u128: &Array2<u128>, modq: &[Modulus]) -> Array2<u64> {
    let v = r_u128
        .outer_iter()
        .zip(modq.iter())
        .flat_map(|(r0_u128, modqi)| modqi.barret_reduction_u128_vec(r0_u128.as_slice().unwrap()))
        .collect_vec();

    Array2::from_shape_vec((r_u128.shape()[0], r_u128.shape()[1]), v).unwrap()
}

/// We can precompute these and store them somewhere. No need to change them until we change parameters.
pub fn sub_from_one_precompute(params: &Arc<BfvParameters>, level: usize) -> Vec<u64> {
    let ctx = params.ciphertext_ctx_at_level(level);
    let q = ctx.modulus_dig();
    let q_mod_t = &q % params.plaintext_modulus;
    let neg_t_inv_modq = (&q - params.plaintext_modulus)
        .mod_inverse(ctx.modulus_dig())
        .unwrap()
        .to_biguint()
        .unwrap();
    let res = (q_mod_t * neg_t_inv_modq) % &q;

    ctx.moduli
        .iter()
        .map(|qi| (&res % qi).to_u64().unwrap())
        .collect_vec()
}

/// Say that you want to encode a plaintext pt in SIMD format. You must follow the following steps:
/// 1. INTT(plaintext)
/// 2. Matrix mapping (to enable rotations)
/// 3. To add/sub resulting pt with ct, you must scale pt, ie calculate [Q/t pt]_Q. (Using remark 3.1 of 2021/204)
/// To scale, we take coefficients of pt and calculate r = [Q*pt]_t and then calculate v = [r*((-t)^-1)]_Q.
///
/// Notice that if pt = [1,1,..], then INTT([1,1,..]) = [1,0,0,..]. Thus our pt polynomial = [1,0,0,...].
/// Matrix mapping of index 0 is 0, causing nothing to change. To scale, we simply need to calculate [[Q]_t * -t_inv]_Q
/// and set that as 0th index coefficient. Hence, scaled_pt_poly = [[[Q]_t * t_inv]_Q, 0, 0, ...]. If the ciphertext ct is in
/// coefficient form, then you can simply reduce (optimisation!) calculating pt(1) - ct to `([[Q]_t * -t_inv]_Q - ct[0]) % Q`. Therefore
/// instead of `degree` modulus subtraction, we do 1 + `degree - 1` subtraction.
pub fn sub_from_one(ct: &mut Ciphertext, precomputes: &[u64]) {
    debug_assert!(ct.c_ref()[0].representation == Representation::Coefficient);

    let ctx = ct.params().ciphertext_ctx_at_level(ct.level());
    debug_assert!(ctx == ct.c_ref()[0].context);

    azip!(
        ct.c_ref_mut()[0].coefficients.outer_iter_mut(),
        ctx.moduli.into_producer(),
        precomputes.into_producer(),
    )
    .for_each(|mut coeffs, qi, scalar| {
        // modulus subtraction for first coefficient
        let r = &mut coeffs[0];
        if scalar > r {
            *r = scalar - *r;
        } else {
            *r = scalar + qi - *r;
        }

        coeffs.iter_mut().skip(1).for_each(|c| {
            *c = *qi - *c;
        })
    });

    azip!(
        ct.c_ref_mut()[1].coefficients.outer_iter_mut(),
        ctx.moduli.into_producer(),
    )
    .for_each(|mut coeffs, qi| {
        coeffs.iter_mut().skip(0).for_each(|c| {
            *c = *qi - *c;
        })
    });
}

pub fn mul_u128_vec(a: &[u64], b: &[u64]) -> Vec<u64> {
    todo!()
}

pub fn fma_reverse_u128_vec(a: &mut [u128], b: &[u64], c: &[u64]) {
    izip!(a.iter_mut(), b.iter(), c.iter()).for_each(|(a0, b0, c0)| {
        *a0 += *b0 as u128 * *c0 as u128;
    });
}

pub fn fma_reverse_u128_poly(d: &mut Array2<u128>, s: &Poly, h: &Poly) {
    debug_assert!(s.representation == h.representation);
    debug_assert!(s.representation == Representation::Evaluation);
    debug_assert!(s.context == h.context);
    debug_assert!(d.shape() == s.coefficients.shape());

    azip!(
        d.outer_iter_mut(),
        s.coefficients.outer_iter(),
        h.coefficients.outer_iter()
    )
    .for_each(|mut d, a, b| {
        fma_reverse_u128_vec(
            d.as_slice_mut().unwrap(),
            a.as_slice().unwrap(),
            b.as_slice().unwrap(),
        );
    });
}

/// Instead of reading pre-computated rotations from disk this fn rotates `s` which is
/// more expensive than reading them from disk.
pub fn optimised_fma_with_rot(
    mut s: Ciphertext,
    hint_a_pts: &[Plaintext],
    sec_len: usize,
    rtk: &GaloisKey,
) -> Ciphertext {
    // only works and sec_len <= 512 otherwise overflows
    debug_assert!(sec_len <= 512);

    let ctx = s.c_ref()[0].context.clone();
    // let mut d = Poly::zero(&ctx, &Representation::Evaluation);
    let mut d_u128 = ndarray::Array2::<u128>::zeros((ctx.moduli.len(), ctx.degree));
    let mut d1_u128 = ndarray::Array2::<u128>::zeros((ctx.moduli.len(), ctx.degree));
    for i in 0..sec_len {
        // dbg!(i);
        fma_reverse_u128_poly(&mut d_u128, &s.c_ref()[0], hint_a_pts[i].poly_ntt_ref());
        fma_reverse_u128_poly(&mut d1_u128, &s.c_ref()[1], hint_a_pts[i].poly_ntt_ref());
        s = rtk.rotate(&s);
    }

    let d = Poly::new(
        barret_reduce_coefficients_u128(&d_u128, &ctx.moduli_ops),
        &ctx,
        Representation::Evaluation,
    );
    let d1 = Poly::new(
        barret_reduce_coefficients_u128(&d1_u128, &ctx.moduli_ops),
        &ctx,
        Representation::Evaluation,
    );

    Ciphertext::new(vec![d, d1], s.params(), s.level())
}

/// Modify this to accept `s` and `hints_pts` as array of file locations instead of ciphertexts.
/// I don't want to read all 512 rotations of `s` in memory at once since each ciphertext is huge.
pub fn optimised_fma(s: &Ciphertext, hint_a_pts: &[Plaintext], sec_len: usize) -> Ciphertext {
    // only works and sec_len <= 512 otherwise overflows
    debug_assert!(sec_len <= 512);

    let ctx = s.c_ref()[0].context.clone();
    let mut d_u128 = ndarray::Array2::<u128>::zeros((ctx.moduli.len(), ctx.degree));
    let mut d1_u128 = ndarray::Array2::<u128>::zeros((ctx.moduli.len(), ctx.degree));
    for i in 0..sec_len {
        fma_reverse_u128_poly(&mut d_u128, &s.c_ref()[0], hint_a_pts[i].poly_ntt_ref());
        fma_reverse_u128_poly(&mut d1_u128, &s.c_ref()[1], hint_a_pts[i].poly_ntt_ref());
    }

    let d = Poly::new(
        barret_reduce_coefficients_u128(&d_u128, &ctx.moduli_ops),
        &ctx,
        Representation::Evaluation,
    );
    let d1 = Poly::new(
        barret_reduce_coefficients_u128(&d1_u128, &ctx.moduli_ops),
        &ctx,
        Representation::Evaluation,
    );

    Ciphertext::new(vec![d, d1], s.params(), s.level())
}

/// r0 += a0 * s
pub fn scalar_mul_u128(r: &mut [u128], a: &[u64], s: u64) {
    let s_u128 = s as u128;
    r.iter_mut().zip(a.iter()).for_each(|(r0, a0)| {
        *r0 += *a0 as u128 * s_u128;
    })
}

/// ciphertext and a vector of u64
pub fn optmised_range_fn_fma(
    res0: &mut Array2<u128>,
    res1: &mut Array2<u128>,
    ct: &Ciphertext,
    scalar_reduced: &[u64],
) {
    debug_assert!(ct.c_ref()[0].representation == Representation::Evaluation);

    azip!(
        res0.outer_iter_mut(),
        ct.c_ref()[0].coefficients.outer_iter(),
        scalar_reduced.into_producer()
    )
    .for_each(|mut r, a, s| {
        scalar_mul_u128(r.as_slice_mut().unwrap(), a.as_slice().unwrap(), *s);
    });
    azip!(
        res1.outer_iter_mut(),
        ct.c_ref()[1].coefficients.outer_iter(),
        scalar_reduced.into_producer()
    )
    .for_each(|mut r, a, s| {
        scalar_mul_u128(r.as_slice_mut().unwrap(), a.as_slice().unwrap(), *s);
    });
}

pub fn add_u128(r: &mut [u128], a: &[u64]) {
    r.iter_mut().zip(a.iter()).for_each(|(r0, a0)| {
        *r0 += *a0 as u128;
    })
}

/// A lot slower than naively adding ciphertexts because u128 additions are
/// lot more expensive than 1 u64 add + 1 u64 cmp (atleast on m1).
pub fn optimised_add_range_fn(res: &mut Array2<u128>, p: &Poly) {
    azip!(res.outer_iter_mut(), p.coefficients.outer_iter(),).for_each(|mut r, a| {
        add_u128(r.as_slice_mut().unwrap(), a.as_slice().unwrap());
    });
}

#[cfg(test)]
mod tests {
    use crate::utils::{precompute_range_constants, read_range_coeffs};

    use super::*;

    #[test]
    fn optimised_fma_works() {
        let mut rng = thread_rng();
        let params = Arc::new(BfvParameters::default(15, 1 << 15));

        let sk = SecretKey::random(&params, &mut rng);

        let mut m = params
            .plaintext_modulus_op
            .random_vec(params.polynomial_degree, &mut rng);
        let mut m2 = params
            .plaintext_modulus_op
            .random_vec(params.polynomial_degree, &mut rng);
        let pt = Plaintext::encode(&m, &params, Encoding::simd(0));
        let pt2 = Plaintext::encode(&m2, &params, Encoding::simd(0));

        let mut ct = sk.encrypt(&pt, &mut rng);
        ct.change_representation(&Representation::Evaluation);
        let pt_vec = vec![pt2; 512];

        // warmup
        (0..4).for_each(|_| {
            optimised_fma(&ct, &pt_vec, pt_vec.len());
        });

        let now = std::time::Instant::now();
        let res_opt = optimised_fma(&ct, &pt_vec, pt_vec.len());
        println!("time optimised: {:?}", now.elapsed());

        // unoptimised fma
        let now = std::time::Instant::now();
        let mut res_unopt = &ct * &pt_vec[0];
        pt_vec.iter().skip(1).for_each(|c| {
            res_unopt.fma_reverse_inplace(&ct, c);
        });
        println!("time un-optimised: {:?}", now.elapsed());

        println!(
            "Noise: optimised={} un-optimised={}",
            sk.measure_noise(&res_opt, &mut rng),
            sk.measure_noise(&res_unopt, &mut rng)
        );

        let v = sk.decrypt(&res_opt).decode(Encoding::simd(0));
        let v2 = sk.decrypt(&res_unopt).decode(Encoding::simd(0));

        params.plaintext_modulus_op.mul_mod_fast_vec(&mut m, &m2);
        params
            .plaintext_modulus_op
            .scalar_mul_mod_fast_vec(&mut m, pt_vec.len() as u64);

        assert_eq!(v, m);
        assert_eq!(v, v2);
    }

    #[test]
    fn optimised_add_range_fn_works() {
        let params = Arc::new(BfvParameters::default(12, 1 << 15));
        let mut rng = thread_rng();
        let m = params
            .plaintext_modulus_op
            .random_vec(params.polynomial_degree, &mut rng);
        let pt = Plaintext::encode(&m, &params, Encoding::simd(0));
        let sk = SecretKey::random(&params, &mut rng);
        let cts = (0..256).map(|_| sk.encrypt(&pt, &mut rng)).collect_vec();

        {
            // warm up
            for _ in 0..2 {
                let mut res_unopt = Ciphertext::zero(&params, 0);
                cts.iter().for_each(|c| {
                    if res_unopt.is_zero() {
                        res_unopt = c.clone();
                    } else {
                        res_unopt += c;
                    }
                });
            }
        }

        // unoptimised ciphertext additions
        let now = std::time::Instant::now();
        let mut res_unopt = Ciphertext::zero(&params, 0);
        cts.iter().for_each(|c| {
            if res_unopt.is_zero() {
                res_unopt = c.clone();
            } else {
                res_unopt += c;
            }
        });
        let time_unopt = now.elapsed();

        // optimised ciphertext additions
        let now = std::time::Instant::now();
        let q_ctx = params.ciphertext_ctx_at_level(0);
        let mut r0 = Array2::<u128>::zeros((q_ctx.moduli.len(), q_ctx.degree));
        let mut r1 = Array2::<u128>::zeros((q_ctx.moduli.len(), q_ctx.degree));
        cts.iter().for_each(|c| {
            optimised_add_range_fn(&mut r0, &c.c_ref()[0]);
            optimised_add_range_fn(&mut r1, &c.c_ref()[1]);
        });
        let p0 = Poly::new(
            barret_reduce_coefficients_u128(&r0, &q_ctx.moduli_ops),
            &q_ctx,
            Representation::Coefficient,
        );
        let p1 = Poly::new(
            barret_reduce_coefficients_u128(&r1, &q_ctx.moduli_ops),
            &q_ctx,
            Representation::Coefficient,
        );
        let res_opt = Ciphertext::new(vec![p0, p1], params.clone(), 0);
        let time_opt = now.elapsed();

        println!("Time: Opt={:?}, UnOpt={:?}", time_opt, time_unopt);
        println!(
            "Noise: Opt={:?}, UnOpt={:?}",
            sk.measure_noise(&res_opt, &mut rng),
            sk.measure_noise(&res_unopt, &mut rng),
        );
    }

    #[test]
    fn sub_from_one_works() {
        let params = Arc::new(BfvParameters::default(5, 1 << 8));
        let mut rng = thread_rng();
        let m = params
            .plaintext_modulus_op
            .random_vec(params.polynomial_degree, &mut rng);
        let pt = Plaintext::encode(&m, &params, Encoding::simd(0));
        let sk = SecretKey::random(&params, &mut rng);
        let mut ct = sk.encrypt(&pt, &mut rng);
        let mut ct_clone = ct.clone();

        {
            for _ in 0..10 {
                let mut ct_clone = ct.clone();
                let precomputes = sub_from_one_precompute(&params, 0);
                sub_from_one(&mut ct_clone, &precomputes);
            }
        }

        let precomputes = sub_from_one_precompute(&params, 0);
        let now = std::time::Instant::now();
        sub_from_one(&mut ct, &precomputes);
        let time_opt = now.elapsed();

        let pt = Plaintext::encode(
            &vec![1; params.polynomial_degree],
            &params,
            Encoding::simd(0),
        );
        let mut poly = pt.to_poly();
        poly.change_representation(Representation::Coefficient);
        let now = std::time::Instant::now();
        ct_clone.sub_reversed_inplace(&poly);
        let time_unopt = now.elapsed();

        println!("Time: Opt={:?}, UnOpt={:?}", time_opt, time_unopt);
        println!(
            "Noise: Opt={:?}, UnOpt={:?}",
            sk.measure_noise(&ct, &mut rng),
            sk.measure_noise(&ct_clone, &mut rng),
        );
    }

    #[test]
    fn test_optimised_range_fn_fma() {
        let mut rng = thread_rng();
        let params = Arc::new(BfvParameters::default(15, 1 << 15));
        let m = params
            .plaintext_modulus_op
            .random_vec(params.polynomial_degree, &mut rng);
        let sk = SecretKey::random(&params, &mut rng);
        let pt = Plaintext::encode(&m, &params, Encoding::simd(0));
        let mut ct = sk.encrypt(&pt, &mut rng);
        // change ct representation to Evaluation for plaintext mul
        ct.change_representation(&Representation::Evaluation);

        let ctx = params.ciphertext_ctx_at_level(0);
        let constants = precompute_range_constants(&ctx);

        {
            // warmup
            let mut tmp = Ciphertext::zero(&params, 0);
            for j in 0..300 {
                if j == 1 {
                    tmp = &ct * &pt;
                } else {
                    tmp += &(&ct * &pt);
                }
            }
        }

        // optimised version
        let now = std::time::Instant::now();
        let mut res0_u128 = Array2::<u128>::zeros((ctx.moduli.len(), ctx.degree));
        let mut res1_u128 = Array2::<u128>::zeros((ctx.moduli.len(), ctx.degree));
        for j in 0..256 {
            optmised_range_fn_fma(
                &mut res0_u128,
                &mut res1_u128,
                &ct,
                constants.slice(s![j, ..]).as_slice().unwrap(),
            );
        }
        let p_res0 = Poly::new(
            barret_reduce_coefficients_u128(&res0_u128, &ctx.moduli_ops),
            &ctx,
            Representation::Evaluation,
        );
        let p_res1 = Poly::new(
            barret_reduce_coefficients_u128(&res1_u128, &ctx.moduli_ops),
            &ctx,
            Representation::Evaluation,
        );
        let res_opt = Ciphertext::new(vec![p_res0, p_res1], ct.params(), ct.level());
        let time_opt = now.elapsed();

        // unoptimised version
        let range_coeffs = read_range_coeffs();
        // prepare range coefficients plaintext
        let pts = (0..256)
            .map(|i| {
                let c = range_coeffs[i];
                let m = vec![c; ctx.degree];
                Plaintext::encode(&m, &params, Encoding::simd(ct.level))
            })
            .collect_vec();
        let now = std::time::Instant::now();
        let mut res_unopt = Ciphertext::zero(&params, ct.level());
        for j in 1..257 {
            if j == 1 {
                res_unopt = &ct * &pts[j - 1];
            } else {
                res_unopt += &(&ct * &pts[j - 1]);
            }
        }
        let time_unopt = now.elapsed();

        println!("Time: Opt={:?}, UnOpt={:?}", time_opt, time_unopt);
        println!(
            "Noise: Opt={:?}, UnOpt={:?}",
            sk.measure_noise(&res_opt, &mut rng),
            sk.measure_noise(&res_unopt, &mut rng),
        );
    }
}
