use crate::pvw::{PvwCiphertext, PvwParameters};
use bfv::{
    mod_inverse_biguint, BfvParameters, Ciphertext, Encoding, EvaluationKey, GaloisKey, Modulus,
    Plaintext, Poly, PolyContext, PolyType, RelinearizationKey, Representation, SecretKey,
};
use itertools::{izip, Itertools};
use ndarray::{azip, s, Array1, Array2, IntoNdProducer};
use num_bigint_dig::{BigUint, ModInverse, ToBigUint};
use num_traits::ToPrimitive;
use rand::thread_rng;

/// Barrett reduction of coefficients in u128 to u64
pub fn barret_reduce_coefficients_u128(r_u128: &Array2<u128>, modq: &[Modulus]) -> Array2<u64> {
    let v = r_u128
        .outer_iter()
        .zip(modq.iter())
        .flat_map(|(r0_u128, modqi)| modqi.barret_reduction_u128_vec(r0_u128.as_slice().unwrap()))
        .collect_vec();

    Array2::from_shape_vec((r_u128.shape()[0], r_u128.shape()[1]), v).unwrap()
}

pub fn coefficient_u128_to_ciphertext(
    params: &BfvParameters,
    c0_coeffs: &Array2<u128>,
    c1_coeffs: &Array2<u128>,
    level: usize,
) -> Ciphertext {
    let ct_ctx = params.poly_ctx(&PolyType::Q, level);

    Ciphertext::new(
        vec![
            Poly::new(
                barret_reduce_coefficients_u128(&c0_coeffs, ct_ctx.moduli_ops()),
                Representation::Evaluation,
            ),
            Poly::new(
                barret_reduce_coefficients_u128(&c1_coeffs, ct_ctx.moduli_ops()),
                Representation::Evaluation,
            ),
        ],
        PolyType::Q,
        level,
    )
}

/// We can precompute these and store them somewhere. No need to change them until we change parameters.
pub fn sub_from_one_precompute(params: &BfvParameters, level: usize) -> Vec<u64> {
    let ctx = params.poly_ctx(&PolyType::Q, level);
    let q = ctx.big_q();
    let q_mod_t = &q % params.plaintext_modulus;
    let neg_t_inv_modq = mod_inverse_biguint(&(&q - params.plaintext_modulus), &q);
    let res = (q_mod_t * neg_t_inv_modq) % &q;

    ctx.iter_moduli_ops()
        .map(|modqi| (&res % modqi.modulus()).to_u64().unwrap())
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
pub fn sub_from_one(params: &BfvParameters, ct: &mut Ciphertext, precomputes: &[u64]) {
    debug_assert!(ct.c_ref()[0].representation == Representation::Coefficient);

    let ctx = params.poly_ctx(&ct.poly_type, ct.level);
    debug_assert!(precomputes.len() == ctx.moduli_count());

    izip!(
        ct.c_ref_mut()[0].coefficients.outer_iter_mut(),
        ctx.moduli_ops().iter(),
        precomputes.iter(),
    )
    .for_each(|(mut coeffs, modqi, scalar)| {
        let qi = modqi.modulus();
        // modulus subtraction for first coefficient
        let r = &mut coeffs[0];
        if scalar > r {
            *r = scalar - *r;
        } else {
            *r = scalar + qi - *r;
        }

        coeffs.iter_mut().skip(1).for_each(|c| {
            *c = qi - *c;
        })
    });

    izip!(
        ct.c_ref_mut()[1].coefficients.outer_iter_mut(),
        ctx.moduli_ops().iter(),
    )
    .for_each(|(mut coeffs, modqi)| {
        let qi = modqi.modulus();
        coeffs.iter_mut().for_each(|c| {
            *c = qi - *c;
        })
    });
}

/// r0 += a0 * s
pub fn scalar_mul_u128(r: &mut [u128], a: &[u64], s: u64) {
    let s_u128 = s as u128;
    r.iter_mut().zip(a.iter()).for_each(|(r0, a0)| {
        *r0 += *a0 as u128 * s_u128;
    })
}

pub fn fma_reverse_u128_vec(a: &mut [u128], b: &[u64], c: &[u64]) {
    izip!(a.iter_mut(), b.iter(), c.iter()).for_each(|(a0, b0, c0)| {
        *a0 += *b0 as u128 * *c0 as u128;
    });
}

pub fn fma_reverse_u128_poly(d: &mut Array2<u128>, s: &Poly, h: &Poly) {
    debug_assert!(s.representation == h.representation);
    debug_assert!(s.representation == Representation::Evaluation);
    debug_assert!(d.shape() == s.coefficients.shape());

    izip!(
        d.outer_iter_mut(),
        s.coefficients.outer_iter(),
        h.coefficients.outer_iter()
    )
    .for_each(|(mut d, a, b)| {
        fma_reverse_u128_vec(
            d.as_slice_mut().unwrap(),
            a.as_slice().unwrap(),
            b.as_slice().unwrap(),
        );
    });
}

/// Instead of reading pre-computated rotations from disk this fn rotates `s` which is
/// more expensive than reading them from disk.
pub fn optimised_pvw_fma_with_rot(
    params: &BfvParameters,
    s: &Ciphertext,
    hint_a_pts: &[Plaintext],
    sec_len: usize,
    rtg: &GaloisKey,
    sk: &SecretKey,
) -> Ciphertext {
    // only works and sec_len <= 512 otherwise overflows
    debug_assert!(sec_len <= 512);

    // let mut d = Poly::zero(&ctx, &Representation::Evaluation);
    let shape = s.c_ref()[0].coefficients.shape();
    let mut d_u128 = ndarray::Array2::<u128>::zeros((shape[0], shape[1]));
    let mut d1_u128 = ndarray::Array2::<u128>::zeros((shape[0], shape[1]));

    // To repeatedly rotate `s` and set output to `s`, `s` must be Ciphertext<T> not its reference.
    // To avoid having to me `s` in function params Ciphertext<T> we perform first plaintext mul
    // outside loop and then set a new `s` after rotating `s` passed in function.
    fma_reverse_u128_poly(&mut d_u128, &s.c_ref()[0], hint_a_pts[0].poly_ntt_ref());
    fma_reverse_u128_poly(&mut d1_u128, &s.c_ref()[1], hint_a_pts[0].poly_ntt_ref());
    let mut s = rtg.rotate(s, params);
    for i in 1..sec_len {
        // dbg!(i);
        fma_reverse_u128_poly(&mut d_u128, &s.c_ref()[0], hint_a_pts[i].poly_ntt_ref());
        fma_reverse_u128_poly(&mut d1_u128, &s.c_ref()[1], hint_a_pts[i].poly_ntt_ref());
        s = rtg.rotate(&s, params);

        // {
        //     let mut rng = thread_rng();
        //     dbg!(sk.measure_noise(&s, &mut rng));
        // }
    }
    coefficient_u128_to_ciphertext(params, &d_u128, &d1_u128, s.level)
}

/// Modify this to accept `s` and `hints_pts` as array of file locations instead of ciphertexts.
/// I don't want to read all 512 rotations of `s` in memory at once since each ciphertext is huge.
pub fn optimised_pvw_fma(
    params: &BfvParameters,
    s: &Ciphertext,
    hint_a_pts: &[Plaintext],
    sec_len: usize,
) -> Ciphertext {
    // only works and sec_len <= 512 otherwise overflows
    debug_assert!(sec_len <= 512);

    let shape = s.c_ref()[0].coefficients.shape();
    let mut d_u128 = ndarray::Array2::<u128>::zeros((shape[0], shape[1]));
    let mut d1_u128 = ndarray::Array2::<u128>::zeros((shape[0], shape[1]));
    for i in 0..sec_len {
        fma_reverse_u128_poly(&mut d_u128, &s.c_ref()[0], hint_a_pts[i].poly_ntt_ref());
        fma_reverse_u128_poly(&mut d1_u128, &s.c_ref()[1], hint_a_pts[i].poly_ntt_ref());
    }

    coefficient_u128_to_ciphertext(params, &d_u128, &d1_u128, s.level)
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

pub fn optimised_poly_fma(
    cts: &[Ciphertext],
    polys: &[Poly],
    res00: &mut Array2<u128>,
    res01: &mut Array2<u128>,
) {
    izip!(cts.iter(), polys.iter()).for_each(|(o, p)| {
        fma_reverse_u128_poly(res00, &o.c_ref()[0], p);
        fma_reverse_u128_poly(res01, &o.c_ref()[1], p);
    });
}

#[cfg(test)]
mod tests {
    use bfv::Evaluator;
    use statrs::function::evaluate;

    use crate::utils::{precompute_range_constants, read_range_coeffs};

    use super::*;

    #[test]
    fn optimised_pvw_fma_works() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(15, 1 << 15);
        let sk = SecretKey::random(params.degree, &mut rng);

        let mut m0 = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);
        let mut m1 = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);

        let evaluator = Evaluator::new(params);
        let pt0 = evaluator.plaintext_encode(&m0, Encoding::default());
        let pt1 = evaluator.plaintext_encode(&m1, Encoding::default());

        let mut ct = evaluator.encrypt(&sk, &pt0, &mut rng);
        evaluator.ciphertext_change_representation(&mut ct, Representation::Evaluation);

        let pt_vec = vec![pt1; 512];

        // warmup
        (0..4).for_each(|_| {
            optimised_pvw_fma(evaluator.params(), &ct, &pt_vec, pt_vec.len());
        });

        let now = std::time::Instant::now();
        let res_opt = optimised_pvw_fma(evaluator.params(), &ct, &pt_vec, pt_vec.len());
        println!("time optimised: {:?}", now.elapsed());

        // unoptimised fma
        let now = std::time::Instant::now();
        let mut res_unopt = evaluator.mul_poly(&ct, pt_vec[0].poly_ntt_ref());
        pt_vec.iter().skip(1).for_each(|p0| {
            evaluator.fma_poly(&mut res_unopt, &ct, p0.poly_ntt_ref());
        });
        println!("time un-optimised: {:?}", now.elapsed());

        println!(
            "Noise: optimised={} un-optimised={}",
            evaluator.measure_noise(&sk, &res_opt),
            evaluator.measure_noise(&sk, &res_unopt)
        );

        let v = evaluator.plaintext_decode(&evaluator.decrypt(&sk, &res_opt), Encoding::default());
        let v2 =
            evaluator.plaintext_decode(&evaluator.decrypt(&sk, &res_unopt), Encoding::default());

        evaluator
            .params()
            .plaintext_modulus_op
            .mul_mod_fast_vec(&mut m0, &m1);
        evaluator
            .params()
            .plaintext_modulus_op
            .scalar_mul_mod_fast_vec(&mut m0, pt_vec.len() as u64);

        assert_eq!(v, m0);
        assert_eq!(v, v2);
    }

    #[test]
    fn optimised_add_range_fn_works() {
        let params = BfvParameters::default(12, 1 << 15);
        let mut rng = thread_rng();
        let m = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);
        let sk = SecretKey::random(params.degree, &mut rng);

        let evaluator = Evaluator::new(params);
        let pt = evaluator.plaintext_encode(&m, Encoding::default());

        let cts = (0..256)
            .map(|_| evaluator.encrypt(&sk, &pt, &mut rng))
            .collect_vec();

        {
            // warm up
            for _ in 0..2 {
                let mut dummy_res = cts[0].clone();
                cts.iter().skip(1).for_each(|c| {
                    evaluator.add(&mut dummy_res, c);
                });
            }
        }

        // unoptimised ciphertext additions
        let now = std::time::Instant::now();
        let mut res_unopt = cts[0].clone();
        cts.iter().skip(1).for_each(|c| {
            evaluator.add(&mut res_unopt, c);
        });
        let time_unopt = now.elapsed();

        // optimised ciphertext additions
        let now = std::time::Instant::now();
        let q_ctx = evaluator.params().poly_ctx(&PolyType::Q, 0);
        let mut r0 = Array2::<u128>::zeros((q_ctx.moduli_count(), q_ctx.degree()));
        let mut r1 = Array2::<u128>::zeros((q_ctx.moduli_count(), q_ctx.degree()));
        cts.iter().for_each(|c| {
            optimised_add_range_fn(&mut r0, &c.c_ref()[0]);
            optimised_add_range_fn(&mut r1, &c.c_ref()[1]);
        });
        let res_opt = coefficient_u128_to_ciphertext(evaluator.params(), &r0, &r1, 0);
        let time_opt = now.elapsed();

        println!("Time: Opt={:?}, UnOpt={:?}", time_opt, time_unopt);
        println!(
            "Noise: Opt={:?}, UnOpt={:?}",
            evaluator.measure_noise(&sk, &res_opt),
            evaluator.measure_noise(&sk, &res_unopt)
        );
    }

    #[test]
    fn sub_from_one_works() {
        let params = BfvParameters::default(5, 1 << 8);
        let mut rng = thread_rng();
        let m = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);
        let sk = SecretKey::random(params.degree, &mut rng);

        let evaluator = Evaluator::new(params);

        let pt = evaluator.plaintext_encode(&m, Encoding::default());
        let mut ct = evaluator.encrypt(&sk, &pt, &mut rng);
        let mut ct_clone = ct.clone();

        {
            for _ in 0..10 {
                let mut ct_clone = ct.clone();
                let precomputes = sub_from_one_precompute(&evaluator.params(), 0);
                sub_from_one(evaluator.params(), &mut ct_clone, &precomputes);
            }
        }

        let precomputes = sub_from_one_precompute(evaluator.params(), 0);
        let now = std::time::Instant::now();
        sub_from_one(evaluator.params(), &mut ct, &precomputes);
        let time_opt = now.elapsed();

        let pt =
            evaluator.plaintext_encode(&vec![1; evaluator.params().degree], Encoding::default());
        let poly = pt.to_poly(evaluator.params(), Representation::Coefficient);
        let now = std::time::Instant::now();
        evaluator.sub_ciphertext_from_poly_inplace(&mut ct_clone, &poly);
        let time_unopt = now.elapsed();

        println!("Time: Opt={:?}, UnOpt={:?}", time_opt, time_unopt);
        println!(
            "Noise: Opt={:?}, UnOpt={:?}",
            evaluator.measure_noise(&sk, &ct),
            evaluator.measure_noise(&sk, &ct_clone),
        );
    }

    #[test]
    fn test_optimised_range_fn_fma() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(15, 1 << 15);
        let m = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);
        let sk = SecretKey::random(params.degree, &mut rng);

        let evaluator = Evaluator::new(params);
        let pt = evaluator.plaintext_encode(&m, Encoding::default());
        let mut ct = evaluator.encrypt(&sk, &pt, &mut rng);
        // change ct representation to Evaluation for plaintext mul
        evaluator.ciphertext_change_representation(&mut ct, Representation::Evaluation);

        let ctx = evaluator.params().poly_ctx(&PolyType::Q, 0);
        let constants = precompute_range_constants(&ctx);

        {
            // warmup
            let mut tmp = evaluator.mul_poly(&ct, pt.poly_ntt_ref());
            for j in 0..300 {
                evaluator.add_assign(&mut tmp, &evaluator.mul_poly(&ct, pt.poly_ntt_ref()));
            }
        }

        // optimised version
        let now = std::time::Instant::now();
        let degree = evaluator.params().degree;
        let mut res0_u128 = Array2::<u128>::zeros((ctx.moduli_count(), degree));
        let mut res1_u128 = Array2::<u128>::zeros((ctx.moduli_count(), degree));
        for j in 0..256 {
            optmised_range_fn_fma(
                &mut res0_u128,
                &mut res1_u128,
                &ct,
                constants.slice(s![j, ..]).as_slice().unwrap(),
            );
        }
        let res_opt =
            coefficient_u128_to_ciphertext(&evaluator.params(), &res0_u128, &res1_u128, ct.level);
        let time_opt = now.elapsed();

        // unoptimised version
        let range_coeffs = read_range_coeffs();
        // prepare range coefficients plaintext
        let pts = (0..256)
            .map(|i| {
                let c = range_coeffs[i];
                let m = vec![c; degree];
                evaluator.plaintext_encode(&m, Encoding::simd(ct.level))
            })
            .collect_vec();
        let now = std::time::Instant::now();
        let mut res_unopt = evaluator.mul_poly(&ct, pts[0].poly_ntt_ref());
        for j in 1..256 {
            evaluator.add_assign(
                &mut res_unopt,
                &evaluator.mul_poly(&ct, pts[j].poly_ntt_ref()),
            );
        }
        let time_unopt = now.elapsed();

        println!("Time: Opt={:?}, UnOpt={:?}", time_opt, time_unopt);
        println!(
            "Noise: Opt={:?}, UnOpt={:?}",
            evaluator.measure_noise(&sk, &res_opt),
            evaluator.measure_noise(&sk, &res_unopt),
        );
    }
}
