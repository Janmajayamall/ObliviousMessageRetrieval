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
    debug_assert!(ct.c_ref()[0].representation() == &Representation::Coefficient);

    let ctx = params.poly_ctx(&ct.poly_type(), ct.level());
    assert!(precomputes.len() == ctx.moduli_count());

    izip!(
        ct.c_ref_mut()[0].coefficients_mut().outer_iter_mut(),
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
        ct.c_ref_mut()[1].coefficients_mut().outer_iter_mut(),
        ctx.moduli_ops().iter(),
    )
    .for_each(|(mut coeffs, modqi)| {
        let qi = modqi.modulus();
        coeffs.iter_mut().for_each(|c| {
            *c = qi - *c;
        })
    });
}

pub fn fma_reverse_u128_vec(a: &mut [u128], b: &[u64], c: &[u64]) {
    izip!(a.iter_mut(), b.iter(), c.iter()).for_each(|(a0, b0, c0)| {
        *a0 += *b0 as u128 * *c0 as u128;
    });
}

pub fn fma_reverse_u128_poly(d: &mut Array2<u128>, s: &Poly, h: &Poly) {
    debug_assert!(s.representation() == h.representation());
    debug_assert!(s.representation() == &Representation::Evaluation);
    debug_assert!(d.shape() == s.coefficients().shape());

    izip!(
        d.outer_iter_mut(),
        s.coefficients().outer_iter(),
        h.coefficients().outer_iter()
    )
    .for_each(|(mut d, a, b)| {
        fma_reverse_u128_vec(
            d.as_slice_mut().unwrap(),
            a.as_slice().unwrap(),
            b.as_slice().unwrap(),
        );
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
    use bfv::{Evaluator, PolyCache};
    use statrs::function::evaluate;

    use crate::utils::{generate_bfv_parameters, precompute_range_constants, read_range_coeffs};

    use super::*;

    #[test]
    fn sub_from_one_works() {
        let params = generate_bfv_parameters();
        let mut rng = thread_rng();
        let m = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);
        let sk = SecretKey::random_with_params(&params, &mut rng);

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

        let pt = evaluator.plaintext_encode(
            &vec![1; evaluator.params().degree],
            Encoding::simd(0, PolyCache::AddSub(Representation::Coefficient)),
        );
        let now = std::time::Instant::now();
        evaluator.sub_ciphertext_from_poly_inplace(&mut ct_clone, &pt.add_sub_poly_ref());
        let time_unopt = now.elapsed();

        println!("Time: Opt={:?}, UnOpt={:?}", time_opt, time_unopt);
        println!(
            "Noise: Opt={:?}, UnOpt={:?}",
            evaluator.measure_noise(&sk, &ct),
            evaluator.measure_noise(&sk, &ct_clone),
        );
    }
}
