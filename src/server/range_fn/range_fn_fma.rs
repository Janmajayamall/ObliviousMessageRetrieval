use crate::optimised::{coefficient_u128_to_ciphertext, fma_reverse_u128_poly};
use crate::preprocessing::{precompute_expand_32_roll_pt, procompute_expand_roll_pt};
use crate::{
    optimised::{barret_reduce_coefficients_u128, optimised_pvw_fma_with_rot, sub_from_one},
    pvw::PvwParameters,
};
use bfv::{
    BfvParameters, Ciphertext, EvaluationKey, Evaluator, GaloisKey, Plaintext, Poly, PolyContext,
    PolyType, RelinearizationKey, Representation, SecretKey,
};
use itertools::{izip, Itertools};
use ndarray::{s, Array2};

/// Performs FMA: r0 += a0 * s
pub fn scalar_mul_u128(r: &mut [u128], a: &[u64], s: u64) {
    let s_u128 = s as u128;
    r.iter_mut().zip(a.iter()).for_each(|(r0, a0)| {
        *r0 += *a0 as u128 * s_u128;
    })
}

/// Performs res[i] += a[i] * scalar[i] where i is index of qi in moduli chain, res is
/// unreduced coefficients corresponding to qi modulus stored in row major form, a[i] are reduced coefficients
/// corresponding to qi modulus stored in row major form, scalar[i] is i^th scalar in scalar_slice.
///
/// Functions calls `scalar_mul_u128` equivalent to moduli count times.
pub fn mul_poly_scalar_u128(res: &mut Array2<u128>, a: &Poly, scalar_slice: &[u64]) {
    izip!(
        res.outer_iter_mut(),
        a.coefficients().outer_iter(),
        scalar_slice.iter(),
    )
    .for_each(|(mut r, a, s)| {
        scalar_mul_u128(r.as_slice_mut().unwrap(), a.as_slice().unwrap(), *s);
    });
}

/// Performs inner k loop of range function. More specifically, function calculates res += single_powers[i] * constants[i]
/// where single_powers[i] is i^th k_power and constants[i] is corresponding constant reduced with respect to moduli chain.
///
/// Function assumes that each qi in moduli chain is <= 50 bits hence modulus reduction can be delayed until last
/// scalar multiplication. Hence modulus vector multiplication and additions are replace with normal vector multiplications, additions.
///
/// In concrete costs, functions call `mul_poly_scalar_u128` * 128 * 2 times.
pub fn optimised_range_fn_fma_u128(
    poly_ctx: &PolyContext<'_>,
    params: &BfvParameters,
    single_powers: &[Ciphertext],
    constants: &Array2<u64>,
    constants_outer_offset: usize,
    level: usize,
) -> Ciphertext {
    let mut res0 = Array2::<u128>::zeros((poly_ctx.moduli_count(), poly_ctx.degree()));
    let mut res1 = Array2::<u128>::zeros((poly_ctx.moduli_count(), poly_ctx.degree()));

    // Starting from 0th index every alternate constant is 0. Since plaintext multiplication by 0 is 0, we don't need to
    // process plaintext multiplications for indices at which constant is 0. Thus, we start from 1st index and process
    // every alternate index.
    for j in (2..257).step_by(2) {
        let power_ct = &single_powers[j / 2 - 1];
        let scalar_reduced = constants.slice(s![constants_outer_offset + (j - 1), ..]);

        mul_poly_scalar_u128(
            &mut res0,
            &power_ct.c_ref()[0],
            scalar_reduced.as_slice().unwrap(),
        );
        mul_poly_scalar_u128(
            &mut res1,
            &power_ct.c_ref()[1],
            scalar_reduced.as_slice().unwrap(),
        );
    }

    coefficient_u128_to_ciphertext(params, &res0, &res1, level)
}

#[cfg(target_arch = "x86_64")]
/// Performs res[i] = a[i] * scalar[i] where i is index of qi in moduli chain, resulting `res` are
/// reduced coefficients corresponding to qi modulus stored in row major form, a[i] are reduced coefficients
/// corresponding to qi modulus stored in row major form, scalar[i] is i^th scalar in scalar_slice.
///
/// Use `fma_poly_scale_slice_hexl` over `fma_poly_scale_slice_u128` on x86 since it performs better
/// assuming that each qi in moduli chain is <=50 bit (ie when hexl uses IFMA instruction set)
///
/// Calls `hexl_rs::elwise_fma_mod` moduli count times.
pub fn mul_poly_scalar_slice_hexl(
    poly_ctx: &PolyContext<'_>,
    res: &mut Poly,
    a: &Poly,
    scalar_slice: &[u64],
) {
    izip!(
        res.coefficients_mut().outer_iter_mut(),
        a.coefficients().outer_iter(),
        scalar_slice.iter(),
        poly_ctx.moduli_ops().iter()
    )
    .for_each(|(mut r, a, s, modqi)| {
        let qi = modqi.modulus();
        hexl_rs::elwise_mult_scalar_mod_2(
            r.as_slice_mut().unwrap(),
            a.as_slice().unwrap(),
            *s,
            qi,
            poly_ctx.degree() as u64,
            1,
        );
    });
}

/// Performs res[i] += a[i] * scalar[i] where i is index of qi in moduli chain, res are
/// reduced coefficients corresponding to qi modulus stored in row major form, a[i] are reduced coefficients
/// corresponding to qi modulus stored in row major form, scalar[i] is i^th scalar in scalar_slice.
///
/// Use `fma_poly_scale_slice_hexl` over `fma_poly_scale_slice_u128` on x86 since it performs better
/// assuming that each qi in moduli chain is <=50 bit (ie when hexl uses IFMA instruction set)
///
/// Calls `hexl_rs::elwise_fma_mod` moduli count times.
#[cfg(target_arch = "x86_64")]
pub fn fma_poly_scale_slice_hexl(
    poly_ctx: &PolyContext<'_>,
    res: &mut Poly,
    a: &Poly,
    scalar_slice: &[u64],
) {
    izip!(
        res.coefficients_mut().outer_iter_mut(),
        a.coefficients().outer_iter(),
        scalar_slice.iter(),
        poly_ctx.moduli_ops().iter()
    )
    .for_each(|(mut r, a, s, modqi)| {
        let qi = modqi.modulus();
        hexl_rs::elwise_fma_mod(
            r.as_slice_mut().unwrap(),
            *s,
            a.as_slice().unwrap(),
            qi,
            poly_ctx.degree() as u64,
            1,
        )
    });
}

/// Performs inner k loop of range function. More specifically, function calculates res += single_powers[i] * constants[i]
/// where single_powers[i] is i^th k_power and constants[i] is corresponding constant reduced with respect to moduli chain.
///
/// Use this function over `optimised_range_fn_fma_u128` on x86 if each qi in moduli chain is <= 50 bites since it performs
/// better by using hexl_rs APIs for scalar multplication and fused multiplication and addition.
///
/// In concrete costs, functions call `mul_poly_scalar_slice_hexl` 2 times and fma_poly_scale_slice_hexl 127 * 2 times. In general
/// costs of `fma_poly_scale_slice_hexl` and `mul_poly_scalar_slice_hexl` can be assumed to be same.
#[cfg(target_arch = "x86_64")]
pub fn optimised_range_fn_fma_hexl(
    poly_ctx: &PolyContext<'_>,
    single_powers: &[Ciphertext],
    constants: &Array2<u64>,
    constants_outer_offset: usize,
    level: usize,
) -> Ciphertext {
    // process the first index using scalar mod instead of fma, since sum_ct is uinitialised
    let mut sum_ct = {
        let scalar_slice = constants.slice(s![constants_outer_offset + 1, ..]);

        let mut p0 = Poly::new(
            unsafe {
                Array2::<u64>::uninit((poly_ctx.moduli_count(), poly_ctx.degree())).assume_init()
            },
            Representation::Evaluation,
        );

        mul_poly_scalar_slice_hexl(
            poly_ctx,
            &mut p0,
            &single_powers[0].c_ref()[0],
            scalar_slice.as_slice().unwrap(),
        );

        let mut p1 = Poly::new(
            unsafe {
                Array2::<u64>::uninit((poly_ctx.moduli_count(), poly_ctx.degree())).assume_init()
            },
            Representation::Evaluation,
        );

        mul_poly_scalar_slice_hexl(
            poly_ctx,
            &mut p1,
            &single_powers[0].c_ref()[1],
            scalar_slice.as_slice().unwrap(),
        );

        Ciphertext::new(vec![p0, p1], PolyType::Q, level)
    };

    for j in (4..257).step_by(2) {
        let power_ct = &single_powers[j / 2 - 1];
        let scalar_reduced = constants.slice(s![constants_outer_offset + (j - 1), ..]);

        fma_poly_scale_slice_hexl(
            poly_ctx,
            &mut sum_ct.c_ref_mut()[0],
            &power_ct.c_ref()[0],
            scalar_reduced.as_slice().unwrap(),
        );
        fma_poly_scale_slice_hexl(
            poly_ctx,
            &mut sum_ct.c_ref_mut()[1],
            &power_ct.c_ref()[1],
            scalar_reduced.as_slice().unwrap(),
        );
    }

    sum_ct
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{generate_bfv_parameters, precompute_range_constants, read_range_coeffs};
    use bfv::{Encoding, PolyCache};
    use rand::thread_rng;

    #[test]
    fn test_optimised_range_fn_fma() {
        let mut rng = thread_rng();
        // let params = BfvParameters::new(&vec![59; 15], 65537, 1 << 15);
        let params = generate_bfv_parameters();
        let m = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);
        let sk = SecretKey::random_with_params(&params, &mut rng);

        let evaluator = Evaluator::new(params);
        let pt = evaluator.plaintext_encode(&m, Encoding::simd(0, PolyCache::Mul(PolyType::Q)));
        let mut ct = evaluator.encrypt(&sk, &pt, &mut rng);
        // change ct representation to Evaluation for plaintext mul
        evaluator.ciphertext_change_representation(&mut ct, Representation::Evaluation);

        let ctx = evaluator.params().poly_ctx(&PolyType::Q, 0);
        let level = 0;
        let single_powers = vec![ct.clone(); 128];
        let constants = precompute_range_constants(&ctx);

        {
            // warmup
            let mut tmp = evaluator.mul_poly(&ct, pt.mul_poly_ref());
            for j in 0..300 {
                evaluator.add_assign(&mut tmp, &evaluator.mul_poly(&ct, pt.mul_poly_ref()));
            }
        }

        // optimised hexl version
        let now = std::time::Instant::now();
        #[cfg(target_arch = "x86_64")]
        let res_opt_hexl = optimised_range_fn_fma_hexl(&ctx, &single_powers, &constants, 0, level);
        let time_opt_hexl = now.elapsed();

        // optimised version
        let now = std::time::Instant::now();
        let res_opt_u128 = optimised_range_fn_fma_u128(
            &ctx,
            evaluator.params(),
            &single_powers,
            &constants,
            0 * 256,
            level,
        );
        let time_opt = now.elapsed();

        // unoptimised fma
        let range_coeffs = read_range_coeffs();
        // prepare range coefficients plaintext
        let pts = (0..256)
            .map(|i| {
                let c = range_coeffs[i];
                let m = vec![c; ctx.degree()];
                evaluator.plaintext_encode(&m, Encoding::simd(0, PolyCache::Mul(PolyType::Q)))
            })
            .collect_vec();
        let now = std::time::Instant::now();
        let mut res_unopt = evaluator.mul_poly(&ct, pts[2].mul_poly_ref());
        for j in (4..257).step_by(2) {
            evaluator.add_assign(
                &mut res_unopt,
                &evaluator.mul_poly(&ct, pts[j - 1].mul_poly_ref()),
            );
        }
        let time_unopt = now.elapsed();

        println!(
            "Time: Opt={:?}, OptHexl={:?}, UnOpt={:?}",
            time_opt, time_opt_hexl, time_unopt
        );

        #[cfg(target_arch = "x86_64")]
        println!(
            "Noise: Opt={:?}, OptHexl={:?}, UnOpt={:?}",
            evaluator.measure_noise(&sk, &res_opt_u128),
            evaluator.measure_noise(&sk, &res_opt_hexl),
            evaluator.measure_noise(&sk, &res_unopt),
        );

        #[cfg(not(target_arch = "x86_64"))]
        println!(
            "Noise: Opt={:?}, UnOpt={:?}",
            evaluator.measure_noise(&sk, &res_opt_u128),
            evaluator.measure_noise(&sk, &res_unopt),
        );
    }
}
