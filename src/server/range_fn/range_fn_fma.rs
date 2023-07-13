use crate::optimised::{coefficient_u128_to_ciphertext, fma_reverse_u128_poly};
use crate::preprocessing::{precompute_expand_32_roll_pt, procompute_expand_roll_pt};
use crate::utils::decrypt_and_print;
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
use std::{collections::HashMap, sync::Arc, time::Instant};

/// r0 += a0 * s
pub fn scalar_mul_u128(r: &mut [u128], a: &[u64], s: u64) {
    let s_u128 = s as u128;
    r.iter_mut().zip(a.iter()).for_each(|(r0, a0)| {
        *r0 += *a0 as u128 * s_u128;
    })
}

/// `mul_poly_scalar_u128` calls `scalar_mul_u128` moduli count times. Hence,
/// cost of `mul_poly_scalar_u128` must be `moduli_count * Cost(scalar_mul_u128)`
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

/// `optimised_range_fn_fma_u128` calls `mul_poly_scalar_u128` 128 * 2 (ie 256) times. Additionally,
/// inside `coefficient_u128_to_ciphertext` it calls `barret_reduction_u128_vec` `moduli_count*2` times.
///
/// Hence the cost of `optimised_range_fn_fma_u128` must be
/// moduli_count * Cost(scalar_mul_u128) * 256 + moduli_count * 2 * Cost(barret_reduction_u128_vec)
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
/// `mul_poly_scalar_slice_hexl` calls `hexl_rs::elwise_mult_scalar_mod_2` moduli_count times. Hence, its cost must
/// be:
/// `moduli_count` * Cost(hexl_rs::elwise_mult_scalar_mod_2)
pub fn mul_poly_scalar_slice_hexl(
    poly_ctx: &PolyContext<'_>,
    res: &mut Poly,
    a: &Poly,
    scalar_slice: &[u64],
) {
    izip!(
        res.coefficients.outer_iter_mut(),
        a.coefficients.outer_iter(),
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

/// `fma_poly_scale_slice_hexl` calls `hexl_rs::elwise_fma_mod` moduli_count times. Hence, its cost must
/// be:
/// `moduli_count` * Cost(hexl_rs::elwise_fma_mod)
#[cfg(target_arch = "x86_64")]
pub fn fma_poly_scale_slice_hexl(
    poly_ctx: &PolyContext<'_>,
    res: &mut Poly,
    a: &Poly,
    scalar_slice: &[u64],
) {
    izip!(
        res.coefficients.outer_iter_mut(),
        a.coefficients.outer_iter(),
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

/// `optimised_range_fn_fma_hexl` calls `mul_poly_scalar_slice_hexl` 2 times and `fma_poly_scale_slice_hexl` 127 times.
/// Hence, its cost must be
/// mul_poly_scalar_slice_hexl * 2 + fma_poly_scale_slice_hexl * 2 * 127.
///
/// If we assume that cost of `hexl_rs::elwise_mult_scalar_mod_2` and `hexl_rs::elwise_fma_mod` as more or less equal, then the cost
/// of `optimised_range_fn_fma_hexl` can be estimated as
/// let hexl_cost = max( `hexl_rs::elwise_mult_scalar_mod_2` , `hexl_rs::elwise_fma_mod`)
/// Cost(optimised_range_fn_fma_hexl) = hexl_cost * moduli_count * 2 + hexl_cost * moduli_count * 2 * 127
/// = (hexl_cost * moduli_count * 2) * 128 = hexl_cost * moduli_count * 256
#[cfg(target_arch = "x86_64")]
pub fn optimised_range_fn_fma_hexl(
    poly_ctx: &PolyContext<'_>,
    single_powers: &[Ciphertext],
    constants: &Array2<u64>,
    constants_outer_offset: usize,
    level: usize,
) -> Ciphertext {
    // process the first index using scalar mod instead of fma
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
    use crate::utils::{precompute_range_constants, read_range_coeffs};
    use bfv::Encoding;
    use rand::thread_rng;

    #[test]
    fn test_optimised_range_fn_fma() {
        let mut rng = thread_rng();
        // let params = BfvParameters::new(&vec![59; 15], 65537, 1 << 15);
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
        let level = 0;
        let single_powers = vec![ct.clone(); 128];
        let constants = precompute_range_constants(&ctx);

        {
            // warmup
            let mut tmp = evaluator.mul_poly(&ct, pt.poly_ntt_ref());
            for j in 0..300 {
                evaluator.add_assign(&mut tmp, &evaluator.mul_poly(&ct, pt.poly_ntt_ref()));
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
                evaluator.plaintext_encode(&m, Encoding::simd(ct.level()))
            })
            .collect_vec();
        let now = std::time::Instant::now();
        let mut res_unopt = evaluator.mul_poly(&ct, pts[2].poly_ntt_ref());
        for j in (4..257).step_by(2) {
            evaluator.add_assign(
                &mut res_unopt,
                &evaluator.mul_poly(&ct, pts[j - 1].poly_ntt_ref()),
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
