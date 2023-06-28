use crate::optimised::{coefficient_u128_to_ciphertext, fma_reverse_u128_poly};
use crate::preprocessing::{precompute_expand_32_roll_pt, procompute_expand_roll_pt};
use crate::utils::decrypt_and_print;
use crate::{
    optimised::{
        barret_reduce_coefficients_u128, optimised_pvw_fma_with_rot, scalar_mul_u128, sub_from_one,
    },
    pvw::PvwParameters,
};
use bfv::{
    BfvParameters, Ciphertext, EvaluationKey, Evaluator, GaloisKey, Plaintext, Poly, PolyContext,
    PolyType, RelinearizationKey, Representation, SecretKey,
};
use itertools::{izip, Itertools};
use ndarray::{s, Array2};
use std::{collections::HashMap, sync::Arc, time::Instant};

/// ciphertext and a vector of u64
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

    // let mut inner_now = Instant::now();
    // Starting from 0th index every alternate constant is 0. Since plintext multiplication by 0 is 0, we don't need to
    // process plaintext multiplications for indices at which constant is 0. Thus, we start from 1st index and process
    // every alternate index.
    for j in (2..257).step_by(2) {
        let power_ct = &single_powers[j / 2 - 1];
        let scalar_reduced = constants.slice(s![constants_outer_offset + (j - 1), ..]);
        izip!(
            res0.outer_iter_mut(),
            power_ct.c_ref()[0].coefficients.outer_iter(),
            scalar_reduced.iter(),
        )
        .for_each(|(mut r, a, s)| {
            scalar_mul_u128(r.as_slice_mut().unwrap(), a.as_slice().unwrap(), *s);
        });
        izip!(
            res1.outer_iter_mut(),
            power_ct.c_ref()[1].coefficients.outer_iter(),
            scalar_reduced.iter()
        )
        .for_each(|(mut r, a, s)| {
            scalar_mul_u128(r.as_slice_mut().unwrap(), a.as_slice().unwrap(), *s);
        });
    }

    coefficient_u128_to_ciphertext(params, &res0, &res1, level)
}

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

pub fn optimised_range_fn_fma_hexl(
    poly_ctx: &PolyContext<'_>,
    single_powers: &[Ciphertext],
    constants: &Array2<u64>,
    constants_outer_offset: usize,
    level: usize,
) -> Ciphertext {
    // Starting from 0th index every alternate constant is 0. Since plintext multiplication by 0 is 0, we don't need to
    // process plaintext multiplications for indices at which constant is 0. Thus, we start from 1st index and process
    // every alternate index.

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
                evaluator.plaintext_encode(&m, Encoding::simd(ct.level))
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
        println!(
            "Noise: Opt={:?}, OptHexl={:?}, UnOpt={:?}",
            evaluator.measure_noise(&sk, &res_opt_u128),
            evaluator.measure_noise(&sk, &res_opt_hexl),
            evaluator.measure_noise(&sk, &res_unopt),
        );
    }
}
