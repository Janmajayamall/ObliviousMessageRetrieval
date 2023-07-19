use bfv::{Encoding, Evaluator, Modulus, PolyType, Representation, SecretKey};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;
use ndarray::Array2;
use omr::{
    server::range_fn::range_fn_fma::{
        mul_poly_scalar_u128, optimised_range_fn_fma_u128, scalar_mul_u128,
    },
    utils::{generate_bfv_parameters, precompute_range_constants},
};
use rand::thread_rng;

#[cfg(target_arch = "x86_64")]
use omr::server::range_fn::range_fn_fma::{
    fma_poly_scale_slice_hexl, mul_poly_scalar_slice_hexl, optimised_range_fn_fma_hexl,
};

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_fn");
    // group.measurement_time(Duration::new(100, 0));
    // group.sample_size(10);

    let mut rng = thread_rng();

    for degree in [1 << 12, 1 << 15] {
        // mod_size should give a baseline of what performance values look like per modulus iteration. More so, the time take by an operation should match its corresponding operation in hexl_rs.
        // For ex, for a given `degree` time taken by `fma_poly_scale_slice_hexl` when mod_size is 1 should equal time take by `hexl_rs::elwise_fma_mod` for vector size `degree`.
        // So far when mod_size is 1 values match their corresponidng values from hexl_rs benches. However, as mod_size increases performance time increases non-linearly.
        // For ex, when degree is 2^15 in `fma_poly_scale_slice_hexl` if it takes `x` us when mod size is 1, we expect it to take `15x` us when mod size is 15. However, this does
        // not happen. I am unsure what's the reason. Is it cache misses? Smaller cache size?
        for mod_size in [1, 3, 7, 15] {
            let params = generate_bfv_parameters();
            let evaluator = Evaluator::new(params);
            let params = evaluator.params();
            let ctx = evaluator.params().poly_ctx(&PolyType::Q, 0);

            let p0 = ctx.random(Representation::Evaluation, &mut rng);
            let p1 = ctx.random(Representation::Evaluation, &mut rng);
            let scalar_slice = (0..ctx.moduli_count()).map(|v| v as u64).collect_vec();

            #[cfg(target_arch = "x86_64")]
            {
                let mut p0_clone = p0.clone();
                group.bench_function(
                    BenchmarkId::new(
                        "mul_poly_scalar_slice_hexl",
                        format!("n={degree}/mod_size={mod_size}"),
                    ),
                    |b| {
                        b.iter(|| {
                            mul_poly_scalar_slice_hexl(&ctx, &mut p0_clone, &p1, &scalar_slice);
                        });
                    },
                );
            }

            #[cfg(target_arch = "x86_64")]
            {
                let mut p0_clone = p0.clone();
                group.bench_function(
                    BenchmarkId::new(
                        "fma_poly_scale_slice_hexl",
                        format!("n={degree}/mod_size={mod_size}"),
                    ),
                    |b| {
                        b.iter(|| {
                            fma_poly_scale_slice_hexl(&ctx, &mut p0_clone, &p1, &scalar_slice);
                        });
                    },
                );
            }

            {
                let m = params
                    .plaintext_modulus_op
                    .random_vec(params.degree, &mut rng);
                let sk = SecretKey::random_with_params(&params, &mut rng);

                let pt = evaluator.plaintext_encode(&m, Encoding::default());
                let ct = evaluator.encrypt(&sk, &pt, &mut rng);
                let ctx = evaluator.params().poly_ctx(&PolyType::Q, 0);
                let level = 0;
                let single_powers = vec![ct.clone(); 128];
                let constants = precompute_range_constants(&ctx);

                #[cfg(target_arch = "x86_64")]
                group.bench_function(
                    BenchmarkId::new(
                        "optimised_range_fn_fma_hexl",
                        format!("n={degree}/mod_size={mod_size}"),
                    ),
                    |b| {
                        b.iter(|| {
                            optimised_range_fn_fma_hexl(&ctx, &single_powers, &constants, 0, level);
                        });
                    },
                );

                group.bench_function(
                    BenchmarkId::new(
                        "optimised_range_fn_fma_u128",
                        format!("n={degree}/mod_size={mod_size}"),
                    ),
                    |b| {
                        b.iter(|| {
                            optimised_range_fn_fma_u128(
                                &ctx,
                                params,
                                &single_powers,
                                &constants,
                                0,
                                level,
                            );
                        });
                    },
                );
            }

            {
                let a = p0.clone();
                let mut res = Array2::<u128>::zeros((ctx.moduli_count(), ctx.degree()));
                group.bench_function(
                    BenchmarkId::new(
                        "mul_poly_scalar_u128",
                        format!("n={degree}/mod_size={mod_size}"),
                    ),
                    |b| {
                        b.iter(|| mul_poly_scalar_u128(&mut res, &a, &scalar_slice));
                    },
                );
            }
        }

        {
            let modulus = Modulus::new(1125899904679937);
            let mut sum = vec![0; degree];
            let a0 = modulus.random_vec(degree, &mut rng);
            group.bench_function(
                BenchmarkId::new("scalar_mul_u128", format!("n={degree}")),
                |b| {
                    b.iter(|| {
                        scalar_mul_u128(&mut sum, &a0, 1125899904679937);
                    });
                },
            );
        }
    }
}

criterion_group!(range_fn, bench);
criterion_main!(range_fn);
