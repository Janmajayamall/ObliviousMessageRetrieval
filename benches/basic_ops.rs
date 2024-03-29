use bfv::{Encoding, Evaluator, Modulus, PolyCache, PolyType, Representation, SecretKey};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::{izip, Itertools};
use ndarray::Array2;
use omr::{
    optimised::{coefficient_u128_to_ciphertext, fma_reverse_u128_vec, optimised_poly_fma},
    utils::generate_bfv_parameters,
};
use rand::thread_rng;

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic-ops");
    // group.measurement_time(Duration::new(100, 0));
    // group.sample_size(10);

    let mut rng = thread_rng();
    let params = generate_bfv_parameters();
    let m = params
        .plaintext_modulus_op
        .random_vec(params.degree, &mut rng);
    let sk = SecretKey::random_with_params(&params, &mut rng);

    let evaluator = Evaluator::new(params);
    let pt = evaluator.plaintext_encode(&m, Encoding::simd(0, PolyCache::Mul(PolyType::Q)));
    let mut ct0 = evaluator.encrypt(&sk, &pt, &mut rng);
    evaluator.ciphertext_change_representation(&mut ct0, Representation::Evaluation);

    let cts = (0..256)
        .map(|_| {
            let mut c = evaluator.encrypt(&sk, &pt, &mut rng);
            evaluator.ciphertext_change_representation(&mut c, Representation::Evaluation);
            c
        })
        .collect_vec();

    // optimised
    let polys = vec![pt.mul_poly_ref().clone(); 256];
    group.bench_function("optimised_fma", |b| {
        b.iter(|| {
            let ctx = evaluator.params().poly_ctx(&PolyType::Q, 0);
            let mut res00 = Array2::<u128>::zeros((ctx.moduli_count(), ctx.degree()));
            let mut res01 = Array2::<u128>::zeros((ctx.moduli_count(), ctx.degree()));
            optimised_poly_fma(&cts, &polys, &mut res00, &mut res01);
            let _ = coefficient_u128_to_ciphertext(evaluator.params(), &res00, &res01, 0);
        });
    });

    // unoptimised fma
    group.bench_function("unoptimised_fma", |b| {
        b.iter(|| {
            let mut res = evaluator.mul_poly(&cts[0], &polys[0]);
            izip!(cts.iter(), polys.iter())
                .skip(1)
                .for_each(|(c, p)| evaluator.fma_poly(&mut res, c, p));
        });
    });

    for degree in [1 << 15] {
        let modulus = Modulus::new(1125899904679937);
        let mut sum = vec![0; degree];
        let a0 = modulus.random_vec(degree, &mut rng);
        let a1 = modulus.random_vec(degree, &mut rng);

        group.bench_function(
            BenchmarkId::new("fma_reverse_u128_vec", format!("n={degree}")),
            |b| {
                b.iter(|| {
                    fma_reverse_u128_vec(&mut sum, &a0, &a1);
                });
            },
        );
    }
}

criterion_group!(basic_ops, bench);
criterion_main!(basic_ops);
