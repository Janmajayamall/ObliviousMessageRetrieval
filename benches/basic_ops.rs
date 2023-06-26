use bfv::{
    nb_theory::generate_primes_vec,
    parameters::BfvParameters,
    poly::{Poly, PolyContext, Representation},
    Encoding, Plaintext, SecretKey,
};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use itertools::{izip, Itertools};
use ndarray::Array2;
use omr::optimised::{coefficient_u128_to_ciphertext, optimised_poly_fma};
use rand::thread_rng;
use std::{result, sync::Arc, time::Duration};

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic-ops");
    group.measurement_time(Duration::new(100, 0));
    group.sample_size(10);

    let mut rng = thread_rng();
    let params = Arc::new(BfvParameters::default(12, 1 << 15));
    let m = params
        .plaintext_modulus_op
        .random_vec(params.polynomial_degree, &mut rng);
    let pt = Plaintext::encode(&m, &params, Encoding::simd(0));
    let sk = SecretKey::random(&params, &mut rng);
    let cts = (0..256)
        .map(|_| {
            let mut c = sk.encrypt(&pt, &mut rng);
            c.change_representation(&Representation::Evaluation);
            c
        })
        .collect_vec();

    // optimised
    let polys = vec![pt.poly_ntt_ref().clone(); 256];
    group.bench_function("optimised_fma", |b| {
        b.iter(|| {
            let mut res00 =
                Array2::<u128>::zeros((params.ciphertext_moduli.len(), params.polynomial_degree));
            let mut res01 =
                Array2::<u128>::zeros((params.ciphertext_moduli.len(), params.polynomial_degree));
            optimised_poly_fma(&cts, &polys, &mut res00, &mut res01);
            let _ = coefficient_u128_to_ciphertext(&params, &res00, &res01, 0);
        });
    });

    // unoptimised fma
    let pts = vec![pt; 256];
    group.bench_function("unoptimised_fma", |b| {
        b.iter(|| {
            let mut res = &cts[0] * &pts[0];
            izip!(cts.iter(), pts.iter()).skip(1).for_each(|(c, p)| {
                res.fma_reverse_inplace(c, p);
            });
        });
    });
}

criterion_group!(basic_ops, bench);
criterion_main!(basic_ops);
