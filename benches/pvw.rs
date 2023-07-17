use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use omr::pvw::{PvwCiphertext, PvwParameters, PvwPublicKey, PvwSecretKey};
use rand::{thread_rng, Rng};

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("pvw");

    let mut rng = thread_rng();
    let params = PvwParameters::default();
    let sk = PvwSecretKey::random(&params, &mut rng);
    let m = vec![0, 0, 0, 0];

    group.bench_function("gen_public_key", |b| {
        b.iter(|| {
            let _ = sk.public_key(&mut rng);
        });
    });

    let pk = sk.public_key(&mut rng);
    group.bench_function("encrypt", |b| {
        b.iter(|| {
            let _ = pk.encrypt(&m, &mut rng);
        });
    });
}

criterion_group!(pvw, bench);
criterion_main!(pvw);
