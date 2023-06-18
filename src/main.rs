use bfv::{
    BfvParameters, Ciphertext, Encoding, GaloisKey, Modulus, Plaintext, RelinearizationKey,
    Representation, SecretKey,
};
use itertools::{izip, Itertools};
use rand::thread_rng;
use std::sync::Arc;

use omr::{
    optimised::sub_from_one_precompute,
    plaintext::{powers_of_x_int, powers_of_x_modulus},
    pvw::*,
    server::{
        encrypt_pvw_sk, even_powers_of_x_ct, powers_of_x_ct, pre_process_batch, pvw_decrypt,
        range_fn,
    },
    utils::precompute_range_constants,
};

fn level_powers_of_x() {
    let mut rng = thread_rng();
    let params = Arc::new(BfvParameters::default(14, 1 << 15));
    let sk = SecretKey::random(&params, &mut rng);
    let m = vec![3; params.polynomial_degree];
    let pt = Plaintext::encode(&m, &params, Encoding::simd(0));
    let ct = sk.encrypt(&pt, &mut rng);
    let rlk = RelinearizationKey::new(&params, &sk, 0, &mut rng);

    powers_of_x_ct(&ct, &rlk, &sk);
}
fn level_range_fn() {
    let params = Arc::new(BfvParameters::default(10, 1 << 14));
    let ctx = params.ciphertext_ctx_at_level(0);

    let mut rng = thread_rng();
    let constants = precompute_range_constants(&ctx);
    let sub_one_precompute = sub_from_one_precompute(&params, 0);
    let sk = SecretKey::random(&params, &mut rng);
    let mut m = params
        .plaintext_modulus_op
        .random_vec(params.polynomial_degree, &mut rng);
    let pt = Plaintext::encode(&m, &params, Encoding::simd(0));
    let mut ct = sk.encrypt(&pt, &mut rng);

    // gen rlk
    let rlk = RelinearizationKey::new(&params, &sk, 0, &mut rng);

    let ct_res = range_fn(&ct, &rlk, &constants, &sub_one_precompute, &sk);
    dbg!(sk.measure_noise(&ct_res, &mut rng));
}

/// 1. Pvw decrypt
/// 2. range fn
/// 3. muls into 1
fn phase1() {
    let mut rng = thread_rng();
    let pvw_params = Arc::new(PvwParameters::default());
    let pvw_sk = PvwSecretKey::random(&pvw_params, &mut rng);
    let pvw_pk = pvw_sk.public_key(&mut rng);

    let params = Arc::new(BfvParameters::default(15, 1 << 15));
    let sk = SecretKey::random(&params, &mut rng);

    // generate hints
    let clue1 = pvw_pk.encrypt(&[0, 0, 0, 0], &mut rng);
    let clues = (0..params.polynomial_degree)
        .into_iter()
        .map(|_| clue1.clone())
        .collect_vec();

    println!("Preprocessing batch...");
    let (hint_a, hint_b) = pre_process_batch(&pvw_params, &params, &clues);

    println!("Encrypting pvw sk...");
    let pvw_sk_cts = encrypt_pvw_sk(&params, &sk, &pvw_sk, &mut rng);

    println!("Generating keys");
    let rlk = RelinearizationKey::new(&params, &sk, 0, &mut rng);
    let rtk = GaloisKey::new(3, &params.ciphertext_ctx_at_level(0), &sk, &mut rng);

    println!("Running Pvw decrypt...");
    let mut now = std::time::Instant::now();
    let decryted_cts = pvw_decrypt(&pvw_params, &hint_a, &hint_b, &pvw_sk_cts, &rtk, &sk);
    println!("Pvw decryption time: {:?}", now.elapsed());

    decryted_cts.iter().for_each(|ct| {
        println!("Decrypted Ct noise: {}", sk.measure_noise(ct, &mut rng));
    });

    let constants = precompute_range_constants(&params.ciphertext_ctx_at_level(0));
    let sub_from_one_precompute = sub_from_one_precompute(&params, 0);

    println!("Running range function...");
    // ranged cts are in coefficient form
    now = std::time::Instant::now();
    let ranged_cts = decryted_cts
        .iter()
        .map(|ct| range_fn(&ct, &rlk, &constants, &sub_from_one_precompute, &sk))
        .collect_vec();
    println!("Range function time: {:?}", now.elapsed());

    ranged_cts.iter().for_each(|ct| {
        assert!(ct.c_ref()[0].representation == Representation::Coefficient);
        println!("Ranged Ct noise: {}", sk.measure_noise(ct, &mut rng));
    });

    now = std::time::Instant::now();
    // multiplication tree
    // ct[0] * ct[1]          ct[2] * ct[4]
    //      v0         *          v1
    //                v
    let v0 = ranged_cts[0].multiply1(&ranged_cts[1]);
    let v0 = rlk.relinearize(&v0);
    let v1 = ranged_cts[2].multiply1(&ranged_cts[3]);
    let v1 = rlk.relinearize(&v1);
    let v = v0.multiply1(&v1);
    // Relinearization of `v` can be modified such that overall ntts can be minized.
    // We expect `v` to be in evaluation form. Thus we convert c0 and c1, not c2, to evaluation form
    // after scale_and_round op. key_switch `c2` (c2 stays in coeffciient form). The c0' and
    // c1' (key switch outputs in evaluation form) to c0 and c1 respectively. Instead if we
    // use normal `relinearize` the output will be in coefficient form and will have to pay
    // for additional 2 Ntts of size Q to convert ouput to evaluation.
    let v = rlk.relinearize(&v);
    println!("Multiplication time: {:?}", now.elapsed());

    println!("phase 1 end ct noise: {}", sk.measure_noise(&v, &mut rng));
}

/// Time of `even_powers_of_x_ct` should be half of `powers_of_x_ct`
fn time_dff_even_all_powers_of_x() {
    let mut rng = thread_rng();
    let params = Arc::new(BfvParameters::default(15, 1 << 15));
    let sk = SecretKey::random(&params, &mut rng);
    let m = vec![3; params.polynomial_degree];
    let pt = Plaintext::encode(&m, &params, Encoding::simd(0));
    let ct = sk.encrypt(&pt, &mut rng);
    let rlk = RelinearizationKey::new(&params, &sk, 0, &mut rng);

    {
        for _ in 0..2 {
            powers_of_x_ct(&ct, &rlk, &sk);
        }
    }

    let now = std::time::Instant::now();
    let even_powers_ct = even_powers_of_x_ct(&ct, &rlk, &sk);
    println!("Time even_powers_of_x_ct = {:?}", now.elapsed());

    let now = std::time::Instant::now();
    let powers_ct = powers_of_x_ct(&ct, &rlk, &sk);
    println!("Time powers_of_x_ct = {:?}", now.elapsed());
}
fn main() {
    // level_powers_of_x();
    // level_range_fn();
    phase1()
    // time_dff_even_all_powers_of_x();
}
