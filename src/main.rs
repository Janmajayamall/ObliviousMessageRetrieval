use bfv::{
    BfvParameters, Ciphertext, Encoding, EvaluationKey, Evaluator, GaloisKey, Modulus, Plaintext,
    PolyType, RelinearizationKey, Representation, SecretKey,
};
use itertools::{izip, Itertools};
use ndarray::Array2;
use rand::thread_rng;
use rayon::slice::ParallelSliceMut;
use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
    hash::Hash,
    sync::Arc,
};

use omr::{
    client::{encrypt_pvw_sk, gen_pv_exapnd_rtgs},
    optimised::{coefficient_u128_to_ciphertext, sub_from_one_precompute},
    preprocessing::{assign_buckets_and_weights, pre_process_batch},
    pvw::*,
    server::{
        mul_and_reduce_ranged_cts_to_1,
        phase2::{self, phase2_precomputes},
        powers_x::evaluate_powers,
        pvw_decrypt::{pvw_decrypt, pvw_decrypt_precomputed, pvw_setup},
        range_fn::{range_fn, range_fn_4_times},
    },
    time_it,
    utils::{generate_random_payloads, precompute_range_constants},
    GAMMA, K,
};

fn phase1() {
    let mut rng = thread_rng();
    let pvw_params = Arc::new(PvwParameters::default());
    let pvw_sk = PvwSecretKey::random(&pvw_params, &mut rng);
    let pvw_pk = pvw_sk.public_key(&mut rng);

    let params = BfvParameters::default(15, 1 << 8);
    let sk = SecretKey::random(params.degree, &mut rng);

    // generate hints
    println!("Generating clues...");
    let clue1 = pvw_pk.encrypt(&[0, 0, 0, 0], &mut rng);
    let clues = (0..params.degree)
        .into_iter()
        .map(|_| clue1.clone())
        .collect_vec();

    let evaluator = Evaluator::new(params);

    println!("Preprocessing batch...");
    let (hint_a, hint_b) = pre_process_batch(&pvw_params, &evaluator, &clues);

    println!("Encrypting pvw sk...");
    let pvw_sk_cts = encrypt_pvw_sk(&evaluator, &sk, &pvw_sk, &mut rng);

    println!("Generating keys");
    let ek = EvaluationKey::new(evaluator.params(), &sk, &[0], &[0], &[1], &mut rng);

    #[cfg(feature = "precomp_pvw")]
    // pre compute pvw_sk_cts rotations for maximum core usage
    let precomputed_pvw_sk_cts = pvw_setup(&evaluator, &ek, &pvw_sk_cts);

    println!("Running Pvw decrypt...");

    #[cfg(feature = "precomp_pvw")]
    let decrypted_cts = {
        pvw_decrypt_precomputed(
            &pvw_params,
            &evaluator,
            &hint_a,
            &hint_b,
            &precomputed_pvw_sk_cts,
            ek.get_rtg_ref(1, 0),
            &sk,
        )
    };

    #[cfg(not(feature = "precomp_pvw"))]
    let decrypted_cts = pvw_decrypt(
        &pvw_params,
        &evaluator,
        &hint_a,
        &hint_b,
        &pvw_sk_cts,
        ek.get_rtg_ref(1, 0),
        &sk,
    );

    decrypted_cts.iter().for_each(|ct| {
        println!("Decrypted Ct noise: {}", evaluator.measure_noise(&sk, ct));
    });

    let constants = precompute_range_constants(&evaluator.params().poly_ctx(&PolyType::Q, 0));
    let sub_from_one_precompute = sub_from_one_precompute(evaluator.params(), 0);

    println!("Running range function...");
    // ranged cts are in coefficient form
    let ranged_cts = range_fn_4_times(
        &decrypted_cts,
        &evaluator,
        &ek,
        &constants,
        &sub_from_one_precompute,
        &sk,
    );

    let mut v = mul_and_reduce_ranged_cts_to_1(&ranged_cts, &evaluator, &ek, &sk);

    println!("phase 1 end ct noise: {}", evaluator.measure_noise(&sk, &v));
    phase2(&evaluator, &sk, &mut v);
}

fn phase2(evaluator: &Evaluator, sk: &SecretKey, pv: &mut Ciphertext) {
    let mut rng = thread_rng();

    // generate evaluation key for phase 2
    let ek = gen_pv_exapnd_rtgs(evaluator.params(), sk, 12);

    // mod down to level 12 irrespective of whether level feature is enabled. Otherwise, phase2 will be very expensive.
    evaluator.mod_down_level(pv, 12);

    // switch to Evaluation representation for efficient rotations and scalar multiplications
    evaluator.ciphertext_change_representation(pv, Representation::Evaluation);

    let degree = evaluator.params().degree;
    let level = 12;

    // phase 2 precomputes
    let (pts_32_batch, pts_4_roll, pts_1_roll) = phase2_precomputes(evaluator, degree, level);
    let (_, buckets, weights) = assign_buckets_and_weights(
        K * 2,
        GAMMA,
        evaluator.params().plaintext_modulus,
        evaluator.params().degree,
        &mut rng,
    );
    let payloads = generate_random_payloads(evaluator.params().degree);

    let (indices_ct, weights_ct) = phase2::phase2(
        evaluator,
        pv,
        &ek,
        level,
        &pts_32_batch,
        &pts_4_roll,
        &pts_1_roll,
        &payloads,
        &buckets,
        &weights,
        sk,
    );

    println!(
        "Indices Ct noise: {}",
        evaluator.measure_noise(sk, &indices_ct)
    );
    println!(
        "Weight Ct noise: {}",
        evaluator.measure_noise(sk, &weights_ct)
    );
}

fn powers_of_x() {
    let mut rng = thread_rng();
    let params = BfvParameters::default(15, 1 << 15);
    let sk = SecretKey::random(params.degree, &mut rng);
    let ek = EvaluationKey::new(&params, &sk, &[0], &[], &[], &mut rng);

    let m = params
        .plaintext_modulus_op
        .random_vec(params.degree, &mut rng);
    let evaluator = Evaluator::new(params);
    let pt = evaluator.plaintext_encode(&m, Encoding::default());
    let ct = evaluator.encrypt(&sk, &pt, &mut rng);

    let cores = 1;

    time_it!("Powers of x [1,256) time: ",
        let placeholder = Ciphertext::placeholder();
        let mut calculated = vec![placeholder.clone(); 255];
        calculated[0] = ct;
        evaluate_powers(&evaluator, &ek, 2, 4, &mut calculated, false, cores);
        evaluate_powers(&evaluator, &ek, 4, 8, &mut calculated, false, cores);
        evaluate_powers(&evaluator, &ek, 8, 16, &mut calculated, false, cores);
        evaluate_powers(&evaluator, &ek, 16, 32, &mut calculated, false, cores);
        evaluate_powers(&evaluator, &ek, 32, 64, &mut calculated, false, cores);
        evaluate_powers(&evaluator, &ek, 64, 128, &mut calculated, false, cores);
        evaluate_powers(&evaluator, &ek, 128, 256, &mut calculated, false, cores);
    );
}

fn call_pvw_decrypt() {
    let mut rng = thread_rng();
    let pvw_params = Arc::new(PvwParameters::default());
    let pvw_sk = PvwSecretKey::random(&pvw_params, &mut rng);
    let pvw_pk = pvw_sk.public_key(&mut rng);

    let params = BfvParameters::default(15, 1 << 15);
    let sk = SecretKey::random(params.degree, &mut rng);

    let clue1 = pvw_pk.encrypt(&[0, 0, 0, 0], &mut rng);
    let clues = (0..params.degree)
        .into_iter()
        .map(|_| clue1.clone())
        .collect_vec();

    let evaluator = Evaluator::new(params);

    let (hint_a, hint_b) = pre_process_batch(&pvw_params, &evaluator, &clues);

    let pvw_sk_cts = encrypt_pvw_sk(&evaluator, &sk, &pvw_sk, &mut rng);

    let ek = EvaluationKey::new(evaluator.params(), &sk, &[0], &[0], &[1], &mut rng);

    time_it!("Pvw Decrypt",
        let _ = pvw_decrypt(
                &pvw_params,
                &evaluator,
                &hint_a,
                &hint_b,
                &pvw_sk_cts,
                ek.get_rtg_ref(1, 0),
                &sk,
            );
    );
}

fn call_pvw_decrypt_precomputed() {
    let mut rng = thread_rng();
    let pvw_params = Arc::new(PvwParameters::default());
    let pvw_sk = PvwSecretKey::random(&pvw_params, &mut rng);
    let pvw_pk = pvw_sk.public_key(&mut rng);

    let params = BfvParameters::default(15, 1 << 15);
    let sk = SecretKey::random(params.degree, &mut rng);

    let clue1 = pvw_pk.encrypt(&[0, 0, 0, 0], &mut rng);
    let clues = (0..params.degree)
        .into_iter()
        .map(|_| clue1.clone())
        .collect_vec();

    let evaluator = Evaluator::new(params);

    let (hint_a, hint_b) = pre_process_batch(&pvw_params, &evaluator, &clues);

    let pvw_sk_cts = encrypt_pvw_sk(&evaluator, &sk, &pvw_sk, &mut rng);

    let ek = EvaluationKey::new(evaluator.params(), &sk, &[0], &[0], &[1], &mut rng);

    println!("Running precomputation...");
    let precomputed_pvw_sk_cts = pvw_setup(&evaluator, &ek, &pvw_sk_cts);

    println!("Starting decrypt...");
    time_it!("Pvw Decrypt Precomputed",
        let _ = pvw_decrypt_precomputed(
                &pvw_params,
                &evaluator,
                &hint_a,
                &hint_b,
                &precomputed_pvw_sk_cts,
                ek.get_rtg_ref(1, 0),
                &sk,
            );
    );
}

/// Wrapper for setting up necessary things before calling range_fn
fn call_range_fn_once() {
    let params = BfvParameters::default(15, 1 << 15);
    let level = 0;
    let ctx = params.poly_ctx(&PolyType::Q, level);

    let mut rng = thread_rng();
    let constants = precompute_range_constants(&ctx);
    let sub_one_precompute = sub_from_one_precompute(&params, level);

    let sk = SecretKey::random(params.degree, &mut rng);
    let mut m = params
        .plaintext_modulus_op
        .random_vec(params.degree, &mut rng);

    let evaluator = Evaluator::new(params);
    let pt = evaluator.plaintext_encode(&m, Encoding::simd(level));
    let mut ct = evaluator.encrypt(&sk, &pt, &mut rng);

    // gen evaluation key
    let ek = EvaluationKey::new(evaluator.params(), &sk, &[0], &[], &[], &mut rng);

    time_it!("Range fn time: ",
        let _ = range_fn(&ct, &evaluator, &ek, &constants, &sub_one_precompute, &sk);
    );
}

fn main() {
    let threads = 8;
    // set global thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    // call_pvw_decrypt();
    // call_pvw_decrypt_precomputed();

    // powers_of_x();
    // call_range_fn_once();

    // range_fn_trial();
}
