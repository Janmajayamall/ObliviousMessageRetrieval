use bfv::{
    BfvParameters, Ciphertext, CiphertextProto, Encoding, EvaluationKey, EvaluationKeyProto,
    Evaluator, GaloisKey, Modulus, Plaintext, Poly, PolyType, RelinearizationKey, Representation,
    SecretKey, TryFromWithParameters,
};
use itertools::{izip, Itertools};
use ndarray::Array2;
use prost::Message;
use rand::{thread_rng, Rng};
use rayon::slice::ParallelSliceMut;
use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
    f32::consts::E,
    hash::Hash,
    io::Write,
    os::unix::thread,
    sync::Arc,
};

use omr::{
    client::{
        construct_lhs, construct_rhs, encrypt_pvw_sk, evaluation_key, gen_pv_exapnd_rtgs,
        pv_decompress, solve_equations,
    },
    optimised::{coefficient_u128_to_ciphertext, sub_from_one_precompute},
    plaintext,
    preprocessing::{assign_buckets_and_weights, pre_process_batch},
    print_noise,
    pvw::{self, *},
    server::{
        mul_and_reduce_ranged_cts_to_1,
        phase2::{self, phase2_precomputes},
        powers_x::evaluate_powers,
        pvw_decrypt::{pvw_decrypt, pvw_decrypt_precomputed, pvw_setup},
        range_fn::{range_fn, range_fn_4_times},
    },
    time_it,
    utils::{
        generate_bfv_parameters, generate_random_payloads, precompute_range_constants,
        prepare_clues_for_demo,
    },
    BUCKET_SIZE, GAMMA, K,
};

fn print_detection_key_size() {
    let mut rng = thread_rng();

    let params = generate_bfv_parameters();

    // Client's PVW keys
    let pvw_params = PvwParameters::default();
    let pvw_sk = PvwSecretKey::random(&pvw_params, &mut rng);

    // Client's BFV keys
    let sk = SecretKey::random(params.degree, &mut rng);

    let evaluator = Evaluator::new(params);

    // Client's detection keys
    let pvw_sk_cts = encrypt_pvw_sk(&evaluator, &sk, &pvw_sk, &mut rng);
    let ek = evaluation_key(evaluator.params(), &sk);

    let mut bytes = 0;
    pvw_sk_cts.iter().for_each(|c| {
        let proto = CiphertextProto::try_from_with_parameters(c, evaluator.params());
        bytes += proto.encode_to_vec().len();
    });

    bytes += EvaluationKeyProto::try_from_with_parameters(&ek, evaluator.params())
        .encode_to_vec()
        .len();

    println!("Detection Key Size: {bytes} bytes");
}

fn demo() {
    let mut rng = thread_rng();

    let params = generate_bfv_parameters();

    // Client's PVW keys
    let pvw_params = PvwParameters::default();
    let pvw_sk = PvwSecretKey::random(&pvw_params, &mut rng);
    let pvw_pk = pvw_sk.public_key(&mut rng);

    // Client's BFV keys
    let sk = SecretKey::random(params.degree, &mut rng);

    // a random secret key as a placeholder to be passed around in functions. Should be
    // removed in production.
    let random_sk = SecretKey::random(params.degree, &mut rng);

    let evaluator = Evaluator::new(params);

    // Client's detection keys
    let pvw_sk_cts = encrypt_pvw_sk(&evaluator, &sk, &pvw_sk, &mut rng);
    let ek = evaluation_key(evaluator.params(), &sk);

    // generate pertinent indices
    let pertinency_message_count = 64;
    let mut pertinent_indices = vec![];
    while pertinent_indices.len() != pertinency_message_count {
        let index = rng.gen_range(0..evaluator.params().degree);
        if !pertinent_indices.contains(&index) {
            pertinent_indices.push(index);
        }
    }
    pertinent_indices.sort();

    // generate clues and payloads
    // let clues = generate_clues(
    //     &pvw_pk,
    //     &pvw_params,
    //     &pertinent_indices,
    //     evaluator.params().degree,
    // );
    let clues = prepare_clues_for_demo(
        &pvw_params,
        &pvw_pk,
        &pertinent_indices,
        evaluator.params().degree,
    );
    let payloads = generate_random_payloads(evaluator.params().degree);

    // pre-processing
    println!("Phase 1 precomputation...");
    let (precomp_hint_a, precomp_hint_b) = pre_process_batch(&pvw_params, &evaluator, &clues);

    #[cfg(feature = "level")]
    let range_precompute_level = 10;

    #[cfg(not(feature = "level"))]
    let range_precompute_level = 0;

    let precomp_range_constants = precompute_range_constants(
        &evaluator
            .params()
            .poly_ctx(&PolyType::Q, range_precompute_level),
    );
    let precomp_sub_from_one = sub_from_one_precompute(evaluator.params(), range_precompute_level);

    // Running phase 1
    println!("Phase 1...");
    time_it!("Phase 1",
        let mut phase1_ciphertext = phase1(
            &evaluator,
            &pvw_params,
            &precomp_hint_a,
            &precomp_hint_b,
            &precomp_range_constants,
            &precomp_sub_from_one,
            &ek,
            &sk,
            &pvw_sk_cts,
        );
    );

    // mod down to level 12 irrespective of whether level feature is enabled. Otherwise, phase2 will be very expensive.
    let level = 12;
    evaluator.mod_down_level(&mut phase1_ciphertext, level);

    print_noise!(
        println!(
            "phase 1 noise (after mod down): {}",
            evaluator.measure_noise(&sk, &phase1_ciphertext)
        );
    );

    // Precomp phase 2
    let (_, buckets, weights) = assign_buckets_and_weights(
        K * 2,
        GAMMA,
        evaluator.params().plaintext_modulus,
        evaluator.params().degree,
        &mut rng,
    );

    let (pts_32_batch, pts_4_roll, pts_1_roll) =
        phase2_precomputes(&evaluator, evaluator.params().degree, level);

    // Phase 2
    println!("Phase 2...");
    time_it!("Phase 2",
        let (indices_ct, weights_ct) = phase2(
            &evaluator,
            &mut phase1_ciphertext,
            &ek,
            &buckets,
            &weights,
            &payloads,
            &pts_32_batch,
            &pts_4_roll,
            &pts_1_roll,
            level,
            &sk,
        );
    );

    // Client
    time_it!("Client",
        let messages = client_processing(
            &evaluator,
            &indices_ct,
            &weights_ct,
            &sk,
            &buckets,
            &weights,
        );
    );

    // Check correctness
    let mut expected_messages = vec![];
    pertinent_indices.iter().for_each(|i| {
        expected_messages.push(payloads[*i].clone());
    });

    assert_eq!(expected_messages, messages);
}

fn phase1(
    evaluator: &Evaluator,
    pvw_params: &PvwParameters,
    precomp_hint_a: &[Plaintext],
    precomp_hint_b: &[Poly],
    precomp_range_constants: &Array2<u64>,
    precomp_sub_from_one: &[u64],
    ek: &EvaluationKey,
    sk: &SecretKey,
    #[cfg(feature = "precomp_pvw")] precomputed_pvw_sk_cts: &[Vec<Ciphertext>],
    #[cfg(not(feature = "precomp_pvw"))] pvw_sk_cts: &[Ciphertext],
) -> Ciphertext {
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
        precomp_hint_a,
        precomp_hint_b,
        &pvw_sk_cts,
        ek.get_rtg_ref(1, 0),
        &sk,
    );

    print_noise!(
        decrypted_cts.iter().enumerate().for_each(|(index, ct)| {
            println!(
                "Decrypted Ct {index} noise: {}",
                evaluator.measure_noise(&sk, ct)
            );
        });
    );

    // ranged cts are in coefficient form
    let ranged_cts = range_fn_4_times(
        &decrypted_cts,
        &evaluator,
        &ek,
        precomp_range_constants,
        precomp_sub_from_one,
        &sk,
    );

    print_noise!(
        println!(
            "Ranged cts noise: {} {} {} {}",
            evaluator.measure_noise(sk, &ranged_cts.0 .0),
            evaluator.measure_noise(sk, &ranged_cts.0 .1),
            evaluator.measure_noise(sk, &ranged_cts.1 .0),
            evaluator.measure_noise(sk, &ranged_cts.1 .1)
        );
    );

    let v = mul_and_reduce_ranged_cts_to_1(&ranged_cts, &evaluator, &ek, &sk);

    print_noise!(
        println!(
            "Phase 1 end noise (before mod down): {}",
            evaluator.measure_noise(sk, &v)
        );
    );

    v
}

fn phase2(
    evaluator: &Evaluator,
    pv: &mut Ciphertext,
    ek: &EvaluationKey,
    buckets: &[Vec<u64>],
    weights: &[Vec<u64>],
    payloads: &[Vec<u64>],
    pts_32_batch: &[Plaintext],
    pts_4_roll: &[Plaintext],
    pts_1_roll: &[Plaintext],
    level: usize,
    sk: &SecretKey,
) -> (Ciphertext, Ciphertext) {
    // switch to Evaluation representation for efficient rotations and scalar multiplications
    evaluator.ciphertext_change_representation(pv, Representation::Evaluation);

    let (mut indices_ct, mut weights_ct) = phase2::phase2(
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

    print_noise!(
        println!(
            "Phase 2 end (before mod down noise) - indices_ct:{}, weights_ct:{} ",
            evaluator.measure_noise(&sk, &indices_ct),
            evaluator.measure_noise(&sk, &weights_ct)
        );
    );

    // mod down to last level
    evaluator.mod_down_level(&mut indices_ct, 14);
    evaluator.mod_down_level(&mut weights_ct, 14);

    print_noise!(
        println!(
            "Phase 2 end (after mod down noise) - indices_ct:{}, weights_ct:{} ",
            evaluator.measure_noise(&sk, &indices_ct),
            evaluator.measure_noise(&sk, &weights_ct)
        );
    );

    (indices_ct, weights_ct)
}

fn client_processing(
    evaluator: &Evaluator,
    indices_ct: &Ciphertext,
    weights_ct: &Ciphertext,
    sk: &SecretKey,
    buckets: &[Vec<u64>],
    weights: &[Vec<u64>],
) -> Vec<Vec<u64>> {
    print_noise!(
        println!(
            "Client's indices_ct noise : {}",
            evaluator.measure_noise(&sk, &indices_ct)
        );
        println!(
            "Client's weights_ct noise : {}",
            evaluator.measure_noise(&sk, &weights_ct)
        );
    );

    // construct lhs
    let pv = pv_decompress(evaluator, indices_ct, sk);
    let lhs = construct_lhs(&pv, buckets, weights, K, GAMMA, evaluator.params().degree);

    // decrypt weights_ct to construct rhs
    let weights_vec =
        evaluator.plaintext_decode(&evaluator.decrypt(sk, weights_ct), Encoding::default());
    let rhs = construct_rhs(&weights_vec, BUCKET_SIZE);

    let messages = solve_equations(lhs, rhs, K, evaluator.params().plaintext_modulus);
    messages
}

fn powers_of_x() {
    let mut rng = thread_rng();
    let params = generate_bfv_parameters();
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
        evaluate_powers(&evaluator, &ek, 2, 4, &mut calculated, false, cores,&sk);
        evaluate_powers(&evaluator, &ek, 4, 8, &mut calculated, false, cores,&sk);
        evaluate_powers(&evaluator, &ek, 8, 16, &mut calculated, false, cores,&sk);
        evaluate_powers(&evaluator, &ek, 16, 32, &mut calculated, false, cores,&sk);
        evaluate_powers(&evaluator, &ek, 32, 64, &mut calculated, false, cores,&sk);
        evaluate_powers(&evaluator, &ek, 64, 128, &mut calculated, false, cores,&sk);
        evaluate_powers(&evaluator, &ek, 128, 256, &mut calculated, false, cores,&sk);
    );
}

fn call_pvw_decrypt() {
    let mut rng = thread_rng();
    let pvw_params = Arc::new(PvwParameters::default());
    let pvw_sk = PvwSecretKey::random(&pvw_params, &mut rng);
    let pvw_pk = pvw_sk.public_key(&mut rng);

    let params = generate_bfv_parameters();
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
        let res = pvw_decrypt(
                &pvw_params,
                &evaluator,
                &hint_a,
                &hint_b,
                &pvw_sk_cts,
                ek.get_rtg_ref(1, 0),
                &sk,
            );
    );

    dbg!(evaluator.measure_noise(&sk, &res[0]));
    dbg!(evaluator.measure_noise(&sk, &res[1]));
    dbg!(evaluator.measure_noise(&sk, &res[2]));
    dbg!(evaluator.measure_noise(&sk, &res[3]));
}

fn call_pvw_decrypt_precomputed() {
    let mut rng = thread_rng();
    let pvw_params = Arc::new(PvwParameters::default());
    let pvw_sk = PvwSecretKey::random(&pvw_params, &mut rng);
    let pvw_pk = pvw_sk.public_key(&mut rng);

    let params = generate_bfv_parameters();
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
    let params = generate_bfv_parameters();
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
    // demo();
    print_detection_key_size();
    // call_pvw_decrypt();
    // call_pvw_decrypt_precomputed();

    // powers_of_x();
    // call_range_fn_once();

    // range_fn_trial();
}
