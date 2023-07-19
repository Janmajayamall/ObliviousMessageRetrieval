use bfv::{
    Ciphertext, CiphertextProto, Encoding, EvaluationKey, EvaluationKeyProto, Evaluator, Plaintext,
    Poly, PolyType, Representation, SecretKey, TryFromWithParameters,
};
use itertools::Itertools;
use ndarray::Array2;
use omr::{
    client::{
        construct_lhs, construct_rhs, encrypt_pvw_sk, evaluation_key, pv_decompress,
        solve_equations,
    },
    level_down,
    optimised::sub_from_one_precompute,
    preprocessing::{assign_buckets_and_weights, pre_process_batch},
    print_noise,
    pvw::*,
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
use prost::Message;
use rand::{thread_rng, Rng};

// Functions to check runtime of expensive components //

fn powers_of_x() {
    let mut rng = thread_rng();
    let params = generate_bfv_parameters();
    let sk = SecretKey::random_with_params(&params, &mut rng);
    let ek = EvaluationKey::new(&params, &sk, &[0], &[], &[], &mut rng);

    let m = params
        .plaintext_modulus_op
        .random_vec(params.degree, &mut rng);
    let evaluator = Evaluator::new(params);
    let pt = evaluator.plaintext_encode(&m, Encoding::default());
    let ct = evaluator.encrypt(&sk, &pt, &mut rng);

    time_it!("Powers of x [1,256) time: ",
        let placeholder = Ciphertext::placeholder();
        let mut calculated = vec![placeholder.clone(); 255];
        calculated[0] = ct;
        evaluate_powers(&evaluator, &ek, 2, 4, &mut calculated, false, &sk);
        evaluate_powers(&evaluator, &ek, 4, 8, &mut calculated, false, &sk);
        evaluate_powers(&evaluator, &ek, 8, 16, &mut calculated, false, &sk);
        evaluate_powers(&evaluator, &ek, 16, 32, &mut calculated, false, &sk);
        evaluate_powers(&evaluator, &ek, 32, 64, &mut calculated, false, &sk);
        evaluate_powers(&evaluator, &ek, 64, 128, &mut calculated, false, &sk);
        evaluate_powers(&evaluator, &ek, 128, 256, &mut calculated, false, &sk);
    );
}

fn call_pvw_decrypt() {
    let mut rng = thread_rng();
    let pvw_params = PvwParameters::default();
    let pvw_sk = PvwSecretKey::random(&pvw_params, &mut rng);
    let pvw_pk = pvw_sk.public_key(&mut rng);

    let params = generate_bfv_parameters();
    let sk = SecretKey::random_with_params(&params, &mut rng);

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
    let pvw_params = PvwParameters::default();
    let pvw_sk = PvwSecretKey::random(&pvw_params, &mut rng);
    let pvw_pk = pvw_sk.public_key(&mut rng);

    let params = generate_bfv_parameters();
    let sk = SecretKey::random_with_params(&params, &mut rng);

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

    let sk = SecretKey::random_with_params(&params, &mut rng);
    let m = params
        .plaintext_modulus_op
        .random_vec(params.degree, &mut rng);

    let evaluator = Evaluator::new(params);
    let pt = evaluator.plaintext_encode(&m, Encoding::simd(level));
    let ct = evaluator.encrypt(&sk, &pt, &mut rng);

    // gen evaluation key
    let ek = EvaluationKey::new(evaluator.params(), &sk, &[0], &[], &[], &mut rng);

    time_it!("Range fn time: ",
        let _ = range_fn(&ct, &evaluator, &ek, &constants, &sub_one_precompute, &sk);
    );

    let cts = (0..4).into_iter().map(|_| ct.clone()).collect_vec();
    time_it!("Range fn 4 times: ",
        let _ = range_fn_4_times(&cts, &evaluator, &ek, &constants, &sub_one_precompute, &sk);
    );
}

// END //

fn print_pvw_decrypt_precompute_size() {
    let mut rng = thread_rng();

    let params = generate_bfv_parameters();
    let sk = SecretKey::random_with_params(&params, &mut rng);
    let evaluator = Evaluator::new(params);
    let pvw_params = PvwParameters::default();
    let pvw_sk = PvwSecretKey::random(&pvw_params, &mut rng);

    let pvw_sk_cts = encrypt_pvw_sk(&evaluator, &sk, &pvw_sk, &mut rng);

    // Don't call `pvw_setup` to get rotations of all `pvw_sk_cts` since it will take sometime. Simply
    // calculate no. of checkpoints required per ciphertext in `pvw_sk_ct` for available threads.

    let total_threads = rayon::current_num_threads();
    assert!(total_threads % 4 == 0);
    assert!(total_threads >= 8);
    // Assign equal no. of threads to each of 4 instances of pvw decrypt
    let threads_by_4 = total_threads / 4;

    // Each instance of pvw decrypt performs 512 rotations and plaintext multiplications of single pvw_sk_ct. Since rotations
    // are serial, to parallelise we must precompute rotatations of pvw_sk_ct at specific checkpoints (depending on threads_by_4) and store them.
    // For ex, when threads_by_4 = 4 it means each call of pvw_decrypt has 4 threads. So we precompute rotations of
    // pvw_sk_ct at checkpints that divide 512 into 4 parts (ie 512/threads_by_4) and each part can be processed serially.
    // Thus the checkpoints must be 4 instances of pvw_sk_ct rotated by 0, 128, 256, 384. At run time each thread will perform
    // `512/4 = 128` rotations.
    let instances_of_each_pvw_sk = threads_by_4;

    let total_ciphertexts_stored = instances_of_each_pvw_sk * pvw_sk_cts.len();

    // Size of each of the 4 `pvw_sk_cts` as well as their `threads_by_4` instances obtained after rotations at checkpoints are equal.
    // So we can serialise just one ciphertext and estimate the total size.
    let mut ct = pvw_sk_cts[0].clone();
    // Serialisation is not allowed in Evaluation representation since `NTT` backend may differ. This may change in future.
    evaluator.ciphertext_change_representation(&mut ct, Representation::Coefficient);
    let proto_ct = CiphertextProto::try_from_with_parameters(&ct, evaluator.params());
    let size = proto_ct.encode_to_vec().len();

    println!(
        "Pvw Precompute Size per User: {} bytes",
        size * total_ciphertexts_stored
    );
}

fn print_detection_key_size() {
    let mut rng = thread_rng();

    let params = generate_bfv_parameters();

    // Client's PVW keys
    let pvw_params = PvwParameters::default();
    let pvw_sk = PvwSecretKey::random(&pvw_params, &mut rng);

    // Client's BFV keys
    let sk = SecretKey::random_with_params(&params, &mut rng);

    let evaluator = Evaluator::new(params);

    // Client's detection keys
    let mut pvw_sk_cts = encrypt_pvw_sk(&evaluator, &sk, &pvw_sk, &mut rng);
    let ek = evaluation_key(evaluator.params(), &sk, &mut rng);

    // change representation of pvw_sk_cts from Evaluation to Coefficient due the possibility that we might
    // use different NTT backends in future
    pvw_sk_cts.iter_mut().for_each(|c| {
        evaluator.ciphertext_change_representation(c, Representation::Coefficient);
    });

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
    if !rayon::current_num_threads().is_power_of_two() {
        println!("WARNING: Threads available is not a power of 2. Program isn't capable of fully utilising all cores if total cores are not a power of 2. Moreover, program might panic!");
    }

    let mut rng = thread_rng();

    let params = generate_bfv_parameters();

    // Client's PVW keys
    let pvw_params = PvwParameters::default();
    let pvw_sk = PvwSecretKey::random(&pvw_params, &mut rng);
    let pvw_pk = pvw_sk.public_key(&mut rng);

    // Client's BFV keys
    let sk = SecretKey::random_with_params(&params, &mut rng);

    // a random secret key as a placeholder to be passed around in functions. Should be
    // removed in production.
    let mut random_sk = SecretKey::random_with_params(&params, &mut rng);

    #[cfg(feature = "noise")]
    {
        // set `random_sk` as `sk` to print correct noise
        random_sk = sk.clone();
    }

    let evaluator = Evaluator::new(params);

    // Client's detection keys
    let pvw_sk_cts = encrypt_pvw_sk(&evaluator, &sk, &pvw_sk, &mut rng);
    let ek = evaluation_key(evaluator.params(), &sk, &mut rng);

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

    // Generate clues and payloads
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

    #[cfg(feature = "precomp_pvw")]
    let pvw_decrypt_precompute = {
        if rayon::current_num_threads() < 8 {
            panic!("Feature `precomp_pvw` does not support threads < 8");
        }
        pvw_setup(&evaluator, &ek, &pvw_sk_cts)
    };

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
            &random_sk,
            #[cfg(not(feature = "precomp_pvw"))]
            &pvw_sk_cts,
            #[cfg(feature = "precomp_pvw")]
            &pvw_decrypt_precompute
        );
    );

    // phase1 mods down to level 12 in the end irrespective of whether level feature is enabled. Otherwise, phase2 will be very expensive.
    let level = 12;

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
            &random_sk,
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
            &precomp_hint_a,
            &precomp_hint_b,
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

    let mut v = mul_and_reduce_ranged_cts_to_1(&ranged_cts, &evaluator, &ek, &sk);

    print_noise!(
        println!(
            "Phase 1 end noise (before mod down): {}",
            evaluator.measure_noise(sk, &v)
        );
    );

    // mod down to level 12 irrespective of whether `level` feature is enabled or not. Otherwise
    // phase 2 will be very expensive
    evaluator.mod_down_level(&mut v, 12);

    print_noise!(
        println!(
            "Phase 1 end noise (after mod down): {}",
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

    level_down!(
        // mod down to last level
        evaluator.mod_down_level(&mut indices_ct, 14);
        evaluator.mod_down_level(&mut weights_ct, 14);
    );

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

fn main() {
    let threads = std::env::args().nth(1);

    // set global thread pool using provided thread count
    if threads.is_some() {
        let threads = threads
            .unwrap()
            .parse::<usize>()
            .expect("Threads must be a positive integer");
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap()
    }

    demo();

    // print_pvw_decrypt_precompute_size();
    // print_detection_key_size();
    // call_pvw_decrypt();
    // call_pvw_decrypt_precomputed();

    // powers_of_x();
    // call_range_fn_once();

    // range_fn_trial();
}
