use bfv::{
    BfvParameters, Ciphertext, Encoding, EvaluationKey, Evaluator, GaloisKey, Modulus, Plaintext,
    PolyType, RelinearizationKey, Representation, SecretKey,
};
use itertools::{izip, Itertools};
use ndarray::Array2;
use rand::thread_rng;
use rayon::{
    prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::{
    cell::Cell,
    collections::{HashMap, HashSet},
    hash::Hash,
    sync::Arc,
};

use omr::{
    client::{encrypt_pvw_sk, gen_pv_exapnd_rtgs},
    optimised::{coefficient_u128_to_ciphertext, sub_from_one_precompute},
    preprocessing::{pre_process_batch, precompute_expand_32_roll_pt, precompute_indices_pts},
    pvw::*,
    server::{
        phase2,
        powers_x::{even_powers_of_x_ct, powers_of_x_ct},
        pvw_decrypt, range_fn,
    },
    utils::precompute_range_constants,
};

fn phase1() {
    let mut rng = thread_rng();
    let pvw_params = Arc::new(PvwParameters::default());
    let pvw_sk = PvwSecretKey::random(&pvw_params, &mut rng);
    let pvw_pk = pvw_sk.public_key(&mut rng);

    let params = BfvParameters::default(15, 1 << 15);
    let sk = SecretKey::random(params.degree, &mut rng);

    // generate hints
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

    println!("Running Pvw decrypt...");
    let mut now = std::time::Instant::now();
    let decryted_cts = pvw_decrypt(
        &pvw_params,
        &evaluator,
        &hint_a,
        &hint_b,
        &pvw_sk_cts,
        ek.get_rtg_ref(1, 0),
        &sk,
    );
    println!("Pvw decryption time: {:?}", now.elapsed());

    decryted_cts.iter().for_each(|ct| {
        println!("Decrypted Ct noise: {}", evaluator.measure_noise(&sk, ct));
    });

    let constants = precompute_range_constants(&evaluator.params().poly_ctx(&PolyType::Q, 0));
    let sub_from_one_precompute = sub_from_one_precompute(evaluator.params(), 0);

    println!("Running range function...");
    // ranged cts are in coefficient form
    now = std::time::Instant::now();
    let ranged_cts = decryted_cts
        .iter()
        .map(|ct| {
            range_fn(
                &ct,
                &evaluator,
                &ek,
                &constants,
                &sub_from_one_precompute,
                &sk,
            )
        })
        .collect_vec();
    println!("Range function time: {:?}", now.elapsed());

    ranged_cts.iter().for_each(|ct| {
        assert!(ct.c_ref()[0].representation == Representation::Coefficient);
        println!("Ranged Ct noise: {}", evaluator.measure_noise(&sk, ct));
    });

    now = std::time::Instant::now();
    // multiplication tree
    // ct[0] * ct[1]          ct[2] * ct[4]
    //      v0         *          v1
    //                v
    let v0 = evaluator.mul(&ranged_cts[0], &ranged_cts[0]);
    let v0 = evaluator.relinearize(&v0, &ek);
    let v1 = evaluator.mul(&ranged_cts[2], &ranged_cts[3]);
    let v1 = evaluator.relinearize(&v1, &ek);

    println!("v0 noise: {}", evaluator.measure_noise(&sk, &v0));
    println!("v1 noise: {}", evaluator.measure_noise(&sk, &v1));

    let v = evaluator.mul(&v0, &v1);
    // Relinearization of `v` can be modified such that overall ntts can be minized.
    // We expect `v` to be in evaluation form. Thus we convert c0 and c1, not c2, to evaluation form
    // after scale_and_round op. key_switch `c2` (c2 stays in coeffciient form). The c0' and
    // c1' (key switch outputs in evaluation form) to c0 and c1 respectively. Instead if we
    // use normal `relinearize` the output will be in coefficient form and will have to pay
    // for additional 2 Ntts of size Q to convert ouput to evaluation.
    let mut v = evaluator.relinearize(&v, &ek);
    println!("Multiplication time: {:?}", now.elapsed());

    println!("phase 1 end ct noise: {}", evaluator.measure_noise(&sk, &v));

    evaluator.ciphertext_change_representation(&mut v, Representation::Evaluation);
    phase2(&evaluator, &sk, &mut v);
}

fn phase2(evaluator: &Evaluator, sk: &SecretKey, pv: &mut Ciphertext) {
    evaluator.mod_down_level(pv, 12);
    dbg!(pv.level());
    let ek = gen_pv_exapnd_rtgs(evaluator.params(), sk, 12);
    phase2::phase2(evaluator, pv, &ek, sk, 12);

    println!("Phase 2 end noise: {}", evaluator.measure_noise(sk, pv));
}

/// Time of `even_powers_of_x_ct` should be half of `powers_of_x_ct`
fn time_dff_even_all_powers_of_x() {
    let mut rng = thread_rng();
    let params = BfvParameters::default(15, 1 << 8);
    let sk = SecretKey::random(params.degree, &mut rng);
    let ek = EvaluationKey::new(&params, &sk, &[0], &[], &[], &mut rng);

    let m = vec![3; params.degree];

    let evaluator = Evaluator::new(params);
    let pt = evaluator.plaintext_encode(&m, Encoding::default());
    let ct = evaluator.encrypt(&sk, &pt, &mut rng);

    {
        for _ in 0..2 {
            let _ = powers_of_x_ct(&ct, &evaluator, &ek, &sk);
        }
    }

    let now = std::time::Instant::now();
    let even_powers_ct = even_powers_of_x_ct(&ct, &evaluator, &ek, &sk);
    println!("Time even_powers_of_x_ct = {:?}", now.elapsed());

    let now = std::time::Instant::now();
    let powers_ct = powers_of_x_ct(&ct, &evaluator, &ek, &sk);
    println!("Time powers_of_x_ct = {:?}", now.elapsed());
}

fn main() {}
