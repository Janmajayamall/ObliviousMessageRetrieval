use bfv::{Ciphertext, EvaluationKey, Evaluator, SecretKey};

pub mod phase2;
pub mod powers_x;
pub mod pvw_decrypt;
pub mod range_fn;

/// Mutliplies all ciphertexts output from range_fn together adn returns a single
/// ciphertext in Coefficient representation.
pub fn mul_and_reduce_ranged_cts_to_1(
    ranged_cts: &((Ciphertext, Ciphertext), (Ciphertext, Ciphertext)),
    evaluator: &Evaluator,
    ek: &EvaluationKey,
    sk: &SecretKey,
) -> Ciphertext {
    // Use binary tree multiplication to reduce multiplicative depth
    // ct[0] * ct[1]          ct[2] * ct[4]
    //      v0         *          v1
    //                v
    let (v0, v1) = rayon::join(
        || {
            let v0 = evaluator.mul(&ranged_cts.0 .0, &ranged_cts.0 .1);
            let v0 = evaluator.relinearize(&v0, &ek);
            v0
        },
        || {
            let v1 = evaluator.mul(&ranged_cts.1 .0, &ranged_cts.1 .1);
            let v1 = evaluator.relinearize(&v1, &ek);
            v1
        },
    );

    // println!("v0 noise: {}", evaluator.measure_noise(&sk, &v0));
    // println!("v1 noise: {}", evaluator.measure_noise(&sk, &v1));

    let v = evaluator.mul(&v0, &v1);
    // Relinearization of `v` can be modified such that overall ntts can be minized.
    // We expect `v` to be in evaluation form. Thus we convert c0 and c1, not c2, to evaluation form
    // after scale_and_round. The we key_switch `c2` and obtain c0' and c1' in Evaluation representation.
    // Then we add c0 and c1 with c0' and c1' respectively without having to switching c0' and
    // c1' to Coefficient representation. This saves us 2 NTTs compared to what we do at present.
    let mut v = evaluator.relinearize(&v, &ek);

    v
}
