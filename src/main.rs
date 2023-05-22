use bfv::{BfvParameters, Ciphertext, Encoding, Modulus, Plaintext, RelinearizationKey, SecretKey};
use itertools::izip;
use rand::thread_rng;
use std::sync::Arc;

use omr::{
    plaintext::{powers_of_x_int, powers_of_x_modulus},
    server::powers_of_x_ct,
};

fn main() {
    // let res_values = powers_of_x(8);
    // let mut values = vec![0u64; 256];
    // for i in (2..(256 + 1)).rev() {
    //     values[i - 1] = 8u64.wrapping_pow(i as u32);
    // }
    // assert_eq!(values, res_values);
    let mut rng = thread_rng();
    let params = Arc::new(BfvParameters::default(4, 1 << 11));
    let sk = SecretKey::random(&params, &mut rng);
    let m = vec![8; params.polynomial_degree];
    let pt = Plaintext::encode(&m, &params, Encoding::simd(0));
    let ct = sk.encrypt(&pt, &mut rng);

    // gen rlk
    let rlk = RelinearizationKey::new(&params, &sk, 0, &mut rng);

    let res_values_ct = powers_of_x_ct(&ct, &rlk);
    let res_values_mod = powers_of_x_modulus(8, &params.plaintext_modulus_op);
    izip!(res_values_ct, res_values_mod).for_each(|(ct, b)| {
        dbg!(sk.measure_noise(&ct, &mut rng));
        let r = sk.decrypt(&ct).decode(Encoding::simd(0));
        r.iter().for_each(|a| assert_eq!(*a, b));
    })
}
