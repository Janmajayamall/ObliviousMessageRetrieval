use bfv::{
    BfvParameters, Ciphertext, Encoding, Evaluator, Modulus, PolyContext, RelinearizationKey,
    SecretKey,
};
use byteorder::{ByteOrder, LittleEndian};
use itertools::Itertools;
use ndarray::Array2;
use rand::{
    distributions::{Standard, Uniform},
    thread_rng, Rng,
};
use std::io::Write;

use crate::MESSAGE_BYTES;

pub fn read_range_coeffs() -> Vec<u64> {
    let bytes = include_bytes!("../target/params_850.bin");
    let mut coeffs = [0u64; 65536];
    LittleEndian::read_u64_into(bytes, &mut coeffs);
    coeffs.to_vec()
}

// Measures time in ms for enclosed code block.
// Credit: https://github.com/zama-ai/demo_z8z/blob/1f24eeaf006263543062e90f1d1692d381a726cf/src/zqz/utils.rs#L28C1-L42C2
#[macro_export]
macro_rules! time_it{
    ($title: tt, $($block:tt)+) => {
        let __now = std::time::SystemTime::now();
        $(
           $block
        )+
        let __time = __now.elapsed().unwrap().as_millis();
        let __ms_time = format!("{} ms", __time);
        println!("{} duration: {}", $title, __ms_time);
    }
}

pub fn store_range_coeffs() {
    let prime = 65537;
    let range = 850;
    let mut sums = vec![];
    for i in 1..prime {
        let mut sum = 0;
        let modq = Modulus::new(prime);
        for a in 0..prime {
            if a <= range || a >= (prime - range) {
                sum = modq.add_mod(sum, modq.exp(a, (prime - 1 - i).try_into().unwrap()));
            }
        }
        sums.push(sum);
    }
    let mut buf = [0u8; 65536 * 8];
    LittleEndian::write_u64_into(&sums, &mut buf);
    let mut f = std::fs::File::create("params_850.bin").unwrap();
    f.write_all(&buf).unwrap();
}

pub fn precompute_range_constants(ctx: &PolyContext<'_>) -> Array2<u64> {
    let coeffs = read_range_coeffs();
    let v = coeffs
        .iter()
        .flat_map(|c| ctx.iter_moduli_ops().map(|modqi| *c % modqi.modulus()))
        .collect_vec();

    Array2::from_shape_vec((65536usize, ctx.moduli_count()), v).unwrap()
}

pub fn generate_random_payloads(set_size: usize) -> Vec<Vec<u64>> {
    let rng = thread_rng();
    let mut payloads = Vec::with_capacity(set_size);
    (0..set_size).into_iter().for_each(|_| {
        let msg: Vec<u64> = rng
            .clone()
            .sample_iter(Uniform::new(0, (1 << 16)))
            .take(MESSAGE_BYTES / 2)
            .collect_vec();
        payloads.push(msg);
    });
    payloads
}

pub unsafe fn decrypt_and_print(evaluator: &Evaluator, ct: &Ciphertext, sk: &SecretKey, tag: &str) {
    let mut rng = thread_rng();
    let v = evaluator.plaintext_decode(&evaluator.decrypt(sk, ct), Encoding::default());
    println!(
        "{tag}= Noise: {}; m: {:?}",
        evaluator.measure_noise(sk, ct),
        &v[..0]
    );
}

pub fn generate_bfv_parameters() -> BfvParameters {
    let moduli = vec![50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 60];
    let mut params = BfvParameters::new(&moduli, 65537, 1 << 15);
    params.enable_hybrid_key_switching(&[50, 50, 50]);
    params
}

mod tests {
    use bfv::Modulus;

    use super::*;

    #[test]
    fn test_store_range_coeffs() {
        store_range_coeffs();
    }

    #[test]
    fn range_coeffs_zeros_count() {
        let coeffs = read_range_coeffs();
        let mut count0 = 0;
        let mut count1 = 0;
        coeffs.iter().for_each(|c| {
            if *c == 0 {
                count0 += 1;
            }
            if *c == 2 {
                count1 += 1;
            }
        });
        coeffs.iter().step_by(2).for_each(|c| assert!(*c == 0));
        dbg!(count0);
        dbg!(count1);
        dbg!(coeffs.iter().max());
        // println!("{:?}", &coeffs[..2]);
    }
}
