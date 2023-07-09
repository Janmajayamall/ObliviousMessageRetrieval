use bfv::{
    BfvParameters, Ciphertext, Encoding, Evaluator, Modulus, PolyContext, RelinearizationKey,
    SecretKey,
};
use byteorder::{ByteOrder, LittleEndian};
use itertools::Itertools;
use ndarray::Array2;
use rand::thread_rng;
use std::io::Write;

pub fn read_range_coeffs() -> Vec<u64> {
    let bytes = include_bytes!("../target/params_850.bin");
    let mut coeffs = [0u64; 65536];
    LittleEndian::read_u64_into(bytes, &mut coeffs);
    coeffs.to_vec()
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

pub unsafe fn decrypt_and_print(evaluator: &Evaluator, ct: &Ciphertext, sk: &SecretKey, tag: &str) {
    let mut rng = thread_rng();
    let v = evaluator.plaintext_decode(&evaluator.decrypt(sk, ct), Encoding::default());
    println!(
        "{tag}= Noise: {}; m: {:?}",
        evaluator.measure_noise(sk, ct),
        &v[..0]
    );
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
