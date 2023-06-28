use bfv::{
    BfvParameters, Ciphertext, Encoding, Evaluator, PolyContext, RelinearizationKey, SecretKey,
};
use byteorder::{ByteOrder, LittleEndian};
use itertools::Itertools;
use ndarray::Array2;
use rand::thread_rng;

pub fn read_range_coeffs() -> Vec<u64> {
    let bytes = include_bytes!("../target/params_850.bin");
    let mut coeffs = [0u64; 65536];
    LittleEndian::read_u64_into(bytes, &mut coeffs);
    coeffs.to_vec()
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
        &v[..256]
    );
}

mod test {
    use super::*;

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
        println!("{:?}", coeffs);
    }
}
