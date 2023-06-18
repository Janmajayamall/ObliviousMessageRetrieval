use bfv::{BfvParameters, Ciphertext, Encoding, PolyContext, RelinearizationKey, SecretKey};
use byteorder::{ByteOrder, LittleEndian};
use itertools::Itertools;
use ndarray::Array2;
use rand::thread_rng;
use std::sync::Arc;
use traits::Ntt;

pub fn read_range_coeffs() -> Vec<u64> {
    let bytes = include_bytes!("../target/params_850.bin");
    let mut coeffs = [0u64; 65536];
    LittleEndian::read_u64_into(bytes, &mut coeffs);
    coeffs.to_vec()
}

pub fn precompute_range_constants<T: Ntt>(ctx: &PolyContext<T>) -> Array2<u64> {
    let coeffs = read_range_coeffs();
    let v = coeffs
        .iter()
        .flat_map(|c| ctx.moduli.iter().map(|qi| *c % *qi))
        .collect_vec();

    Array2::from_shape_vec((65536usize, ctx.moduli.len()), v).unwrap()
}

pub unsafe fn decrypt_and_print<T: Ntt>(ct: &Ciphertext<T>, sk: &SecretKey<T>, tag: &str) {
    let mut rng = thread_rng();
    let v = sk.decrypt(ct).decode(Encoding::simd(0));
    println!(
        "{tag}= Noise: {}; m: {:?}",
        sk.measure_noise(ct, &mut rng),
        "Too big!"
    );
}

fn gen_rlks<T: traits::Ntt>(
    levels: &[usize],
    sk: &SecretKey<T>,
    params: &Arc<BfvParameters<T>>,
) -> Vec<RelinearizationKey<T>> {
    let mut rng = thread_rng();
    levels
        .iter()
        .map(|l| RelinearizationKey::new(params, sk, *l, &mut rng))
        .collect_vec()
}

pub fn gen_rtgs() {}

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
