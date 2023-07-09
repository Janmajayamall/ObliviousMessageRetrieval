use bfv::Modulus;
use rayon::{
    prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use crate::utils::read_range_coeffs;

fn evalulate_powers(start: usize, end: usize, calculated: &mut [u64], bases: bool) {
    // start is always a power of two. -1 is equal to mask.
    let mask = start - 1;

    // all values from start..start*2 require calculated value of start. So we calculated it here
    if !bases {
        calculated[start - 1] = calculated[start / 2 - 1] * calculated[start / 2 - 1];
    }

    // split at start+1 to assure that mutated values are disjoint
    let (done, pending) = calculated.split_at_mut(start - 1 + 1);

    // we only operate on slice from start+1 till end-1
    let pending = &mut pending[..(end - 1 - start)];
    let cores = 1;
    let size = (pending.len() as f64 / cores as f64).ceil() as usize;

    let pending_chunks = pending.par_chunks_mut(size);

    pending_chunks
        .enumerate()
        .into_par_iter()
        .for_each(|(index, chunk)| {
            chunk.iter_mut().enumerate().for_each(|(chunk_index, v)| {
                // calculate real_index of the value to figure out the mask
                let real_index = size * index + chunk_index + start + 1;
                *v = done.last().unwrap() * done[(real_index & mask) - 1];
            });
        });
}

pub fn powers_of_x_modulus(x: u64, modq: &Modulus, degree: usize) -> Vec<u64> {
    let mut cache = vec![0u64; degree];
    let mut calculated = vec![0u64; degree];
    cache[0] = x;
    calculated[0] = 1;
    let mut mul_count = 0;
    for i in (0..(degree + 1)).rev() {
        let mut exp = i;
        let mut base = 1usize;
        let mut res = 0usize;
        while exp > 0 {
            if exp & 1 == 1 {
                if res != 0 && calculated[res + base - 1] == 0 {
                    cache[res + base - 1] = modq.mul_mod_fast(cache[res - 1], cache[base - 1]);
                    calculated[res + base - 1] = 1;
                }
                res += base;
            }
            exp >>= 1;
            if exp != 0 {
                if calculated[base * 2 - 1] == 0 {
                    cache[base * 2 - 1] = modq.mul_mod_fast(cache[base - 1], cache[base - 1]);
                    calculated[base * 2 - 1] = 1;
                }
                base *= 2;
            }
        }
    }
    cache
}

fn range_fn_modulus() {
    let modq = Modulus::new(65537);

    // value on which range_fn is evaluated
    let v = 28192;
    let single = powers_of_x_modulus(v, &modq, 256);
    let double = powers_of_x_modulus(single[255], &modq, 255);

    let range_coeff = read_range_coeffs();

    let mut sum = 0;
    for i in 0..256 {
        let mut tmp = 0;
        for j in 1..257 {
            tmp = modq.add_mod_fast(
                tmp,
                modq.mul_mod_fast(single[j - 1], range_coeff[(i * 256) + (j - 1)]),
            );
        }

        if i == 0 {
            sum = tmp;
        } else {
            sum = modq.add_mod_fast(sum, modq.mul_mod_fast(double[i - 1], tmp));
        }
        dbg!(sum);
    }

    sum = modq.sub_mod_fast(1, sum);
    dbg!(sum);
}

mod tests {
    use super::evalulate_powers;

    fn test_evaluate_even_powers() {
        let mut calculated = vec![0; 128];
        calculated[0] = 9;
        calculated[4 / 2 - 1] = 3u64.pow(4);
        calculated[8 / 2 - 1] = 3u64.pow(8);
        calculated[16 / 2 - 1] = 3u64.pow(16);
        calculated[32 / 2 - 1] = 3u64.pow(32);
        calculated[64 / 2 - 1] = 3u64.pow(64);
        calculated[128 / 2 - 1] = 3u64.pow(128);
        calculated[256 / 2 - 1] = 3u64.pow(256);
        // even powers
        evalulate_powers(2, 4, &mut calculated, true);
        evalulate_powers(4, 8, &mut calculated, true);
        evalulate_powers(8, 16, &mut calculated, true);
        evalulate_powers(16, 32, &mut calculated, true);
        evalulate_powers(32, 64, &mut calculated, true);
        evalulate_powers(64, 128, &mut calculated, true);
    }
}
