use bfv::Modulus;

use crate::utils::read_range_coeffs;

pub fn powers_of_x_int(x: u64) -> Vec<u64> {
    let mut val = 256;
    let mut cache = vec![0u64; 256];
    let mut calculated = vec![0u64; 256];
    let mut mul_count = 0;
    for i in (0..(val + 1)).rev() {
        let mut exp = i;
        let mut base = x;
        let mut res = 1u64;
        let mut base_deg = 1u64;
        let mut res_deg = 0u64;
        while exp > 0 {
            if exp & 1 == 1 {
                let p_res_deg = res_deg;
                res_deg += base_deg;
                if calculated[(res_deg - 1) as usize] == 1 {
                    res = cache[(res_deg - 1) as usize];
                } else {
                    if res_deg == base_deg {
                        res = base;
                    } else {
                        res = res.wrapping_mul(base);
                        println!("{p_res_deg}, {base_deg}");
                        mul_count += 1;
                    }
                    cache[(res_deg - 1) as usize] = res;
                    calculated[(res_deg - 1) as usize] = 1;
                }
            }
            exp >>= 1;
            if exp != 0 {
                let tmp = base_deg;
                base_deg *= 2;
                if calculated[(base_deg - 1) as usize] == 0 {
                    base = base.wrapping_mul(base);
                    println!("{tmp} base_deg");
                    mul_count += 1;
                    cache[(base_deg - 1) as usize] = base;
                    calculated[(base_deg - 1) as usize] = 1;
                } else {
                    base = cache[(base_deg - 1) as usize];
                }
            }
        }
    }
    // dbg!(cache);
    // dbg!(mul_count);
    cache
}

pub fn powers_of_x_modulus(x: u64, modq: &Modulus) -> Vec<u64> {
    let mut val = 256;
    let mut cache = vec![0u64; 256];
    let mut calculated = vec![0u64; 256];
    let mut mul_count = 0;
    for i in (0..(val + 1)).rev() {
        let mut exp = i;
        let mut base = x;
        let mut res = 1u64;
        let mut base_deg = 1u64;
        let mut res_deg = 0u64;
        while exp > 0 {
            if exp & 1 == 1 {
                let p_res_deg = res_deg;
                res_deg += base_deg;
                if calculated[(res_deg - 1) as usize] == 1 {
                    res = cache[(res_deg - 1) as usize];
                } else {
                    if res_deg == base_deg {
                        res = base;
                    } else {
                        res = modq.mul_mod_fast(res, base);
                        mul_count += 1;
                    }
                    cache[(res_deg - 1) as usize] = res;
                    calculated[(res_deg - 1) as usize] = 1;
                }
            }
            exp >>= 1;
            if exp != 0 {
                base_deg *= 2;
                if calculated[(base_deg - 1) as usize] == 0 {
                    base = modq.mul_mod_fast(base, base);
                    mul_count += 1;
                    cache[(base_deg - 1) as usize] = base;
                    calculated[(base_deg - 1) as usize] = 1;
                } else {
                    base = cache[(base_deg - 1) as usize];
                }
            }
        }
    }
    cache
}

fn range_fn_modulus() {
    let modq = Modulus::new(65537);

    // value on which range_fn is evaluated
    let v = 28192;
    let single = powers_of_x_modulus(v, &modq);
    let double = powers_of_x_modulus(single[255], &modq);

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
