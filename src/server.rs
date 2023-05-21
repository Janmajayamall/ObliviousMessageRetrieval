use crate::pvw::{self, PvwCiphertext, PvwParameters};
use bfv::{
    BfvParameters, Ciphertext, Encoding, GaloisKey, Modulus, Plaintext, Poly, RelinearizationKey,
    Representation, SecretKey,
};
use itertools::izip;
use ndarray::{azip, Array2, IntoNdProducer};
use rand::thread_rng;
use std::{hint, sync::Arc, task::Poll};

pub fn pre_process_batch(
    pvw_params: &Arc<PvwParameters>,
    bfv_params: Arc<BfvParameters>,
    hints: &[PvwCiphertext],
) -> Vec<Plaintext> {
    // can only process as many as polynomial_degree hints in a batch
    debug_assert!(hints.len() <= bfv_params.polynomial_degree);

    let sec_len = pvw_params.n.next_power_of_two();
    let mut plaintexts = vec![];
    for i in 0..sec_len {
        let mut m = vec![];
        for j in 0..hints.len() {
            let index = (j + i) % sec_len;
            if index < pvw_params.n {
                m.push(hints[j].a[index]);
            } else {
                m.push(0);
            }
        }
        plaintexts.push(Plaintext::encode(&m, &bfv_params, Encoding::simd(0)));
    }

    // length of plaintexts will be sec_len
    plaintexts
}

// rotate by 1 and perform plaintext mutiplication for each ell
pub fn pvw_decrypt(
    pvw_params: &Arc<PvwParameters>,
    hint_pts: &[Plaintext],
    pvw_sk_cts: &[Ciphertext],
    rtk: &GaloisKey,
) {
    let sec_len = pvw_params.n.next_power_of_two();
    debug_assert!(hint_pts.len() == sec_len);
    debug_assert!(pvw_sk_cts.len() == pvw_params.ell);

    // d[j] = s[j][0] * p[0] + s[j][1] * p[1] + ... + s[j][sec_len-1] * p[sec_len-1]
    // where s[j][a] is s[j] rotated to left by 1 `a` times.
    // Each operation is further broken down to: d[j] += s[j][0] * p[0]. Can we take
    // advantage of fused multiplication addition to speed this up? For ex, hexl has
    // an API for FMA which is faster (should be right?) than perfoming vector multiplication
    // and addition in a sequence.
    // There's an additinal optimisation for FMA operation. We can perform FMA in 128 bits without
    // modulur reduction followed by 128 bit barret reduction in the end. Since we will only be adding 512 128 bits values,
    // result will not overflow. Amazing!
    // TODO: Provide and API for FMA (ie Ct + Ct * Pt) in Bfv.
    //
    // Length of `d == ell`.
    // let mut d = vec![];
    // for i in 0..sec_len {
    //     for j in 0..pvw_params.ell {
    //         // multiply sk[j] * hints_pts and add it to d[j]
    //         pvw_sk_cts[j].multiply1(rhs)
    //     }
    // }
}

pub fn powers_of_x_ct(x: &Ciphertext, rlk: &RelinearizationKey) -> Vec<Ciphertext> {
    let mut values = vec![Ciphertext::zero(&x.params(), x.level()); 256];
    let mut calculated = vec![0u64; 256];
    let mut mul_count = 0;
    for i in (0..257).rev() {
        let mut exp = i;
        let mut base = x.clone();
        let mut res = Ciphertext::zero(&x.params(), x.level());
        let mut base_deg = 1;
        let mut res_deg = 0;

        while exp > 0 {
            if exp & 1 == 1 {
                res_deg += base_deg;
                // Covers the case when res is Ciphertext::zero
                if res_deg == base_deg {
                    res = base.clone();
                } else if calculated[res_deg - 1] == 1 {
                    res = values[res_deg - 1].clone();
                } else {
                    mul_count += 1;
                    let tmp = res.multiply1(&base);
                    res = rlk.relinearize(&tmp);
                    values[res_deg - 1] = res.clone();
                    calculated[res_deg - 1] = 1;
                }
            }
            exp >>= 1;
            if exp != 0 {
                base_deg *= 2;

                if calculated[base_deg - 1] == 1 {
                    base = values[base_deg - 1].clone();
                } else {
                    mul_count += 1;
                    let tmp = base.multiply1(&base);
                    values[base_deg - 1] = rlk.relinearize(&tmp);
                    base = values[base_deg - 1].clone();
                    calculated[base_deg - 1] = 1;
                }
            }
        }
    }
    dbg!(mul_count);
    values
}
