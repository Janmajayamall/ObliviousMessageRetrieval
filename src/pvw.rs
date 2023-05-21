use bfv::Modulus;
use itertools::{izip, Itertools};
use ndarray::{Array, Array1, Array2, Axis};
use rand::{
    distributions::{Distribution, Uniform},
    thread_rng, CryptoRng, RngCore, SeedableRng,
};
use rand_chacha::ChaChaRng;
use statrs::distribution::Normal;
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub struct PvwParameters {
    pub n: usize,
    pub m: usize,
    pub ell: usize,
    pub variance: f64,
    pub q: u64,
}

impl Default for PvwParameters {
    fn default() -> Self {
        Self {
            n: 450,
            m: 16000,
            ell: 4,
            variance: 1.3,
            q: 65537,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PvwCiphertext {
    par: Arc<PvwParameters>,
    pub a: Vec<u64>,
    pub b: Vec<u64>,
}

// impl PvwCiphertext {
//     pub fn to_bytes(&self) -> Vec<u8> {
//         let proto = PvwCiphertextProto::from(self);
//         proto.write_to_bytes().unwrap()
//     }

//     pub fn from_bytes(bytes: &[u8], par: &Arc<PvwParameters>) -> Option<PvwCiphertext> {
//         let from = PvwCiphertextProto::parse_from_bytes(bytes).ok()?;
//         let p_bits = (64 - (par.q - 1).leading_zeros()) as usize;
//         let v = transcode_from_bytes(&from.v, p_bits);
//         let b = v[..par.ell].to_vec();
//         let a = v[par.ell..(par.ell + par.n)].to_vec();

//         Some(PvwCiphertext {
//             par: par.clone(),
//             a,
//             b,
//         })
//     }
// }

// impl From<&PvwCiphertext> for PvwCiphertextProto {
//     fn from(value: &PvwCiphertext) -> Self {
//         let mut proto = PvwCiphertextProto::new();
//         let p_bits = (64 - (value.par.q - 1).leading_zeros()) as usize;
//         let v = [value.b.clone(), value.a.clone()].concat();
//         proto.v = transcode_to_bytes(&v, p_bits);
//         proto
//     }
// }

pub struct PvwPublicKey {
    a: Array2<u64>,
    b: Array2<u64>,
    seed: <ChaChaRng as SeedableRng>::Seed,
    par: Arc<PvwParameters>,
}

impl PvwPublicKey {
    pub fn encrypt<R: RngCore + CryptoRng>(&self, m: &[u64], rng: &mut R) -> PvwCiphertext {
        debug_assert!(m.len() == self.par.ell);

        let error = Uniform::new(0u64, 2)
            .sample_iter(rng)
            .take(self.par.m)
            .collect_vec();

        let q = Modulus::new(self.par.q);
        let ae = Array1::from_vec(
            self.a
                .outer_iter()
                .map(|a_n_m| {
                    let mut r = a_n_m.to_vec();
                    q.mul_mod_fast_vec(&mut r, &error);
                    q.reduce(r.iter().sum::<u64>())
                })
                .collect(),
        );

        let t = m.iter().map(|v| {
            if *v == 1 {
                q.reduce((3 * self.par.q) / 4)
            } else {
                q.reduce(self.par.q / 4)
            }
        });
        let be = Array1::from_vec(
            izip!(self.b.outer_iter(), t)
                .map(|(b_ell_m, ti)| {
                    let mut r = b_ell_m.to_vec();
                    q.mul_mod_fast_vec(&mut r, &error);
                    q.add_mod_fast(q.reduce(r.iter().sum::<u64>()), ti)
                })
                .collect(),
        );

        PvwCiphertext {
            par: self.par.clone(),
            a: ae.to_vec(),
            b: be.to_vec(),
        }
    }

    // pub fn to_bytes(&self) -> Vec<u8> {
    //     let proto = PvwPublicKeyProto::from(self);
    //     proto.write_to_bytes().unwrap()
    // }

    // pub fn from_bytes(bytes: &[u8], par: &Arc<PvwParameters>) -> PvwPublicKey {
    //     let from = PvwPublicKeyProto::parse_from_bytes(bytes).unwrap();

    //     let seed = <ChaChaRng as SeedableRng>::Seed::try_from(from.seed).unwrap();
    //     let mut rng = ChaChaRng::from_seed(seed);
    //     let a = Modulus::new(par.q)
    //         .unwrap()
    //         .random_vec(par.n * par.m, &mut rng);

    //     let p_bits = 64 - (par.q - 1).leading_zeros() as usize;
    //     let mut b = transcode_from_bytes(&from.b, p_bits);
    //     b.truncate(par.ell * par.m);

    //     PvwPublicKey {
    //         a: Array::from_shape_vec((par.n, par.m), a).unwrap(),
    //         b: Array::from_shape_vec((par.ell, par.m), b).unwrap(),
    //         par: par.clone(),
    //         seed,
    //     }
    // }
}

// impl From<&PvwPublicKey> for PvwPublicKeyProto {
//     fn from(value: &PvwPublicKey) -> Self {
//         let mut proto = PvwPublicKeyProto::new();
//         let p_bits = 64 - (value.par.q - 1).leading_zeros() as usize;
//         proto.seed = value.seed.to_vec();
//         proto.b = transcode_to_bytes(
//             value
//                 .b
//                 .outer_iter()
//                 .flat_map(|ell_m| ell_m.to_vec())
//                 .collect_vec()
//                 .as_slice(),
//             p_bits,
//         );
//         proto
//     }
// }

pub struct PvwSecretKey {
    pub key: Array2<u64>,
    pub par: Arc<PvwParameters>,
}

impl PvwSecretKey {
    pub fn random<R: RngCore + CryptoRng>(
        params: &Arc<PvwParameters>,
        rng: &mut R,
    ) -> PvwSecretKey {
        let q = Modulus::new(params.q);

        let sk = Array::from_shape_vec(
            (params.ell, params.n),
            q.random_vec(params.n * params.ell, rng),
        )
        .unwrap();

        PvwSecretKey {
            key: sk,
            par: params.clone(),
        }
    }

    pub fn public_key<R: RngCore + CryptoRng>(&self, rng: &mut R) -> PvwPublicKey {
        let q = Modulus::new(self.par.q);

        let mut seed = <ChaChaRng as SeedableRng>::Seed::default();
        thread_rng().fill_bytes(&mut seed);
        let mut rng2 = ChaChaRng::from_seed(seed);

        let a = Array::from_shape_vec(
            (self.par.n, self.par.m),
            q.random_vec(self.par.n * self.par.m, &mut rng2),
        )
        .unwrap();

        // sk * a;
        let distr = Normal::new(0.0, self.par.variance).unwrap();
        let error = Array::from_shape_vec(
            (self.par.ell, self.par.m),
            q.reduce_vec_i64_small(
                &distr
                    .sample_iter(rng)
                    .take(self.par.ell * self.par.m)
                    .map(|v| v.round() as i64)
                    .collect_vec(),
            ),
        )
        .unwrap();

        let mut ska = izip!(self.key.outer_iter(), error.outer_iter())
            .flat_map(|(key_ell_n, e_ell_m)| {
                let key_ell_n = key_ell_n.as_slice().unwrap();
                let ska_ell_m = izip!(a.axis_iter(Axis(1)), e_ell_m.iter())
                    .map(|(m_n, e_value)| {
                        let mut r = m_n.to_vec();
                        q.mul_mod_fast_vec(&mut r, key_ell_n);
                        let r = (r.iter().sum::<u64>()) + *e_value;
                        r
                    })
                    .collect_vec();
                ska_ell_m
            })
            .collect_vec();
        q.reduce_vec(&mut ska);
        let ska = Array::from_shape_vec((self.par.ell, self.par.m), ska).unwrap();

        PvwPublicKey {
            a,
            b: ska,
            par: self.par.clone(),
            seed,
        }
    }

    pub fn decrypt(&self, ct: PvwCiphertext) -> Vec<u64> {
        let q = Modulus::new(self.par.q);

        izip!(ct.b.iter(), self.key.outer_iter())
            .map(|(b_ell, k_ell_n)| {
                let mut r = ct.a.clone();
                q.mul_mod_fast_vec(&mut r, &k_ell_n.to_vec());
                let d = q.sub_mod_fast(*b_ell, q.reduce(r.iter().sum::<u64>()));
                (d >= self.par.q / 2) as u64
            })
            .collect()
    }

    pub fn decrypt_shifted(&self, ct: PvwCiphertext) -> Vec<u64> {
        let q = Modulus::new(self.par.q);

        izip!(ct.b.iter(), self.key.outer_iter())
            .map(|(b_ell, k_ell_n)| {
                let mut r = ct.a.clone();
                q.mul_mod_fast_vec(&mut r, &k_ell_n.to_vec());

                // shift value left by q/4 so that
                // indices encrypting 0 are near value 0.
                let d = q.sub_mod_fast(
                    q.sub_mod_fast(*b_ell, q.reduce(r.iter().sum::<u64>())),
                    self.par.q / 4,
                );

                // Now values encrypting zero should be in range
                // q - 850 < v < 850 with high probability
                !(self.par.q - 850 <= d || d <= 850) as u64
            })
            .collect()
    }

    // pub fn to_bytes(&self) -> Vec<u8> {
    //     let proto = PvwSecretKeyProto::from(self);
    //     proto.write_to_bytes().unwrap()
    // }

    // pub fn from_bytes(bytes: &[u8], par: &Arc<PvwParameters>) -> PvwSecretKey {
    //     let from = PvwSecretKeyProto::parse_from_bytes(bytes).unwrap();
    //     let p_bits = 64 - (par.q - 1).leading_zeros() as usize;
    //     let mut key = transcode_from_bytes(&from.key, p_bits);
    //     key.truncate(par.ell * par.n);
    //     PvwSecretKey {
    //         key: Array::from_shape_vec((par.ell, par.n), key).unwrap(),
    //         par: par.clone(),
    //     }
    // }
}

// impl From<&PvwSecretKey> for PvwSecretKeyProto {
//     fn from(value: &PvwSecretKey) -> Self {
//         let mut proto = PvwSecretKeyProto::new();
//         let p_bits = 64 - (value.par.q - 1).leading_zeros() as usize;
//         proto.key = transcode_to_bytes(
//             value
//                 .key
//                 .outer_iter()
//                 .flat_map(|ell_i| ell_i.to_vec())
//                 .collect_vec()
//                 .as_slice(),
//             p_bits,
//         );

//         proto
//     }
// }

#[cfg(test)]
mod tests {

    use super::*;
    use rand::thread_rng;

    #[test]
    fn encrypt() {
        let mut rng = thread_rng();
        let params = Arc::new(PvwParameters::default());
        for _ in 0..10 {
            let sk = PvwSecretKey::random(&params, &mut rng);
            let pk = sk.public_key(&mut rng);

            let distr = Uniform::new(0u64, 2);
            let m = distr
                .sample_iter(rng.clone())
                .take(params.ell)
                .collect_vec();
            let ct = pk.encrypt(&m, &mut rng);
            dbg!(ct.a.len(), ct.b.len());

            let d_m = sk.decrypt_shifted(ct);

            assert_eq!(m, d_m)
        }
    }

    #[test]
    fn check_probs() {
        let params = Arc::new(PvwParameters::default());

        let mut rng = thread_rng();
        let sk = PvwSecretKey::random(&params, &mut rng);
        let pk = sk.public_key(&mut rng);

        let sk1 = PvwSecretKey::random(&params, &mut rng);
        let pk1 = sk1.public_key(&mut rng);

        let mut count = 0;
        let mut count1 = 0;
        let observations = 1000;
        for _ in 0..observations {
            let ct = pk.encrypt(&[0, 0, 0, 0], &mut rng);
            let ct1 = pk1.encrypt(&[0, 0, 0, 0], &mut rng);

            if sk.decrypt_shifted(ct) == vec![0, 0, 0, 0] {
                count += 1;
            }

            if sk.decrypt_shifted(ct1) == vec![0, 0, 0, 0] {
                count1 += 1;
            }
        }
        assert!((count as f64 / observations as f64) == 1.0);
        assert!((count1 as f64 / observations as f64) == 0.0);
    }
}
