use bfv::{
    BfvParameters, Ciphertext, Encoding, Evaluator, Modulus, PolyContext, RelinearizationKey,
    SecretKey, TryFromWithParameters,
};
use byteorder::{ByteOrder, LittleEndian};
use bytes::Bytes;
use itertools::Itertools;
use ndarray::Array2;
use prost::Message;
use rand::{
    distributions::{Standard, Uniform},
    thread_rng, Rng,
};
use rayon::{
    prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelBridge, ParallelIterator},
    vec,
};
use std::{
    io::{BufReader, Read, Write},
    path::{Path, PathBuf},
};
use walkdir::{DirEntry, WalkDir};

use crate::{
    pvw::{PvwCiphertext, PvwCiphertextProto, PvwParameters, PvwPublicKey, PvwSecretKey},
    MESSAGE_BYTES,
};

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

#[macro_export]
macro_rules! print_noise {
    ($($block:tt)+) => {
        #[cfg(feature="noise")]
        {
            $(
                $block
            )+
        }
    };
}

#[macro_export]
macro_rules! level_down {
    ($($block:tt)+) => {
        #[cfg(feature="level")]
        {
            $(
                $block
            )+
        }
    };
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
    params.enable_hybrid_key_switching(&[50, 50, 60]);
    params
}

/// Generates random clues using a single public key.
fn generate_random_clues(pvw_params: &PvwParameters, set_size: usize) -> Vec<PvwCiphertext> {
    let mut rng = thread_rng();
    let random_sk = PvwSecretKey::random(pvw_params, &mut rng);
    let pk = random_sk.public_key(&mut rng);

    let mut clues = vec![];
    (0..set_size)
        .into_par_iter()
        .map(|_| {
            let mut rng = thread_rng();
            pk.encrypt(&[0, 0, 0, 0], &mut rng)
        })
        .collect_into_vec(&mut clues);

    clues
}

fn store_clues(clues: &[PvwCiphertext], params: &PvwParameters) {
    let output_dir = Path::new("./data/clues");
    std::fs::create_dir_all("./data/clues").expect("Create clue directory failed");

    clues.iter().enumerate().for_each(|(index, c)| {
        let bytes = PvwCiphertextProto::try_from_with_parameters(c, params).encode_to_vec();
        let mut clue_path = PathBuf::from(output_dir);
        clue_path.push(format!("{index}.bin"));

        let mut f = std::fs::File::create(clue_path).expect("Couldn't create clue file");
        f.write_all(&bytes)
            .expect("Couldn't write bytes to clue file");
    });
}

fn read_clues(params: &PvwParameters) -> Vec<PvwCiphertext> {
    let paths = std::fs::read_dir("./data/clues").expect("data/clue directory not found");
    let mut cts = vec![];
    paths.into_iter().for_each(|path| {
        let path = path.unwrap();

        if path.file_type().unwrap().is_file() {
            let file = std::fs::read(path.path())
                .expect(&format!("Unable to open file at {:?}", path.path()));
            let bytes = Bytes::from(file);
            let proto = PvwCiphertextProto::decode(bytes)
                .expect(&format!("Invalid clue file {:?}", path.path()));
            let ct = PvwCiphertext::try_from_with_parameters(&proto, params);
            cts.push(ct);
        }
    });
    cts
}

fn is_bin(entry: &Result<walkdir::DirEntry, walkdir::Error>) -> bool {
    entry.as_ref().map_or(false, |c| {
        c.file_name()
            .to_str()
            .map(|s| s.contains(".bin"))
            .unwrap_or(false)
    })
}

/// If random clues have been generated and stored under ./data/clues then the function
/// reads them, otherwise generates new random clues and store them under ./data/clues
/// taking significantly longer time. Once random clues have either been read or generated
/// the function generates new pertinent clues for `pk` and places them at specific indices
/// in `clues` vector according to `pertinency_indices`.
pub fn prepare_clues_for_demo(
    pvw_params: &PvwParameters,
    pk: &PvwPublicKey,
    pertinent_indices: &[usize],
    set_size: usize,
) -> Vec<PvwCiphertext> {
    let clue_dir = Path::new("./data/clues");
    // std::fs::read_dir(clue_dir).expect(&format!("Cannot open {}", clue_dir.to_str().unwrap())).si;
    let mut clues = vec![];

    if WalkDir::new(clue_dir).into_iter().filter(is_bin).count() < set_size {
        println!("/data/clues not found. Generating random clues...");
        time_it!("Generate random clues",
        clues = generate_random_clues(pvw_params, set_size););
        // store clues for later
        store_clues(&clues, pvw_params);
    } else {
        println!("Reading clues stored under /data/clues...");
        clues = read_clues(pvw_params);
        clues.truncate(set_size);
    }

    // generate pertinent clues and place them at pertinent indices
    println!("Generating pertinent clues...");
    let mut rng = thread_rng();
    pertinent_indices.iter().for_each(|i| {
        clues[*i] = pk.encrypt(&[0, 0, 0, 0], &mut rng);
    });

    clues
}

/// All non pertinent clues are clone of a single clues generated under a random public key.
/// All pertinent clues are generated fresh using `pvw_pk`. Only used for testing purposes.
fn generate_clues_fast(
    pvw_pk: &PvwPublicKey,
    pvw_params: &PvwParameters,
    pertinent_indices: &[usize],
    count: usize,
) -> Vec<PvwCiphertext> {
    let mut rng = thread_rng();

    let other_pvw_sk = PvwSecretKey::random(pvw_params, &mut rng);
    let other_pvw_pk = other_pvw_sk.public_key(&mut rng);
    let non_peritnent_clue = other_pvw_pk.encrypt(&[0, 0, 0, 0], &mut rng);

    // generate hints
    let partinent_clue = pvw_pk.encrypt(&[0, 0, 0, 0], &mut rng);
    let clues = (0..count)
        .into_iter()
        .map(|i| {
            if pertinent_indices.contains(&i) {
                partinent_clue.clone()
            } else {
                non_peritnent_clue.clone()
            }
        })
        .collect_vec();

    clues
}

mod tests {
    use std::collections::HashSet;

    use bfv::Modulus;

    use super::*;

    #[test]
    fn test_store_range_coeffs() {
        store_range_coeffs();
    }

    #[test]
    fn generate_and_store_random_clues() {
        let pvw_params = PvwParameters::default();
        let mut clues = generate_random_clues(&pvw_params, 2);
        store_clues(&clues, &pvw_params);
        let mut clues_back = read_clues(&pvw_params);
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
