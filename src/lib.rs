pub mod client;
pub mod optimised;
pub mod plaintext;
pub mod preprocessing;
pub mod pvw;
pub mod server;
pub mod utils;

// Fixing a few constants here. They can modified
// later or probably varied across runs
const GAMMA: usize = 5;
const MESSAGE_BYTES: usize = 512;
const K: usize = 64; // 64*2*256 = 32768
const BUCKET_SIZE: usize = 32768 / MESSAGE_BYTES / 2; // since each lane fits 2 bytes
