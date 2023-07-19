#!/usr/bin/env bash

# clean
if [ -d "./data" ]; then
    rm -rf ./data
fi

export RUSTFLAGS=-Awarnings

cargo test --release utils::tests::test_store_range_coeffs
cargo test --release utils::tests::generate_and_store_random_clues
