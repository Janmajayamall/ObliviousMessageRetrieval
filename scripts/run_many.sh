#!/usr/bin/env bash
export RUSTFLAGS=-Awarnings
for x in {0..15} ; do
    cargo run --release --features "level" 1
done