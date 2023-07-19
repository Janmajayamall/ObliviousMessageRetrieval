Oblivious Message Retrieval (OMR)

This repository contains an efficient implementation of [Oblivious Message Retrieval](https://eprint.iacr.org/2021/1256.pdf).

In production OMR can be divided into 2 phases. Phase 1 regularly collects and batches all transactions in batch of size 32768, then processed each batch for each user. Phase 2 is the request phase, that is when users requests for their message digest over arbitrary range.

## How to run

Before running the program there are a few things to be kept in mind

1. The code is only optimised to use power of 2 no. of cores. So running it on a machine with non power of 2 cores wouldn't use all cores maximally.
2. The library has 3 features, `noise`, `level`, and `precomp_pvw`.
    - `noise`: Only used for debugging purposes. Not relevant for testing performance.
    - `level`: Enables levelled implementation. This should be enabled when testing performance.
    - `precomp_pvw`: Combination of `level` with `precomp_pvw` will give the best performance if there are more than 8 no. of cores available. However, `precomp_pvw` comes with additonal assumptions mentioned in [PVW Precompute](#pvw-precompute).
3. The program performs best on x86 machines with AVX512IFMA instruction set.
   <br></br>

First run setup

```
chmod +x setup.sh
./setup.sh
```

Basic structure of the command is

```
cargo run --release --features "[features]" [option] [cores]
```

1. `features` is a string consisting of enabled features
2. `option` can either be 1,2, or 3 and must be supplied

    - (1) run demo
    - (2) prints detection key size
    - (3) print the precomputed data size per user for PVW precompute for a given no. of cores.

3. `cores` is optional. If supplied it restricts the program to use given no. of cores. Otherwise program defaults to using all available cores.

To run demo with only `level` feature without PVW precompute

```
cargo run --release --features "level" 1
```

To restrict the demo to use a fixed no. of `cores`

```
cargo run --release --features "level" 1 [cores]
```

To run demo with `level` and `precomp_pvw` (best performance)

```
cargo run --release --features "level,precomp_pvw" 1
```

To print the detection key size

```
cargo run --release --features "level" 2
```

To print the precomputed data size per user for PVW precompute for a given no. of `cores`

```
cargo run --release --features "level" 3 [cores]
```

## PVW Precompute

You can enable PVW precompute by enabling `precomp_pvw` feature. Combination of `level` and `precomp_vw` feature provides best performance. However, enabling PVW Precompute assumes that you are willing to store some amount of additonal precomputed data per user. This precomputation only needs to be performed once and can be stored throughout user's lifetime.

If PVW precompute is not enabled then phase 1 is bottlenecked by pvw_decrypt part, which can only use maximum of 4 cores at once. Thus, to scale performance of phase 1 with more no. of cores PVW precompute becomes essential.

You can get an estimate of data required to store per user for PVW precompute for a given number of cores by running:

```
cargo run --release --features "level" 3 [cores]
```

## Performance

All benchmarks were performed on `r6i.8xlarge` ec2 instance equipped with 3rd Gen Intel Xeon and 32vcpus (16 physical cores).

Single run of phase 1 and phase 2 processes 32768 clues.

Performance with only `level` feature:

| Cores        | Phase 1 time (in ms) | Phase 2 time (in ms) | Total time (in ms) |
| ------------ | -------------------- | -------------------- | ------------------ |
| 1            | 313680               | 844228               | 1157908            |
| 16 (32vcpus) | Content Cell         |

Performance with `level` and `precomp_pvw` features.
(Note: cores = 1 is omitted since `precomp_pvw` only works with >=8 cores)
| Cores | Phase 1 time (in ms) | Phase 2 time (in ms) | Total time (in ms) |
| ------------ | -------------------- | -------------------- | ------------------ |
| 1 | 313680 | 844228 | 1157908 |
| 16 (32vcpus) | Content Cell |

## Security

Please [check](./Security.md)

## Contribution

If you want to work togther on OMR (and other FHE related projects) then please check [contact](#contact) and send a dm. Also feel free to join this telegram channel.

## Use in production

If you want test OMR in production, then please check [contact](#contact) and reach out directly!

## Contact

Telegram:
Email:

## Acknowledgements

Development of OMR is support through a grant from [Aztec](https://aztec.network/).
