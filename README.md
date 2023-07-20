# Oblivious Message Retrieval (OMR)

This repository contains an efficient implementation of [Oblivious Message Retrieval](https://eprint.iacr.org/2021/1256.pdf).

## What's OMR?

Asynchronous messaging without leaking metadata is hard and every existing way to send message asynchronously requires metadata. The reason for this is obvious. To send a message to your friend asynchronously you atleast need to tag your message with your friend's identity. This enables the server to notify your friend of your message when they come online. Thus every message requires a "to" field. OMR is a way to send message asynchronously without revealing the "to" field. This means using OMR the server can deliver messages intended for your friend when they come online without ever learning which of the messages from the set of all messages were intended for them.

## Where's OMR useful?

Other than being useful for asynchronous messaging without leaking metadata, OMR can be useful for improving the user experience of privacy preserving blockchains.

In privacy preserving blockchains like Aztec/Zcash all transactions are encrypted. At present the only way for users to find their pertaining transactions is to download entire set (or a representation) of transactions. Then trial decrypt them to find their pertaining ones. The problem with trial decryption is obvious. Client's experience of using the chain degrades as the chain gains traction. This is because client's cost of trial decryption, in terms of bandwidth and local computation, increases linearly with no. of transactions sent on chain globally. With OMR the processing of transactions (ie messages) is offloaded to the server. The client simply comes online, requests for their encrypted digest (containing _only_ their transactions) from server, then decrypts the digest to find their transactions. With OMR client only needs to download 800Kb of data and perfom single decryption locally (decryption time: ~50ms). Hence, the entire process of retrieving transactions takes only few seconds and vastly improves user experience.

## Implementation details

For security and efficiency, OMR processes messages in set of $2^{15}$. Thus it is more natural to divide OMR into two phases. The first phase, phase 1, happens on server and does not require any client interaction. The server collects all messages and divides them in batches of $2^{15}$. Then proceeds to process each batch for each client. The second phase, phase 2, starts when client requests for their encrypted message digest. Note that the client can request for messages over any arbitrary range and the restriction of batching in sets of $2^{15}$ does not holds here.

## So where are we?

For users to be able to receive messages as soon as they are sent to server and for an amazing user experience of privacy preserving protocols it is important for runtime of Phase 1 and Phase 2 to be as low as possible. The good news is even though FHE computations are expensive, they are highly parallelizable. For example, as visible in benchmarks, on single thread phase 1 roughly takes 5.2 minutes and phase 2 roughly takes 14 minutes. But increasing threads to 16, reduces phase 1 time to 52.2 seconds and phase 2 time to 58 seconds.

Now if you are wondering whether time reduces linearly with more cores, yes it does. Phase 2 time reduces linearly with more cores. So does Phase 1, but with a caveat of some precomputation required per user that needs to be done only once and stored.

## How to run

Before running the program there are a few things to be kept in mind

1. The code is only optimised to use power of 2 no. of cores. So running it on a machine with non power of 2 cores wouldn't use all cores maximally.
2. The library has 3 features, `noise`, `level`, and `precomp_pvw`.
    - `noise`: Only used for debugging purposes. Not relevant for testing performance.
    - `level`: Enables levelled implementation. This should be enabled when testing performance.
    - `precomp_pvw`: Combining `level` with `precomp_pvw` will give the best performance if there are more than 8 no. of cores available. However, `precomp_pvw` comes with additonal assumptions mentioned in [PVW Precompute](#pvw-precompute).
3. The program performs best on x86 machines with AVX512IFMA instruction set.
4. OMR has high memory requirements. For example, with 16 cores phase 1 consumes around 20GB and phase 2 consumes around 100GB.
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
    - (3) prints the precomputed data size required per user for PVW precompute for a given no. of cores.

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

To print the precomputed data size required per user for PVW precompute for a given no. of `cores`

```
cargo run --release --features "level" 3 [cores]
```

## PVW Precompute

You can enable PVW precompute by enabling `precomp_pvw` feature. Combining `level` with `precomp_vw` feature provides best performance. However, enabling PVW precompute assumes that you are willing to store some amount of additonal precomputed data per user. This precomputation only needs to be performed once and can be stored throughout user's lifetime.

If PVW precompute is not enabled then phase 1 is bottlenecked by `pvw_decrypt` part, which can only use maximum of 4 cores at once. Thus, to scale performance of phase 1 with more no. of cores PVW precompute becomes essential.

You can get an estimate of data required to store per user for PVW precompute for a given number of cores by running:

```
cargo run --release --features "level" 3 [cores]
```

## Performance

All benchmarks were performed on `r6i.8xlarge` ec2 instance equipped with 3rd Gen Intel Xeon and 32vcpus (16 physical cores).

Single run of phase 1 and phase 2 processes $2^{15}$ messages.
<br></br>
**Performance with only `level` feature:**

| Cores        | Phase 1 time (in ms) | Phase 2 time (in ms) | Total time (in ms) |
| ------------ | -------------------- | -------------------- | ------------------ |
| 1            | 313680               | 844228               | 1157908            |
| 16 (32vcpus) | 52127                | 58095                | 110222             |

<br></br>
**Performance with `level` and `precomp_pvw` features.**
(Note: cores = 1 is omitted since `precomp_pvw` only works with >=8 cores)
| Cores | Phase 1 time (in ms) | Phase 2 time (in ms) | Total time (in ms) |
| ------------ | -------------------- | -------------------- | ------------------ |
| 16 (32vcpus) | 28202 | 57930 | 84226

Storage size for PVW Precompute per user on 16 cores: 47.5 MB

## Detection Key Size

Each user needs to upload detection key to server. The key does not reveal anything. It's size is **163 MB**.

## Message Digest Size

The size of message digest that user downloads at the time of request is **800Kb**.

## Security

Please [check](./Security.md)

## Contribution

If you want to work togther on OMR (and other FHE related projects) then please check [contact](#contact) and send a dm! Also feel free to join [this](https://t.me/+rDHqU-Py34s4N2M1) telegram channel.

## Use in production

If you want run OMR in production, then please check [contact](#contact) and reach out directly!

## Contact

Telegram: @janmajayamall <br />
Email: janmajaya@caird.xyz

## Acknowledgements

Development of OMR is supported through a grant from [Aztec](https://aztec.network/).
