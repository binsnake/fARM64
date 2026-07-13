# Validation

Correctness for Arm architectural instructions is defined against the Arm
Architecture Reference Manual (the "Arm ARM"). External disassemblers and
corpora are useful independent cross-checks, not the architectural authority.
Apple AMX/GXF are implementation-defined and are validated against their public
reverse-engineering references plus focused tests.

This document intentionally contains no fixed test count or corpus percentage.
Both change as encodings and external datasets evolve. Reproduce results from
the current checkout with the commands below.

## Required release checks

The normal release gate covers:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo test --features no-alloc-audit --test no_alloc_audit
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
cargo check -p fARM64 --target wasm32-unknown-unknown --no-default-features
cargo check -p fARM64 --target aarch64-unknown-none --no-default-features
cargo publish -p fARM64 --dry-run
```

CI additionally checks the exact declared MSRV (Rust 1.74.1), the supported
feature combinations, and tests from the extracted package. Package testing is
important because `refs/` and other local development inputs are deliberately
excluded from crates.io.

The allocation audit installs a counting global allocator, samples the count
around the decoder plus `FmtFormatter`/`BufSink` path, and asserts that it does
not increase. A passing audit demonstrates that the advertised default core
path does not allocate.

## Focused tests

Checked-in unit and integration tests cover representative valid words,
unallocated/reserved neighbors, runtime feature admission, formatting, and
semantic encoding. Extension changes should add cases in all applicable
categories.

The semantic encoder is tested as:

1. decode an input word to an `Instruction`;
2. encode from `Code`, operands, and instruction address (never the stored raw
   word);
3. decode the produced word; and
4. compare the resulting instruction semantics.

Semantic equality is the contract. Some encodings contain architecturally
ignored or constrained fields that the decoder does not retain, so the encoder
may emit a canonical word instead of reproducing every input bit. Examples
include ignored fields in multiply-high, exclusive load/store, instruction-cache
operations, some SIMD element forms, and floating-point compare-with-zero.

## Optional differential checks

The repository contains three ignored development sweeps:

| Test | Purpose | External requirement |
|-|-|-|
| `tests/golden.rs` | Compare decoded UAL text with a Vector 35 `arch-arm64` corpus | Local `test_cases.txt` |
| `tests/llvm_diff.rs` | Compare selected words with LLVM disassembly | `llvm-mc` on `PATH` or `FARM64_LLVM` |
| `tests/roundtrip.rs` | Run semantic encode/decode round trips over the corpus | Local `test_cases.txt` |

Run them explicitly after supplying the relevant input:

```sh
cargo test --features "std full" --test golden -- --ignored --nocapture
cargo test --features "std full" --test llvm_diff -- --ignored --nocapture
cargo test --features "std full" --test roundtrip -- --ignored --nocapture
```

`FARM64_CORPUS=<path>` overrides the corpus path. `FARM64_GROUP=<substring>`
filters corpus labels, `FARM64_LIMIT=<n>` caps a run, and
`FARM64_LLVM=<executable>` selects LLVM. The corpus parser streams cases rather
than loading the whole dataset.

These sweeps are optional because the external files and tools are not packaged.
The explicitly ignored golden and round-trip commands require their corpus;
the LLVM sweep skips when LLVM is absent. Ordinary package tests skip optional
corpus checks and do not depend on these inputs.

## Comparing formatter output

`tests/common/mod.rs` applies conservative normalization to oracle text: case,
whitespace, comment, separator, and bracket-padding differences are ignored.
It deliberately does not erase semantic differences by changing immediates,
register ranges, or alias choices.

When an oracle disagrees with fARM64, inspect the encoding against the Arm ARM
and at least one independent implementation where possible. Record focused
regression words in a checked-in test. Do not convert a one-machine differential
result into a permanent global coverage claim.

## Apple implementation-defined validation

Apple AMX operation naming and encodings reference
[`corsix/amx`](https://github.com/corsix/amx). GXF encodings reference Asahi
Linux's [encoding documentation](https://asahilinux.org/docs/hw/cpu/apple-instructions/)
and Sven Peter's [GXF article](https://blog.svenpeter.dev/posts/m1_sprr_gxf/).
`tests/apple_amx_gxf.rs` checks
decode/render behavior, independent runtime gates, semantic round trips, and
reserved forms. These families are identified separately from Arm architectural
extensions in both API documentation and `NOTICE`.
