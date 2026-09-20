# fARM64 roadmap

fARM64 is a `no_std`, zero-heap AArch64 disassembler and semantic encoder. This
roadmap records current public priorities; completed development-session logs
and local-machine measurements belong in version control history, not in the
published crate documentation.

## Delivered for 0.1.0

- Implicit register reads/writes: the architectural state an instruction
  touches without naming it in an operand (link register, pointer-
  authentication modifiers, the FEAT_LS64 eight-register group, the SVE `FFR`,
  `PC`, and `NZCV`), reported allocation-free and merged into
  `InstructionInfo::used_registers()`.
- Four implicit-state pseudo-registers (`Nzcv`, `Ffr`, `Za`, `Pc`) in a new
  `RegClass::Special`, never produced by the decoder or rendered by a formatter.
- In-place instruction editing, so a decoded `Instruction` can be re-encoded
  with different registers, immediates, memory operands, conditions, branch
  targets, encodings, or address. Every setter is total.
- A feature-gated test suite: `cargo test` now passes for every combination of
  Cargo features rather than only `--all-features`.

## Delivered for 0.0.2

- Complete public `Code::values()` and `Mnemonic::values()` enumeration in
  stable discriminant order, generated from the authoritative declarations.
- Safe, constant-time `from_u16()` and iced-compatible `TryFrom<usize>`
  conversion without weakening the crate-wide `forbid(unsafe_code)` policy.
- Complete enum metadata tests replacing the former manually maintained,
  partial encoding catalog.

## Delivered for 0.0.1

- A borrowing `Decoder` and `Copy` `Instruction` model for fixed-width A64.
- Hand-written Arm architectural decode paths, default UAL formatting, and a
  semantic single-instruction encoder.
- Base A64, floating-point/Advanced SIMD, SVE/SVE2, SME/SME2, crypto, and a broad
  set of later architectural extensions.
- Fine-grained runtime admission through `FeatureSet`, plus compile-out gates
  for the large `sve`, `sme`, and Advanced SIMD `crypto` decoder modules.
- Zero-allocation register/memory/flow analysis through `instruction_info` and
  an `alloc`-gated reusable `InstructionInfoFactory`.
- A fixed-buffer formatter sink, owned-string conveniences behind `alloc`, and
  `std::error::Error` implementations behind `std`.
- Separately gated Apple AMX and GXF support based on public reverse-engineering
  references; see [extension status](./REMAINING_EXTENSIONS.md) and `NOTICE`.

## Release gates

The release workflow should keep these checks green:

- `cargo fmt --all -- --check` and strict Clippy for the workspace and feature matrix.
- Fast unit and integration tests, including the allocation-counting audit of
  the advertised zero-heap decode/default-format path.
- Rustdoc with warnings denied and documentation examples compiled.
- The declared minimum supported Rust version and the current stable toolchain.
- Default-feature `wasm32-unknown-unknown`, `aarch64-unknown-none`, and hosted builds.
- `cargo package`/`cargo publish --dry-run`, followed by tests against the
  extracted package so excluded external corpora are not accidentally required.

External corpus and LLVM sweeps remain optional development checks because
their inputs/toolchains are not distributed with the crate. Their results must
not be represented as timeless coverage percentages.

## Near-term priorities

1. Keep the crates.io package small, reproducible, and independent of local
   reference material.
2. Add targeted conformance tests whenever an Arm revision or an implementation-
   defined encoding is added; cover reserved neighbors and runtime gating as
   well as successful decode/encode.
3. Give `GnuFormatter` a verified GNU/objdump-specific policy. Until then it is
   documented as UAL-equivalent.
4. Expand fuzz/property testing of decoder totality and semantic round trips.
5. Benchmark decode, encode, formatting, and `instruction_info` without turning
   benchmark-only dependencies into runtime dependencies.
6. Evaluate newly published Arm extensions, including LS64WB, without promising
   support before focused tests and documentation land.

## Stable project constraints

- The default crate remains `#![no_std]` and does not require `alloc`.
- Decode and default fixed-buffer formatting remain zero-heap.
- Public `#[non_exhaustive]` enum discriminants are append-only.
- The library has no runtime dependencies, build script, or network access.
- Cargo compile-out features and runtime `FeatureSet` admission remain distinct:
  not every architectural feature should become a Cargo feature.
- Arm architectural behavior follows the Arm Architecture Reference Manual.
  Differential tools and corpora are cross-checks. Apple implementation-defined
  support is identified separately with its public reverse-engineering sources.
