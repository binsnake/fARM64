# Architecture and design

fARM64 is a pure-Rust, `#![no_std]` AArch64 (A64) disassembler and semantic
encoder. It is organized around a borrowing decoder, a self-contained `Copy`
instruction value, push-based formatting, and fixed-capacity access analysis.

Companion documents:

- [API.md](./API.md) describes the public types and examples.
- [ENCODING.md](./ENCODING.md) describes group dispatch, fields, aliases, and
  Arm pseudocode helpers.
- [VALIDATION.md](./VALIDATION.md) gives reproducible release and differential
  test procedures.
- [REMAINING_EXTENSIONS.md](./REMAINING_EXTENSIONS.md) records current extension
  status and known limitations.
- [ROADMAP.md](./ROADMAP.md) lists future priorities.

## Core data flow

An A64 instruction is one little-endian 32-bit word. `Decoder` reads four bytes,
records their address, and dispatches the word to `decode::decode_into`. The
decode tree selects an encoding group from bits 28:25, validates subordinate
fields and feature requirements, and fills a caller-owned `Instruction`.

```text
&[u8] + address
      │
      ▼
   Decoder ── u32 + FeatureSet ──► decode group tree
                                      │
                                      ▼
                                Copy Instruction
                                  │     │     │
                                  ▼     ▼     ▼
                              formatter info encoder
```

`Instruction` stores `Code`, resolved `Mnemonic`, operands, address, raw word,
and flow/flag metadata inline. It contains no pointers and is bounded by a
compile-time `size_of::<Instruction>() <= 112` assertion. A failed or
unallocated decode produces `Code::Invalid` and a typed `DecodeError`; the
cursor still follows A64's fixed four-byte width.

## Decode organization

`src/decode/mod.rs` performs top-level dispatch. The main implementations are:

- `dp_imm.rs` — data processing, immediate;
- `branch_sys.rs` — branches, exceptions, hints, and system instructions;
- `ldst.rs` and `ldst_simd.rs` — scalar, atomic, and SIMD loads/stores;
- `dp_reg.rs` — data processing, register;
- `simd_fp/` — scalar floating point, Advanced SIMD, and optional crypto decode;
- `sve/` and `sme/` — optional scalable vector/matrix trees;
- `apple.rs` — separately gated Apple implementation-defined AMX/GXF words;
- `bits.rs` — shared bit extraction and Arm pseudocode helpers.

Group decoders use ordinary Rust control flow and build the public value
directly. `Code` preserves the encoding identity while `Mnemonic` carries the
preferred or alias spelling. Alias conditions live beside the encoding logic;
formatting can request the canonical spelling where supported.

The semantic encoder in `src/encode/` dispatches on `Code` and reconstructs a
word from operands and address. It never uses `Instruction::word()` as its
answer. The contract is semantic round-trip: a successfully encoded result must
decode to equivalent instruction semantics. Architecturally ignored fields may
be canonicalized, so byte identity is not promised for every accepted input.

## Formatting

The `Formatter` trait pushes classified text into a `FormatterOutput` sink.
`FmtFormatter` implements the default Arm UAL policy. `BufSink` writes to a
caller-provided byte slice and reports overflow without allocating. Any
`core::fmt::Write` can also act as a sink; owned `String` conveniences require
the `alloc` feature.

`GnuFormatter`, behind `fmt-gnu`, is currently a compatibility adapter that
delegates to `FmtFormatter` and emits identical UAL text. Its separate type is a
stable place to add and test GNU/objdump-specific rendering later.

## Instruction access analysis

`instruction_info(&Instruction)` is implemented on the default no-allocation
path. It returns fixed-capacity register and memory access lists, control-flow
classification, and flags-read/flags-written state. Analysis accounts for
explicit operands, memory base/index registers, writeback, common accumulator
destinations, and implicit link/flag effects. The `alloc`-gated
`InstructionInfoFactory` caches a result for repeated use.

## Compile-time and runtime features

Cargo and runtime features answer different questions:

| Layer | Purpose |
|-|-|
| Cargo `sve` | Compile the SVE/SVE2 decoder and encoder modules. |
| Cargo `sme` | Compile the SME/SME2 decoder and encoder modules. |
| Cargo `crypto` | Compile the Advanced SIMD crypto decoder. Public enum variants and encoder support remain available. |
| Cargo `full` | Enable `sve`, `sme`, and `crypto`. |
| Runtime `FeatureSet` | Admit or reject individual architectural and implementation-defined encodings. |

FP16, BF16, LSE, PAuth, MTE, and most other runtime `Feature` variants are
always compiled and have no same-named Cargo switch. Conversely, setting a
runtime bit cannot restore a module omitted by Cargo. `FeatureSet::BASE` admits
base encodings only; `FeatureSet::ALL` is the permissive default useful for
general disassembly and differential testing.

The public enums remain stable across Cargo configurations. They are
`#[non_exhaustive]`, and discriminants follow an append-only policy.

## Portability tiers

| Tier | Feature | Contents |
|-|-|-|
| A | default | `#![no_std]`, no `alloc`; decoder, encoder, UAL formatter, `BufSink`, enums, and `instruction_info` |
| B | `alloc` | owned-string/token conveniences and `InstructionInfoFactory` |
| C | `std` | implies `alloc`; standard error integration and test conveniences |

The default core depends only on `core`. Names are borrowed from static tables;
there is no build script, runtime dependency, filesystem/network access, or
allocator requirement. CI checks the declared Rust 1.74.1 MSRV, stable Rust,
`wasm32-unknown-unknown`, and `aarch64-unknown-none`, along with the supported
feature matrix.

The `no-alloc-audit` test feature enables an allocation-counting global
allocator. The audit measures allocation calls around decode and fixed-buffer
formatting and requires the count to remain zero.

## Source and validation posture

Arm architectural decode and encode behavior is implemented from the publicly
documented Arm Architecture Reference Manual. LLVM, GNU binutils, and the
Vector 35 `arch-arm64` corpus are optional development cross-checks; external
corpora are read locally and excluded from the published package.

Apple AMX and GXF are not Arm architectural extensions. AMX names and encodings
reference [`corsix/amx`](https://github.com/corsix/amx). GXF encodings reference
Asahi Linux's [Apple Proprietary Instructions](https://asahilinux.org/docs/hw/cpu/apple-instructions/)
documentation; Sven Peter's [GXF article](https://blog.svenpeter.dev/posts/m1_sprr_gxf/)
provides architectural background. Both families have
independent runtime feature bits and focused validity/gating/round-trip tests.

The crate is distributed under the MIT License. `NOTICE` records the development
references and implementation-defined sources without changing the license.

## Release invariants

- Default decode and `FmtFormatter`/`BufSink` formatting remain zero-heap.
- Malformed words yield typed invalid results rather than out-of-bounds access.
- Public discriminants are append-only.
- External corpora are never needed to build or test the published crate.
- Release documentation reports reproducible commands rather than timeless test
  counts or coverage percentages.
