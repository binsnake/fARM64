# fARM64 — Architecture & Design

> Pure-Rust, `no_std`, zero-heap AArch64 (A64) disassembler.
> Crate name: **`fARM64`** ("farm" + ARM64). The crate root carries
> `#![allow(non_snake_case)]` so the stylized name builds warning-free; the
> package and lib name are both `fARM64`, and imports read `use fARM64::...`.

**Companion documents**

- [API.md](./API.md) — the full public surface (types, signatures, usage examples).
- [ENCODING.md](./ENCODING.md) — the encoding taxonomy, the bit-field layouts per group, the ARM pseudocode helpers, and the alias / preferred-disassembly rules.
- [ROADMAP.md](./ROADMAP.md) — milestones (completed + remaining) and the per-extension rollout order.
- [VALIDATION.md](./VALIDATION.md) — the validation methodology and the measured corpus coverage / parity results, plus the documented spec-vs-binja divergence table.

**Status (measured).** 99.35% corpus coverage at 99.78% parity of attempted (42016/42289 attempted, 41925 matched). Base ISA + scalar-FP/Advanced-SIMD + SVE/SVE2 + partial SME + crypto are implemented; the remaining mismatches are verified-spec-correct binja rendering bugs (see [VALIDATION.md](./VALIDATION.md)).

This document is the architectural source of truth: *why* the crate is built the way it is, *how* a 32-bit word becomes an instruction, and *how* the pieces fit together. It is deliberately implementation-agnostic about line-level code; see the companions for the concrete API and encoding detail.

---

## 1. Project overview & goals

`fARM64` decodes and formats the AArch64 **A64** instruction set (fixed 32-bit, little-endian, 4-byte-aligned encodings). It targets three outcomes, in priority order.

### 1.1 Faithful disassembly (the correctness contract)

The definition of "correct" for this project is:

> **Matches the ARM Architecture Reference Manual (the "ARM ARM").** Every encoding, field derivation, alias condition, and preferred-disassembly rule is implemented from the published ARM specification.

Because no single oracle is infallible, correctness is *cross-checked* against several independent tools rather than defined by any one of them:

- **LLVM** — `llvm-mc -disassemble` / `llvm-objdump`.
- **GNU binutils** — `objdump -d`.
- **Binary Ninja's corpus** — `refs/arch-arm64-master/disassembler/test_cases.txt` is used as **one differential development guide**, read locally, never shipped, and **not authoritative**.

Where these oracles disagree, the ARM ARM is the tiebreaker. There is a known, load-bearing case where Binary Ninja *deliberately diverges* from the spec: in `refs/arch-arm64-master/disassembler/pcode.c:88`, `DecodeBitMasks` takes a shortcut and sets `tmask == wmask` under a `// TODO: do this right` comment. **fARM64 follows the spec**: it computes `tmask` correctly per the ARM ARM and records the intentional divergence on a documented allow-list in the differential harness (see [§9](#9-verification-posture-summary)). "Binja-correct" is never the target; "spec-correct, multiply cross-checked" is.

### 1.2 iced-x86-inspired ergonomics

The public surface mirrors the shape that made [`iced-x86`](https://github.com/icedland/iced) pleasant: a `Copy` value-type `Instruction`, a borrowing `Decoder` that is also an `Iterator`, a `decode_into` hot path, clean `Code` / `Mnemonic` / `Register` / `OpKind` enums, a rich `Operand` enum, a `Formatter` trait writing into a token sink, and an `InstructionInfo` flow/access facility. This public API is **strategy-independent** — it is the same regardless of how the decode core is built, and it does not change as a result of the hand-written pivot. The internal decode representation never reaches users — **the public API is a projection over an internal decode core.**

### 1.3 Base ISA first, extensions by gating

The base integer/branch/load-store/DP ISA is brought up first; FP/NEON, then FP16, LSE, PAuth, SME, MTE, BF16, SVE/SVE2, and crypto follow in ROI order. Each extension is a *gated region of the hand-written decode tree* plus its enum/name-table entries — see [§7](#7-extension--feature-gating-model).

### 1.4 Portability as a hard, cross-cutting requirement

`fARM64` is `#![no_std]` **unconditionally**, with **no `alloc` dependency on the core decode path or the default formatter path**. It builds for hosted targets, `wasm32-unknown-unknown`, and `aarch64-unknown-none` (bare-metal, no-CRT, no OS, possibly no allocator). All names are `&'static str` from `const`/`static` tables; there are no thread-locals, no env/IO/time, no panics-as-control-flow, and no hard floating-point reliance in the decoder. `alloc` and `std` are strictly *additive* opt-in features. See [§8](#8-no_std--alloc--std-posture).

---

## 2. Chosen architecture & rationale

### 2.1 The architecture: a hand-written recursive decode tree

`fARM64` is a **hand-written, spec-derived recursive decode tree** projected through a *clean idiomatic public API*. Decode is plain Rust control flow: a top-level dispatch on the A64 encoding-group selector chooses one of eight hand-written group decoders, each of which matches its sub-fields, applies the ARM pseudocode helpers it needs, and builds the `Copy` `Instruction` directly.

```
                 +-------------------------------------------------+
   USER  <---->  |  (2) IDIOMATIC PUBLIC SURFACE                   |
                 |      Decoder / Instruction / Operand / Code /   |
                 |      Mnemonic / Register / Formatter / Info     |
                 |      (Copy value types, iced-x86 shape)         |
                 +-------------------------------------------------+
                 |  (1) HAND-WRITTEN RECURSIVE DECODE TREE         |
                 |      match (word >> 25) & 0xf  ->  group fn     |
                 |      per-group decoders match sub-fields and    |
                 |      build Instruction directly                 |
                 +-------------------------------------------------+
                 |  ARM pseudocode helpers (HAND-WRITTEN, spec):   |
                 |      decode_bit_masks (correct tmask!=wmask),   |
                 |      adv_simd_expand_imm, vfp_expand_imm,       |
                 |      move_wide_preferred, replicate, ...        |
                 +-------------------------------------------------+
```

**(1) Hand-written recursive decode tree** (`src/decode/`). `decode::decode_into(word, ip, features, out)` performs a top-level `match (word >> 25) & 0xf` on **`op0 = bits[28:25]`** and dispatches to one of the eight A64 encoding-group decoders. Each group decoder is ordinary, reviewable Rust: it matches the discriminating sub-fields of its group, extracts named fields via small inline bit helpers (`src/decode/bits.rs`), runs any ARM pseudocode it needs, applies the ARM ARM **alias conditions** to pick the preferred mnemonic, and writes operands straight into the `Instruction`. The ARM pseudocode functions — `DecodeBitMasks`, `AdvSIMDExpandImm`, `VFPExpandImm`, `MoveWidePreferred`, `Replicate`, `HighestSetBit`, sign/zero-extend — are plain hand-written functions in `bits.rs`, transcribed from the spec.

**(2) Idiomatic public surface** (`src/{decoder,instruction,operand,enums,...}.rs`). The clean-types layer: `Copy` value types, a borrowing iterator decoder, rich enums, a `core::fmt::Write`-friendly formatter. Users never see the internal decode representation. `decoder.rs::decode_into` simply calls `crate::decode::decode_into`.

**Aliases / preferred disassembly** (MOV←ORR, CMP←SUBS, MUL←MADD, LSL/LSR/ASR←UBFM/SBFM, NOP←HINT, …) are applied **in code**, inside each group decoder, exactly per the ARM ARM "alias conditions" for that encoding. They are not a separate data table; the canonical `Code` is always recorded, and the resolved alias `Mnemonic` is selected by the decoder and surfaced under the formatter's alias option.

### 2.2 Rationale — why hand-written, and why not the alternatives

Three evaluation lenses (correctness, maintainability/idiom, effort/extensibility) converge on a hand-written tree once the provenance goal is fixed: **fARM64 is an original implementation derived from the ARM ARM, not a derivative of any other decoder.** That goal rules out the two table-driven shortcuts and selects clean Rust control flow.

| Alternative | What it is | Why rejected |
|-|-|-|
| Transpile Binary Ninja's `arch-arm64` | Machine-port binja's generated C into Rust tables | Makes the crate a **derivative work** of a third party (the provenance goal forbids it); also inherits binja's *deliberate* spec divergences (e.g. `tmask==wmask` at `pcode.c:88`), which we explicitly do **not** want |
| Generic table-interpreter (`EncId`/`EncDef`/`OpStep`/`Guard`/`FieldSpec`) | A data-driven engine interpreting `&'static` encoding rows | The encoding table still has to come from *somewhere*; hand-authoring thousands of opaque rows is less reviewable than code, the interpreter adds an indirection layer with no correctness benefit, and irregular cases (SVE element math, SME tiles) fight the data model |
| **Hand-written recursive decode tree** (chosen) | Plain Rust `match` per encoding group, ARM pseudocode as functions | **Original provenance**; decode logic is *readable Rust*, not opaque data; spec divergences are an explicit, auditable choice; trivially `no_std`/zero-alloc; aliases live next to the encoding they alias |

**Why hand-written wins on correctness.** The ARM ARM is the authority; transcribing it directly into Rust functions keeps the spec and the code in one-to-one correspondence, which is exactly what makes per-encoding review and unit-testing-against-the-spec tractable. Deliberate divergences (like a correct `tmask`) are visible at the call site, not buried in a generated table or an upstream approximation we inherited by accident.

**Why hand-written wins on portability.** A recursive `match` tree over integer bit-fields is `const`-friendly, allocation-free, and wasm/bare-metal-friendly *by nature*. There is no interpreter state, no `Vec` of operand steps, no table relocation — just integer math writing into a `Copy` struct. The pseudocode helpers are a few hundred lines of pure integer logic, ideal for `wasm32-unknown-unknown` and `aarch64-unknown-none`.

**Why hand-written wins on maintainability.** Decode logic reads like the ARM ARM it came from. A reviewer compares a group decoder against the manual's encoding diagram directly; there is no second grammar (`OpStep`/`Guard` bytecode) to learn and keep in sync. New encodings are new match arms next to their siblings.

---

## 3. Decode strategy — 32-bit word → (Code, operands)

A 32-bit little-endian word becomes `(Code, [Operand; up to MAX_OPERANDS])` with **zero heap allocation**. The whole path is hand-written Rust: a top-level group dispatch, a per-group decoder, and shared bit/pseudocode helpers.

### Word ingest

`Decoder::decode_into` reads 4 bytes LE at the cursor into a `u32 word` and captures `ip`. There is no lookahead — A64 is fixed 32-bit, 4-byte aligned, and PC is always `ip` (the A64 PC for a PC-relative instruction is the address of the instruction itself; `next_ip == ip + INSN_LEN`, and `INSN_LEN == 4`).

### Top-level dispatch — `decode::decode_into`

The entry function performs `match (word >> 25) & 0xf` on **`op0 = bits[28:25]`**, the A64 encoding-group selector from the ARM ARM's top-level classification (manual section C4). It also handles the universal special cases here: `UDF`/reserved patterns and the `HINT` space (so `NOP`/`YIELD`/`SEV`/… alias resolution is anchored at the entry point). Each arm calls one hand-written group decoder:

| `op0 = (word>>25)&0xf` | A64 encoding group | Decoder module |
|-|-|-|
| `0b0000` | Reserved (`UDF`) + SME (gated, `word<31>==1`) | entry `decode_reserved` → `sme/` |
| `0b0001`, `0b0011` | Unallocated | (left `Invalid`) |
| `0b0010` | SVE / SVE2 (gated) | `sve/` |
| `0b1000`, `0b1001` | Data Processing — Immediate | `dp_imm.rs` |
| `0b1010`, `0b1011` | Branches, Exception, System | `branch_sys.rs` |
| `0b0100`, `0b0110`, `0b1100`, `0b1110` (`x1x0`) | Loads & Stores | `ldst.rs` (+ `ldst_simd.rs`) |
| `0b0101`, `0b1101` (`x101`) | Data Processing — Register | `dp_reg.rs` |
| `0b0111`, `0b1111` (`x111`) | Data Processing — Scalar FP & Advanced SIMD | `simd_fp/` |

(The exact bit-pattern partitioning of the load/store, DP-register, and FP/SIMD groups follows the ARM ARM C4 tables; the table above is the dispatch summary, not the full predicate set. SME shares the reserved `0b0000` region with `UDF`, disambiguated by `word<31>`: `UDF` has `word<31:16>==0`, SME has `word<31>==1`.)

### Per-group decoders (hand-written)

Each group module owns the encodings in its space and turns a word into an `Instruction`:

- **`decode/dp_imm.rs`** — PC-relative `ADR`/`ADRP`; add/subtract immediate; logical immediate (via `decode_bit_masks`); move wide (`MOVZ`/`MOVN`/`MOVK`, with `MOV` alias via `move_wide_preferred`); bitfield (`SBFM`/`BFM`/`UBFM` and their `LSL`/`LSR`/`ASR`/`SXT*`/`UXT*`/`BFI`/… aliases); extract (`EXTR`, `ROR` alias).
- **`decode/branch_sys.rs`** — unconditional branch `B`/`BL`; conditional `B.cond`; compare-and-branch `CBZ`/`CBNZ`; test-bit `TBZ`/`TBNZ`; unconditional branch register `BR`/`BLR`/`RET`/`ERET`; exception-generating `SVC`/`HVC`/`SMC`/`BRK`/…; system register move `MRS`/`MSR`; `SYS`/`SYSL` and their `AT`/`DC`/`IC`/`TLBI` aliases; barriers `DMB`/`DSB`/`ISB`; the `HINT` space and aliases `NOP`/`YIELD`/`SEV`/`SEVL`/`WFE`/`WFI`/…
- **`decode/ldst.rs`** — load literal; load/store register (unsigned imm offset, register offset, unscaled `LDUR`/`STUR`); pre/post-index; load/store pair (and `LDP`/`STP` no-allocate forms); load/store exclusive; LSE atomics (behind the `lse` feature); SIMD load/store single-structure and multiple-structures.
- **`decode/dp_reg.rs`** — logical (shifted register); add/subtract (shifted register) and (extended register); add/subtract with carry; rotate/conditional compare; conditional select; data-processing (1-source, 2-source, 3-source) including the `MUL`/`MNEG`/`MOV`/`NEG`/`CMP`/`CMN`/`TST`/`NGC` aliases.
- **`decode/ldst_simd.rs`** — Advanced-SIMD load/store of multiple and single structures (`LD1`..`LD4` / `ST1`..`ST4`), producing `Operand::MultiReg{ regs, count, arr, lane }` with the post-index immediate/register variants.
- **`decode/simd_fp/`** — scalar floating-point and Advanced SIMD, split into sub-modules:
  - `scalar_fp.rs` — scalar FP data-processing: conversions to/from integer and fixed-point, 1/2/3-source FP, compares, conditional compare/select, FP immediate (via `vfp_expand_imm`).
  - `simd_arith.rs` — Advanced SIMD arithmetic (three-same / three-different / pairwise / across-lanes / scalar variants).
  - `simd_data.rs` — Advanced SIMD data-movement (permute / table / copy / modified-immediate via `adv_simd_expand_imm` / shift-by-immediate / extract).
  - `crypto.rs` — AES/SHA/SM3/SM4 family, gated by the `crypto` cargo feature.
- **`decode/sve/`** (gated by `sve`) — SVE/SVE2, dispatched by `word<31:29>` into `sve_int.rs` (integer/shift/reduction/INDEX/INC-DEC/CNT/compare-imm/MOV-DUP-CPY/DOT and the SVE2 multiply-add/widening), `sve_perm.rs` (permute/predicate-logical/table/unpack), `sve_fp.rs` (floating-point), and `sve_mem.rs` (gather/scatter/contiguous loads/stores/prefetch with `MUL VL` and the SVE addressing modes in `Operand::SveMem`).
- **`decode/sme/`** (gated by `sme`) — SME: outer-products (`FMOPA`/`BFMOPA`/`[US]MOPA`…), `MOVA`/`ADDHA`/`ADDVA`, and ZA-array load/store, producing `Operand::SmeTile`/`SmeTileSlice`. Reached from `decode_reserved` (`op0==0b0000`, `word<31>==1`).

Register-31 SP-vs-ZR is resolved **in the group decoder** per the encoding's class, so reg-31 is stored as `SP`/`WSP`/`XZR`/`WZR` and **never leaks raw** to callers.

### Shared bit-field & ARM pseudocode helpers — `decode/bits.rs`

All groups share a small hand-written helper module:

- **Bit extraction:** inline `bits(word, hi, lo)`, single-bit, sign-extend (`sign_extend(value, from_bits)`), zero-extend, `HighestSetBit`, `Replicate`, `ones`.
- **ARM pseudocode (transcribed from the spec):**
  - `decode_bit_masks(N, imms, immr, immediate)` — logical-immediate and bitfield mask derivation. **Computes `tmask` correctly per the ARM ARM**, distinct from `wmask` (the intentional divergence from binja's `pcode.c:88` shortcut, recorded on the harness allow-list).
  - `move_wide_preferred(sf, N, imms, immr)` — selects the `MOV` (wide-immediate) alias for `MOVZ`/`MOVN`.
  - `adv_simd_expand_imm(op, cmode, imm8)` — Advanced SIMD modified-immediate expansion.
  - `vfp_expand_imm(imm8, width)` — scalar FP immediate expansion.
  - `decode_shift`, condition decoding, and the small extend/shift selectors.

All integer math, no heap, deterministic.

### Aliases & errors

- **Aliases / preferred disassembly:** decode always records the canonical `code: Code`. The group decoder additionally resolves the preferred `mnemonic: Mnemonic` per the ARM ARM alias conditions; the formatter's `aliases` option (default on) chooses which to print — mirroring iced's `Code`/`Mnemonic` split.
- **Errors:** any unallocated / reserved / feature-gated-out pattern returns a typed `DecodeError` (e.g. `Reserved`/`Unallocated`/`Undefined`/`FeatureRequired`/`EndOfInstruction`). On failure the decoder yields an `Instruction` whose `code == Code::Invalid` and records `last_error()`.

---

## 4. Module layout

```
fARM64/                          # workspace
├── src/                           # the published crate (#![no_std])
│   ├── lib.rs                     # crate root; #![no_std], #![allow(non_snake_case)];
│   │                              #   module decls, public re-exports, crate docs;
│   │                              #   static_asserts (Operand<=16B, Instruction<=112B, Copy
│   │                              #   witnesses); MAX_OPERANDS, INSN_LEN
│   │
│   ├── decoder.rs                 # Decoder<'a>: borrowing, zero-alloc, IntoIterator.
│   │                              #   decode_into calls crate::decode::decode_into
│   │                              #   key: Decoder<'a>, DecoderOptions, DecoderIntoIter, DecoderIter
│   ├── instruction.rs             # Copy value-type Instruction (derives PartialEq only;
│   │                              #   FpImm payload => no Eq/Hash). flags: u8.
│   ├── operand.rs                 # rich Operand enum + OpKind discriminant view
│   │                              #   key: Operand, OpKind, MemIndexMode, PredQual,
│   │                              #        SliceIndicator, SveMemMode
│   ├── register.rs                # #[repr(u16)] Register (+ Pf0..Pf31) + const gp_register
│   │                              #   key: Register, RegClass, RegWidth
│   ├── enums.rs                   # Condition, ShiftType, ExtendType, VectorArrangement,
│   │                              #   FlagEffect, FlowControl
│   ├── mnemonic.rs                # Code (via codes! macro) + Mnemonic + name lookup
│   │                              #   key: Code, Mnemonic
│   ├── sysreg.rs                  # SystemReg packed key + binary-search name()
│   ├── sysop.rs                   # SysToken: &'static str system-instruction keyword operands
│   ├── features.rs                # FeatureSet (features0/features1), Feature enum, presets
│   ├── error.rs                   # DecodeError (core Display; std::error::Error under cfg(std))
│   ├── info.rs                    # InstructionInfo flow/access (instruction_info: STUB today)
│   │                              #   + alloc-gated InstructionInfoFactory
│   │
│   ├── decode/                    # HAND-WRITTEN recursive decode tree (the decode core)
│   │   ├── mod.rs                 #   decode_into / decode; top-level match (word>>25)&0xf;
│   │   │                          #   UDF/reserved + SME/SVE routing
│   │   ├── bits.rs               #   shared bit-field extraction + ARM pseudocode helpers
│   │   ├── dp_imm.rs              #   Data Processing -- Immediate
│   │   ├── branch_sys.rs          #   Branches, Exception generating & System
│   │   ├── ldst.rs                #   Loads & Stores (LSE atomics behind `lse`)
│   │   ├── ldst_simd.rs           #   Adv-SIMD load/store of structures (LD1..LD4 / ST1..ST4)
│   │   ├── dp_reg.rs              #   Data Processing -- Register
│   │   ├── simd_fp/               #   Scalar FP & Advanced SIMD
│   │   │   ├── mod.rs             #     C4.1.97 sub-classifier
│   │   │   ├── scalar_fp.rs       #     scalar FP data-processing
│   │   │   ├── simd_arith.rs      #     Adv-SIMD arithmetic
│   │   │   ├── simd_data.rs       #     Adv-SIMD data-movement / modified-immediate
│   │   │   └── crypto.rs          #     AES/SHA/SM3/SM4 (feature `crypto`)
│   │   ├── sve/                   #   SVE/SVE2 (feature `sve`)
│   │   │   ├── mod.rs             #     dispatch on word<31:29>
│   │   │   ├── sve_int.rs         #     integer / shift / reduction / INDEX / DOT / SVE2
│   │   │   ├── sve_perm.rs        #     permute / predicate-logical / table / unpack
│   │   │   ├── sve_fp.rs          #     floating-point
│   │   │   └── sve_mem.rs         #     loads / stores / prefetch (MUL VL, gather/scatter)
│   │   └── sme/                   #   SME (feature `sme`)
│   │       └── mod.rs             #     outer-products / MOVA / ADDHA-ADDVA / ZA load-store
│   │
│   ├── encode/                    # HAND-WRITTEN ENCODER — the inverse of decode/
│   │   ├── mod.rs                 #   encode(&Instruction)->Result<u32,EncodeError> +
│   │   │                          #   Instruction::encode; EncodeError; dispatch on code()
│   │   ├── bits.rs                #   inverse pseudocode: encode_bit_masks, inverse
│   │   │                          #   vfp_expand_imm / adv_simd_expand_imm; field packers
│   │   ├── dp_imm.rs              #   Data Processing -- Immediate (inverse)
│   │   ├── dp_reg.rs              #   Data Processing -- Register (inverse)
│   │   ├── branch_sys.rs          #   Branches, Exception generating & System (inverse)
│   │   ├── ldst.rs                #   Loads & Stores (inverse)
│   │   ├── ldst_simd.rs           #   Adv-SIMD load/store of structures (inverse)
│   │   ├── simd_fp.rs             #   Scalar FP & Advanced SIMD dispatch (inverse), with
│   │   │                          #   simd_fp_scalar / simd_fp_arith / simd_fp_data /
│   │   │                          #   simd_fp_crypto siblings
│   │   ├── sve.rs                 #   SVE/SVE2 (inverse)
│   │   └── sme.rs                 #   SME (inverse)
│   │
│   ├── tables/                    # MECHANICAL name/enum tables ONLY
│   │   ├── mod.rs                 #   facade for the name lookups
│   │   └── names.rs               #   register/condition/mnemonic/sysreg name tables (&'static str)
│   │
│   └── format/                    # formatter — ZERO-ALLOC core path
│       ├── mod.rs                 # Formatter + FormatterOutput + FormatterOptions + TokenKind +
│       │                          #   SymbolResolver/SymbolResult + format_to_string (alloc)
│       ├── fmt_writer.rs          # default no_std FmtFormatter (UAL); BufSink(&mut [u8])
│       └── gnu.rs                 # optional GNU/objdump dialect (feature 'fmt-gnu')
│
├── xtask/                         # OPTIONAL host-only tooling — NOT part of the published crate.
│
├── tests/
│   ├── common/mod.rs              # corpus streaming + normalize pipeline + fixed test address
│   ├── golden.rs                  # binja-corpus differential sweep (coverage + parity report)
│   ├── llvm_diff.rs               # llvm-mc cross-check (skips if llvm-mc absent)
│   └── roundtrip.rs               # decode -> encode -> re-decode; semantic round-trip parity
│
└── examples/
    └── disasm.rs                  # tiny CLI: hex word(s) -> text via BufSink + FmtFormatter
```

> There is no `src/runtime/` table-interpreter and no `EncId`/`EncDef`/`OpStep`/`Guard`/`FieldSpec`/`AliasRule` machinery: decode is plain Rust control flow. There is currently no `benches/` directory.

### Layer ownership at a glance

| Layer | Modules | Authored how |
|-|-|-|
| Public surface (projection) | `lib`, `decoder`, `instruction`, `operand`, `register`, `enums`, `mnemonic`, `sysreg`, `sysop`, `features`, `error`, `format`, `info` | Hand-written idiomatic Rust |
| Hand-written decode core | `decode::{mod, dp_imm, branch_sys, ldst, ldst_simd, dp_reg, simd_fp/*, sve/*, sme/*}` | Hand-written from the ARM ARM |
| Hand-written encoder (inverse) | `encode::{mod, dp_imm, dp_reg, branch_sys, ldst, ldst_simd, simd_fp*, sve, sme}` | Hand-written; inverse of the decode core |
| ARM pseudocode + bit helpers | `decode::bits`, `encode::bits` (inverse) | Hand-written from the ARM ARM |
| Mechanical name/enum tables | `tables::{mod, names}`, the `codes!` macro in `mnemonic.rs` | Declarative source, committed |
| Build-time tooling | `xtask` | Hand-written, **not published**, optional |
| Verification | `tests/{golden, llvm_diff, roundtrip}.rs`, `examples/disasm.rs` | Hand-written |

---

## 5. Data-flow diagram

```
                              bytes: &[u8]  (caller-owned, borrowed)
                                     |
                                     v  read 4 LE @ cursor; capture ip (no lookahead)
                            +------------------+
                            |   u32  word      |
                            +------------------+
                                     |
         ===================  HAND-WRITTEN DECODE CORE  ====================
                                     |
                                     v
                    decode::decode_into(word, ip, features, out)   [decode/mod.rs]
                       match (word >> 25) & 0xf   (op0 = bits[28:25])
                       + UDF/reserved + HINT special-case
                                     |
         +---------+-----------+-----------+---------+-----------+---------+
         v         v           v           v         v           v         v
      dp_imm   branch_sys   ldst(+      dp_reg   simd_fp/   sve/    sme/
      [dp_imm] [branch_sys] ldst_simd)  [dp_reg] [scalar_fp (gated)  (gated)
              |          |    [ldst]      |       simd_arith   |        |
              |          |        |       |       simd_data]   |        |
              +----------+--------+-------+-----------+--------+--------+
                                     |
                                     v  each group decoder:
                       1. extract named fields        [decode/bits.rs]
                       2. run ARM pseudocode as needed [decode/bits.rs]
                          decode_bit_masks (correct tmask), move_wide_preferred,
                          adv_simd_expand_imm, vfp_expand_imm, ...
                       3. apply ARM alias conditions -> resolved Mnemonic
                       4. resolve reg-31 -> SP/ZR via gp_register; write operands
                          directly into Instruction  (no alloc)
                                     |
                                     v  on no match -> typed DecodeError; code = Invalid
                                     |
         =========================  PUBLIC PROJECTION  =====================
                                     v
                          +-----------------------------+
                          |  Instruction (Copy)         |   code, mnemonic, ip, op_count,
                          |  [Operand; MAX_OPERANDS]    |   flags, raw word
                          +-----------------------------+
                                     |
            +------------------------+-------------------------+
            v                                                  v
   Formatter::format(insn, &mut out)               InstructionInfo (flow/access)
   [format/fmt_writer.rs]                          [info.rs]
            |  tokens                                          |
            v                                                  v
   FormatterOutput sink:                            used_registers / used_memory /
     - any T: core::fmt::Write (no_std)               flow_control  (fixed-capacity,
     - BufSink(&mut [u8])   (no alloc)                 no alloc on core path)
     - String collector     (feature = alloc)
```

The cursor advances by exactly `INSN_LEN` (4) bytes per decode; `next_ip` advances to `ip + 4`. On any decode failure the decoder returns an `Instruction` whose `code == Code::Invalid` and records `last_error()`.

---

## 6. Relation to the Binary Ninja reference

`refs/arch-arm64-master/disassembler` is Binary Ninja's `arch-arm64`. In the **new** architecture it is **not a source** for fARM64 — fARM64 contains no transpiled binja code and is an original implementation derived from the ARM ARM.

Binja's role is limited to **one differential test oracle during development**:

- Its corpus, `refs/arch-arm64-master/disassembler/test_cases.txt`, is read **locally only**, used as **a development guide** alongside LLVM and GNU binutils, and is **never shipped** and **not authoritative**.
- Where binja deliberately diverges from the ARM ARM, fARM64 **follows the spec** and records the divergence on a documented allow-list in the differential harness. The canonical example is `DecodeBitMasks` at `pcode.c:88`, where binja short-cuts `tmask = wmask` (`// TODO: do this right`); fARM64 computes the correct `tmask`, so those cases are expected to differ from the binja corpus and are allow-listed, not "fixed" toward binja.

There is **no required-attribution obligation to Vector 35**, because nothing of binja's is incorporated. The repository's `NOTICE` is a *courtesy* acknowledgment of the cross-check oracles (it credits the corpus, not the decode logic) and imposes no obligation on users. fARM64 ships under `MIT` (the Rust-ecosystem default) as original work — see [§8](#8-no_std--alloc--std-posture) and [Cargo.toml].

---

## 7. Extension / feature-gating model

Extensions are **gated regions of the hand-written decode tree** plus their enum/name-table entries. There is no day-one ingest of a full table; each extension's decode logic is authored (or stubbed) in its group module and turned on by a feature.

**(1) Gate by feature bit — two independent layers.**

| Layer | What it controls | Mechanism |
|-|-|-|
| Cargo feature (`fp16`, `bf16`, `lse`, `pauth`, `mte`, `sme`, `sve`, `crypto`, `full`) | What is **compiled** | `#[cfg(feature = …)]`-gates the per-extension decode arms in the group modules and the corresponding `Code`/`Register`/name-table entries. A base-only wasm build omits SVE/SME entirely (smaller binary). |
| Runtime `FeatureSet` | What is **accepted** at decode time | Each group decoder checks the active `FeatureSet` before admitting an extension encoding; an extension pattern decoded without its feature yields `DecodeError::FeatureRequired` rather than silently mis-decoding as base ISA. |

`FeatureSet` is a compact bitset (two `u64` words) so decode-time and pcode-time feature questions are cheap. The top-level dispatch routes `op0 = 001x` into the SVE sub-tree only when `Feature::Sve` is set, and `0000`/`0001` into SME only when `Feature::Sme` is set; otherwise the entry function returns the appropriate `DecodeError`. Within base groups, the feature test is the **innermost** gate, applied after structural narrowing.

**(2) Incremental per-extension work is hand-written decode arms + helpers.** Bringing up an extension means: add its encoding arms to the relevant group module(s) (often a dedicated sibling file for SVE/SME), add any new ARM pseudocode helper it needs to `decode/bits.rs`, and append its `Code` rows (to the `codes!` macro in `mnemonic.rs`) plus any new `Mnemonic`/`Register`/name-table entries. The public enums grow by **appended** variants under `#[non_exhaustive]` (append-only discriminant policy, snapshot-tested), so downstream code never breaks.

**(3) Rollout order by ROI:** Base → FP/NEON → FP16 → LSE → PAuth → SME → MTE → BF16 → SVE/SVE2 → crypto/SM3/SM4. Each extension's *done* = its encodings passing the per-encoding spec unit tests and agreeing with the LLVM/binutils oracles (and the binja corpus modulo the allow-list).

---

## 8. `no_std` / `alloc` / `std` posture

Portability is a **hard, structural** requirement, not an afterthought. The crate is `#![no_std]` unconditionally and the **core decode path AND the default formatter are zero-heap** — no `alloc` dependency at all on those paths. The crate root also carries `#![allow(non_snake_case)]` so the `fARM64` name builds warning-free.

### Portability tiers

| Tier | Feature | Always works? | Contents |
|-|-|-|-|
| **A** | *default* (`no_std`, **no alloc**) | **Yes, non-negotiable** | `Decoder`, `decode`/`decode_into`, `FmtFormatter` → `core::fmt::Write` / `BufSink`, all enums & name tables, `InstructionInfo` core, `DecodeError`. The `fmt-gnu` and per-extension features stay within Tier A. |
| **B** | `alloc` | opt-in | `String`/`Vec` conveniences, `InstructionInfoFactory` (allocate-once/refill), token-collecting `String` sink, `format_to_string` |
| **C** | `std` (implies `alloc`) | opt-in | `std::error::Error` impl, std-only test/bench helpers |

**Show the no-alloc path first; treat `String`/`Vec` returns as alloc-gated extras.** The default `FmtFormatter` writes into a caller-supplied `&mut dyn core::fmt::Write` or a fixed `&mut [u8]` (`BufSink`, with overflow tracking — never writes past the buffer). A blanket `FormatterOutput` impl is provided for all `T: core::fmt::Write`, so any `core::fmt::Write` works in `no_std`. `String` gets a token-collecting impl only under `feature = "alloc"`.

### Hard invariants

- **All names are `&'static str`** from `const`/`static` tables (`REG_NAMES`, `MNEMONIC_NAMES`, `SYSREG_NAMES`, arrangement/condition tables).
- **No thread-locals, no env/IO/time, no panics-as-control-flow, deterministic.**
- **No hard floating-point reliance in the decoder.** The decoder is integer-only; the only FP is `Operand::FpImm(f32)` constructed via `f32::from_bits` (a bit-cast, not arithmetic) and its decimal rendering, isolated in the formatter helper. This keeps soft-float / `wasm` / bare-metal targets safe.
- **`Copy` value types with pinned sizes**, enforced by static assertions in `lib.rs`: `size_of::<Operand>() <= 16`, `size_of::<Instruction>() <= 112`, plus `Copy` witnesses; `MAX_OPERANDS` and `INSN_LEN` are crate consts.
- **Hermetic downstream builds**: there is **no `build.rs`**, no XML, and no network at downstream build time. `Code` and its `mnemonic()`/`feature()` accessors are generated in-crate by the declarative `codes!` macro in `mnemonic.rs`; the `&'static str` name tables are committed source under `src/tables/`. `xtask` is host-only, optional, and not part of the published crate.

### Provenance & licensing

`fARM64` is an **original implementation derived from the ARM Architecture Reference Manual**. It is **not** a derivative of Binary Ninja's `arch-arm64` (which is used only as one local development oracle). The crate ships under **`MIT`** (set in `Cargo.toml`) with **no required-attribution obligation to any third party**.

### Supported targets (enumerated in `lib.rs`)

| Target | Tier | Notes |
|-|-|-|
| `x86_64-*`, `aarch64-*` (hosted) | A/B/C | Primary dev/test targets |
| `wasm32-unknown-unknown` | A | Default features only; built in CI |
| `aarch64-unknown-none` | A | Bare-metal / no-CRT / no OS; `-Zbuild-std=core` smoke in CI |
| any target with `core` | A | The default path needs only `core` |

### Enforcement (CI, non-negotiable from P0)

- Build the core + default formatter for `wasm32-unknown-unknown` and `aarch64-unknown-none` with **default features** — a compile failure is a test failure.
- A `no-alloc-audit` feature installs a global allocator that **panics on any allocation** and runs `decode_into` + `FmtFormatter`+`BufSink` over the corpus to *prove* zero heap.
- Static assertions pin `size_of`/`align`/`Copy` for `Instruction` and `Operand`.

---

## 9. Verification posture (summary)

Correctness is defined against the **ARM ARM** and enforced by a differential harness plus unit tests; full detail and the measured results live in [VALIDATION.md](./VALIDATION.md).

- **Unit tests.** `cargo test --lib` covers the pseudocode helpers and the public enums/accessors (condition round-trip, shift/extend decode, arrangement suffixes, flow/flag classification).
- **Differential harness — two oracles.** `tests/golden.rs` decodes the **binja corpus** (`test_cases.txt`, a development guide) at the fixed test address, formats with the default `FmtFormatter`, normalizes both sides (`tests/common/mod.rs`) and reports coverage + parity bucketed by group. `tests/llvm_diff.rs` cross-checks individual words against **LLVM** `llvm-mc` to confirm spec-correctness (skips cleanly if `llvm-mc` is absent). The normalization pass handles excusable differences (`cs==hs`, `cc==lo`, hex/decimal, float forms, `{zN-zM}` range expansion); the remaining mismatches are the documented spec-vs-binja divergences (notably the correct `DecodeBitMasks` `tmask`, REVD/pmul/histseg/match/nmatch/saba/uaba/sqxtn*/sqrdml* renderings) — see the divergence table in [VALIDATION.md](./VALIDATION.md).
- **Robustness.** Decode is total and panic-free for all 2^32 words; `len()` is always `INSN_LEN` (4). Current measured result: **99.35% coverage at 99.78% parity** (42016/42289 attempted, 41925 matched).

---

## 10. Cross-references

- **[API.md](./API.md)** — `Decoder`, `Instruction`, `Operand`, `Code`/`Mnemonic`/`Register`/`OpKind`, `Formatter`/`FormatterOutput`/`FormatterOptions`/`TokenKind`, `FeatureSet`, `DecodeError`, `InstructionInfo` — every public signature with examples (no-alloc path shown first).
- **[ENCODING.md](./ENCODING.md)** — the full encoding taxonomy (top-level `op0` dispatch, DP-Immediate, Branch/System, Loads & Stores, DP-Register, FP/SIMD, SVE, SME), the per-group bit-field layouts, and the ARM pseudocode helpers (`DecodeBitMasks`, `AdvSIMDExpandImm`, `VFPExpandImm`, `MoveWidePreferred`) plus the alias / preferred-disassembly rules.
- **[ROADMAP.md](./ROADMAP.md)** — completed milestones (base ISA, Adv-SIMD/FP, SVE, SIMD ld/st, partial SME, crypto) and what remains.
- **[VALIDATION.md](./VALIDATION.md)** — methodology + measured results (coverage/parity), the two oracles, the harness, and the spec-vs-binja divergence table.

---

*`fARM64` is an original implementation derived from the ARM Architecture Reference Manual. It is licensed `MIT` and is not a derivative of any third-party decoder.*
