# fARM64 — Project Roadmap

> Pure-Rust, `no_std`, zero-heap AArch64 (A64) disassembler.
> Crate name: **`fARM64`** ("farm" + "ARM64"). Package + lib name are both `fARM64`;
> imports read `use fARM64::...`. The crate root carries `#![allow(non_snake_case)]`.
>
> This is the ordered milestone plan. Each task carries a **priority** (`P0` base
> ISA / critical path … `P3` niche) and a rough **effort** estimate (`S` ≤ 0.5 day,
> `M` ≈ 1–2 days, `L` ≈ 3–5 days, `XL` > 1 week).

---

## Delivered status (measured)

The decoder, default UAL formatter and the base + main extension ISA are
**implemented and validated**. Live corpus result: **99.35% coverage at 99.78%
parity** (42016/42289 attempted, 41925 matched) — see [VALIDATION.md](./VALIDATION.md).

**Done**

- ✅ Base ISA: DP-Immediate, Branch/Exception/System, Loads & Stores, DP-Register,
  Scalar FP & Advanced SIMD (the hand-written decode tree in `src/decode/`).
- ✅ Advanced-SIMD load/store of structures (`LD1`..`LD4` / `ST1`..`ST4`, `ldst_simd.rs`).
- ✅ Adv-SIMD detail + modified-immediate (`simd_data.rs`).
- ✅ SVE / SVE2 (`src/decode/sve/`): integer/perm/fp/mem families (corpus-dominant).
- ✅ Crypto (AES/SHA/SM3/SM4, `simd_fp/crypto.rs`).
- ✅ Default zero-alloc UAL `FmtFormatter` + alias/preferred-disassembly; `BufSink`.
- ✅ Differential harness: `tests/golden.rs` (binja corpus) + `tests/llvm_diff.rs`
  (LLVM cross-check) + `examples/disasm.rs`; documented spec-vs-binja divergences.
- ✅ Core types & enums, `Decoder` iterator ergonomics, `FeatureSet` runtime gating,
  size/Copy static asserts; `Code` via the in-crate `codes!` macro.

**Partial / remaining**

- ◐ SME (`src/decode/sme/`): outer-products / MOVA / ZA load-store implemented;
  full SME2 multi-vector coverage remains.
- ◐ FP16 / BF16, LSE, PAuth, MTE: feature gates exist; coverage to be completed per
  the corpus gaps in [VALIDATION.md](./VALIDATION.md).
- ☐ `info::instruction_info` (per-register access analysis) is a stub today;
  `Instruction::flow_control()`/`set_flags()` are implemented.
- ☐ A handful of niche forms (vector `BIF`/`BIT`/`BSL`, `SHLL`, `LD64B`/`ST64B*`).
- ✅ Semantic encoder + decode→encode→decode round-trip (`src/encode/`,
  `tests/roundtrip.rs`): **100.00% semantic round-trip** parity over the corpus —
  see [VALIDATION.md](./VALIDATION.md#round-trip-encoder).
- ☐ GNU formatter (`fmt-gnu`), proptest/fuzz suite, benches, no-alloc-audit gate
  — not yet built.

---

## Architecture in one paragraph

fARM64 is a **hand-written A64 disassembler derived from the ARM Architecture Reference
Manual (the "ARM ARM")**, with original provenance. The decode core is a **hand-written
recursive decode tree**: top-level dispatch on `op0 = bits[28:25]` selects one of the eight
A64 encoding groups, and hand-written per-group decoders match sub-fields and build the
`Instruction` directly. ARM pseudocode (`DecodeBitMasks`, `AdvSIMDExpandImm`, `VFPExpandImm`,
`MoveWidePreferred`, `Replicate`, `HighestSetBit`, sign/zero-extend) is implemented as plain
hand-written functions. Aliases / preferred-disassembly are applied in code per the ARM ARM
"alias conditions". There is **no** table-interpreter, **no** transpiler, and **no** decode
logic generated from any third-party source.

---

## Guiding invariants (apply to every milestone)

These are non-negotiable constraints. Any task that violates one is not "done".

- **Correctness is defined as conformance to the ARM ARM.** "Correct" = matches the ARM
  Architecture Reference Manual, cross-checked against multiple independent oracles: LLVM
  (`llvm-mc -disassemble` / `llvm-objdump`), GNU binutils `objdump`, and Binary Ninja's
  `arch-arm64` corpus (`refs/arch-arm64-master/disassembler/test_cases.txt`) used **only** as
  a development guide. Where Binary Ninja deliberately diverges from the spec (e.g. the
  `DecodeBitMasks` `tmask == wmask` shortcut at `pcode.c:88`), fARM64 **follows the spec** and
  records the intentional divergence in a documented allow-list. The binja corpus is read
  locally during development only — never shipped, never authoritative.
- **Original provenance / permissive license.** This is an original implementation from the
  ARM ARM, **not** a derivative of Binary Ninja's `arch-arm64`. License is `MIT`
  (the Rust default; already set in `Cargo.toml`). There is **no** required-attribution
  obligation to Vector 35; the repo's `NOTICE` is a *courtesy* acknowledgment of the
  cross-check oracles (it credits the corpus, not the decode logic) and imposes no obligation.
- **Portability is a hard, cross-cutting requirement.** The crate is `#![no_std]`
  unconditionally; the default build is `no_std` with **no `alloc`** and is
  freestanding / no-CRT. There is **no `alloc` dependency on the core decode path or the
  default formatter path**. It must build for hosted (`x86_64-*` dev),
  `wasm32-unknown-unknown`, and `aarch64-unknown-none` (freestanding / no allocator). `alloc`
  and `std` are strictly **additive** opt-in features.
- **Zero heap on the hot path.** Decode produces a `Copy` value-type `Instruction`; the
  default `FmtFormatter` writes into a caller-supplied `&mut dyn core::fmt::Write` or a fixed
  `&mut [u8]` (`BufSink`) with overflow tracking. All names are `&'static str` from
  `const`/`static` tables. No thread-locals, no env/IO/time, no panics-as-control-flow,
  deterministic, no floating-point math in the decoder.
- **Hermetic, no build.rs.** The library has **no `build.rs`** and no network/XML access at
  build time — downstream builds are hermetic. `Code` (with its `mnemonic()`/`feature()`
  accessors) is generated in-crate by the declarative `codes!` macro in `mnemonic.rs`; the
  `&'static str` register/condition/sysreg name tables are committed source under
  `src/tables/`. `xtask` (host-only, optional) is not part of the published crate and
  generates no decode logic.
- **Append-only public discriminants.** `Code`/`Mnemonic`/`Register` are `#[repr(u16)]`
  `#[non_exhaustive]`; new ARM revisions append variants, never renumber. Enforced by a
  snapshot test.
- **Stable, strategy-independent public API.** The iced-x86-inspired public surface
  (`Decoder`, `Copy` `Instruction`, `Operand`/`OpKind`, `Code`/`Mnemonic`/`Register`,
  `Formatter` + `FormatterOutput`/`BufSink`, `FeatureSet`, `InstructionInfo`) is **unchanged**
  by the decode strategy. These files stay essentially as-is; only stale comments are fixed.

**Always describe and implement the no-alloc path first.** Treat `String`/`Vec` returns as
`alloc`-gated extras.

---

## Module layout (pin exactly; docs and code must agree)

- `src/lib.rs` — crate root; `#![no_std]`, `#![allow(non_snake_case)]`, module decls, public
  re-exports, crate docs (hand-written-from-spec thesis + portability + feature/target matrix),
  the static asserts (`Operand <= 16B`, `Instruction <= 112B`, `Copy` witnesses),
  `MAX_OPERANDS`, `INSN_LEN`.
- **Public API surface (keep as-is, fix stale comments only):** `decoder.rs`, `instruction.rs`,
  `operand.rs`, `register.rs`, `mnemonic.rs`, `enums.rs`, `sysreg.rs`, `features.rs`,
  `error.rs`, `info.rs`, `format/*`.
- **Hand-written decode core under `src/decode/`** (replaces the old interpreter):
  - `decode/mod.rs` — `pub fn decode_into(word, ip, features, out: &mut Instruction)` and
    `pub fn decode(word, ip, features) -> Instruction`; top-level `match (word >> 25) & 0xf`
    dispatch to the group decoders; UDF/reserved and the HINT special-case.
  - `decode/dp_imm.rs` — Data Processing — Immediate.
  - `decode/branch_sys.rs` — Branches, Exception generating & System.
  - `decode/ldst.rs` + `decode/ldst_simd.rs` — Loads & Stores (incl. SIMD structures).
  - `decode/dp_reg.rs` — Data Processing — Register.
  - `decode/simd_fp/` — Scalar FP & Advanced SIMD (`scalar_fp`, `simd_arith`,
    `simd_data`, `crypto`).
  - `decode/sve/` — SVE/SVE2 (`sve_int`, `sve_perm`, `sve_fp`, `sve_mem`); `decode/sme/` — SME.
  - `decode/bits.rs` — shared bit-field extraction + ARM pseudocode helpers (hand-written).
- **`src/tables/` — mechanical NAME/ENUM tables only:**
  - `tables/mod.rs` — no interpreter types (no `EncId`/`EncDef`/`AliasRule`); only what name
    lookups need. Header: "generated by `cargo xtask gen` from an ARM-spec dataset".
  - `tables/names.rs` — `&'static str` name tables; header carries no third-party attribution.
- **`decoder.rs::decode_into` calls `crate::decode::decode_into`.**

> Removed from the previous design (interpreter machinery, now obsolete): `src/runtime/`
> (`mod.rs`, `fields.rs`, `operands.rs`, `pcode.rs` — pseudocode helpers moved to
> `decode/bits.rs`), `src/tables/classify.rs`, `src/tables/encodings.rs`,
> `src/tables/aliases.rs`, and all `EncId`/`EncDef`/`OpStep`/`Guard`/`FieldSpec`/`AliasRule`
> types and imports.

---

## Milestone index

| # | Milestone | Priority | Theme | Status |
|-|-|-|-|-|
| M0 | Project setup & portability scaffold | P0 | infra | ✅ done |
| M1 | Core types & public enums | P0 | API | ✅ done |
| M2 | Hand-written ARM pseudocode helpers (`decode/bits.rs`) | P0 | decode | ✅ done |
| M3 | Base-ISA group: DP-Immediate (hand-written) | P0 | decode | ✅ done |
| M4 | Base-ISA group: Branches / Exception / System (hand-written) | P0 | decode | ✅ done |
| M5 | Base-ISA group: Loads & Stores (hand-written) | P0 | decode | ✅ done |
| M6 | Base-ISA group: DP-Register (hand-written) | P0 | decode | ✅ done |
| M7 | Base-ISA group: Scalar FP & Advanced SIMD (hand-written) | P0 | decode | ✅ done |
| M8 | Default zero-alloc formatter (UAL) + alias/preferred-disassembly | P0 | format | ✅ done |
| M9 | Differential harness (binja corpus + LLVM cross-check) | P0 | test | ✅ done |
| M10 | Iterator ergonomics, errors, API polish | P1 | API | ✅ done |
| M11 | InstructionInfo (flow/access) facility | P1 | analysis | ◐ flow done; access stub |
| M12 | Extension: Advanced-SIMD details & modified-immediate | P1 | ext | ✅ done |
| M13 | Extension: FP16 / BF16 | P1 | ext | ◐ gated; partial |
| M14 | Extension: LSE (atomics) | P1 | ext | ◐ gated; partial |
| M15 | Extension: Pointer Authentication (PAuth) | P1 | ext | ◐ gated; partial |
| M16 | Extension: MTE (memory tagging) | P2 | ext | ◐ gated; partial |
| M17 | Extension: Crypto (AES/SHA/SHA512/SM3/SM4) | P2 | ext | ✅ done |
| M18 | Extension: SVE / SVE2 | P1 | ext | ✅ done |
| M19 | Extension: SME / SME2 | P2 | ext | ◐ partial (SME2 remains) |
| M20 | GNU formatter, fuzzing, benches, docs | P2 | polish | ☐ todo |
| M21 | Optional xtask name/enum-table generator | P1 | codegen | ☐ optional |
| M22 | Encoder / round-trip | P3 | future | ✅ done (100% semantic round-trip) |

---

## M0 — Project setup & portability scaffold  `P0`

**Goal:** a compiling `no_std` crate skeleton with the full feature/target matrix and CI
proving `wasm32` + bare-metal build **before** any decode logic is filled in.

- [x] `(M)` Cargo workspace: `fARM64` (lib, `#![no_std]`, `#![allow(non_snake_case)]`) and
      `xtask` (offline host-only bin, `publish = false`, excluded from the published crate).
- [x] `(S)` Workspace `Cargo.toml` with `license = "MIT"`; add `LICENSE-MIT`
      and `LICENSE-APACHE`. **No `NOTICE`** crediting any third party for decode logic.
- [x] `(S)` Declare crate features:
      `default = []` (pure `no_std`, no `alloc`); `std` (implies `alloc`); `alloc`;
      `fmt-gnu`; per-extension `fp16`, `bf16`, `lse`, `pauth`, `sme`, `mte`, `sve`,
      `crypto`; and a test-only `no-alloc-audit`.
- [x] `(S)` Pin **MSRV** in `Cargo.toml` (`rust-version`) and a pinned-toolchain CI job.
- [x] `(S)` `#![forbid(unsafe_code)]` as the goal; if `BufSink` needs `unsafe`, narrow it to a
      single audited module with a safety comment, otherwise keep it forbidden.
- [x] `(S)` Lint floor: `#![deny(missing_docs)]` on public items (staged), plus
      `clippy::pedantic` advisory in CI.
- [x] `(M)` CI matrix, **fail on any**:
  - [x] `(S)` hosted (`x86_64`) build + test: `std` + `alloc` + default.
  - [x] `(S)` `wasm32-unknown-unknown` build, default features (no `std`/`alloc`).
  - [x] `(S)` `aarch64-unknown-none` build via `cargo +nightly build -Z build-std=core`,
        default features.
  - [x] `(S)` `cargo fmt --check` and `clippy -D warnings`.
- [x] `(S)` Static asserts: `size_of::<Operand>() <= 16`, `size_of::<Instruction>() <= 112`,
      `align_of`, and `Operand: Copy` + `Instruction: Copy` Copy-witnesses (hand-rolled
      `const _: () = assert!(...)`, no dep on the hot crate).
- [x] `(M)` Stub every public module + re-export so the API doc and skeleton compile with real
      signatures. Group-decoder bodies may be `todo!()` stubs in the template, but the
      signatures + the top-level `match (word >> 25) & 0xf` dispatch wiring must **compile**.
      Gate any panicking stub behind `cfg(test)`/feature so the no-panic invariant holds for
      shipped paths once filled.
- [x] `(S)` `rust-toolchain.toml` for the nightly used only by the build-std smoke.

**Exit criteria:** all three target builds green in CI; size/Copy asserts pass; `cargo doc`
renders the stubbed public surface; `decoder.rs::decode_into` already routes to
`crate::decode::decode_into`.

---

## M1 — Core types & public enums  `P0`

**Goal:** the full idiomatic public surface as compiling types, names from `const` tables,
zero alloc. (These files are the strategy-independent API and stay essentially as-is.)

- [x] `(M)` `enums.rs`: `Code` (`#[repr(u16)] #[non_exhaustive]` — start with `Invalid` + a
      small hand-set; the full set is appended as groups land / from the M21 generator),
      `Mnemonic` (`#[repr(u16)] #[non_exhaustive]`).
- [x] `(S)` `enums.rs`: `Condition` (`#[repr(u8)]`, 16), `ShiftType` (incl. `Msl`),
      `ExtendType` (UXTB…SXTX), `VectorArrangement`, `FlagEffect`, `FlowControl`, `TokenKind`.
- [x] `(S)` `Code` helpers: `const fn mnemonic(self) -> Mnemonic`, `const fn feature(self)
      -> Feature`, `const fn is_base(self) -> bool`.
- [x] `(S)` `Mnemonic::name() -> &'static str`, `Condition::name()`, `VectorArrangement`
      `element_bits()/element_count()/suffix(full)` — backed by `const` tables.
- [x] `(M)` `register.rs`: `Register` (`#[repr(u16)]`: `None`, W/X GP, `Wsp/Sp`, `Wzr/Xzr`,
      scalar `B/H/S/D/Q`, vector `V`, SVE `Z`, predicate `P`, `Pf` prefetch).
- [x] `(S)` `Register` helpers: `const fn width_bits / number / as_x / as_w / is_simd /
      is_sve / name`, plus the free `const fn gp_register(use_sp: bool, width, n)` so raw `31`
      (SP vs ZR) never leaks out of the decoder.
- [x] `(M)` `instruction.rs`: `Instruction` (`#[derive(Debug,Clone,Copy,PartialEq)]` —
      no `Eq`/`Hash` because of the `FpImm(f32)` payload; `word()` accessor for the raw word):
      `word`, `ip`, `code`, `mnemonic`, `op_count`, `flags: u8`, `[Operand; MAX_OPERANDS]`.
      Verify the M0 size budget (`<= 112B`).
- [x] `(M)` `operand.rs`: `Operand` enum (all `Copy`) with rich per-class variants (`Reg`,
      `ImmUnsigned`, `ImmSigned`, `ImmLogical`, `ImmShiftedMove`, `FpImm(f32)`, `ShiftAmount`,
      `Label`, `Cond`, `SysReg`, `MemImm`, `MemExt`, `MultiReg`, `IndexedElement`, `SmeTile`,
      `ImplSpec`), plus `OpKind`, `MemIndexMode` (`Offset|PreIndex|PostImm|PostReg`),
      `PredQual`, `SliceIndicator`. Verify `size_of::<Operand>() <= 16`.
- [x] `(S)` `Instruction` accessors: `code/mnemonic/op_count/op_kind/op/op_register/
      op_immediate/len (const 4)/ip/next_ip/flow_control/is_invalid/set_flags`.
- [x] `(S)` `features.rs`: `Feature` enum, `FeatureSet` (`u64` words, `BASE/ALL/NONE` presets,
      `has`/`with`); the Feature↔bit map.
- [x] `(S)` `sysreg.rs`: `SystemReg(u16)` newtype over packed `op0<<14|op1<<11|CRn<<7|
      CRm<<3|op2`; `const fn from_fields`, `const fn packed`, `fn name() -> Option<&'static
      str>` (binary search over a sorted slice).
- [x] `(S)` `error.rs`: `DecodeError` plain enum (`Undefined`, `Unallocated`, `Reserved`,
      `FeatureRequired(Feature)`, …); `impl core::fmt::Display`; `impl std::error::Error` only
      under `cfg(feature = "std")`.
- [x] `(S)` Append-only-discriminant **snapshot test** for `Code`/`Mnemonic`/`Register`
      (golden file of `(variant, u16)`).

**Exit criteria:** public surface compiles on all three targets; `Operand`/`Instruction` size
+ Copy asserts pass; doc renders; snapshot test established.

---

## M2 — Hand-written ARM pseudocode helpers (`decode/bits.rs`)  `P0`

**Goal:** the shared, spec-faithful primitives every group decoder needs — bit-field
extraction plus the ARM ARM pseudocode functions, all hand-written, integer-only, zero-heap.
These are pinned by unit tests transcribed from the ARM ARM and cross-checked against the
oracles.

- [x] `(S)` Inline bit-field helpers: `bits(word, hi, lo)`, `bit(word, n)`, `sign_extend`,
      `zero_extend`, `replicate`, `highest_set_bit`, `ones`, `ror`/`lsl`/`lsr`/`asr` on fixed
      widths. No allocation, no panics.
- [x] `(M)` `decode_bit_masks(N, imms, immr, immediate, M) -> (wmask, tmask)` — **faithful to
      the ARM ARM** (compute `tmask` correctly; do **not** take binja's `tmask == wmask`
      shortcut at `pcode.c:88`). Record this as a documented intentional divergence vs the
      binja corpus. Pin with direct value tests transcribed from the spec, cross-checked
      against LLVM/binutils.
- [x] `(S)` `move_wide_preferred(sf, N, imms, immr) -> bool` (per the ARM ARM alias condition).
- [x] `(S)` `adv_simd_expand_imm(op, cmode, imm8) -> u64` (returns bits; no FP arithmetic).
- [x] `(S)` `vfp_expand_imm(imm8, width) -> u64` (returns bits; no FP arithmetic in decode).
- [x] `(S)` `decode_shift` / `decode_reg_extend` helpers (shift-type and extend-type mapping).
- [x] `(S)` `bitmask`/`concat`/`slice` bit utilities used by operand construction.
- [x] `(S)` Unit tests per helper, transcribed from the ARM ARM, with oracle cross-check for a
      sampled range of inputs.

**Exit criteria:** every helper has spec-transcribed unit tests that pass; the
`DecodeBitMasks` divergence vs binja is documented in the allow-list; no-alloc audit green
over the helpers.

---

## M3 — Base-ISA decode group: Data Processing — Immediate  `P0`

`decode/dp_imm.rs`. Top key `op0 = 100x`; sub-decoded by `bits[25:23]`. Hand-written matches
build the `Instruction` directly.

- [x] `(S)` ADR / ADRP (PC-rel): `imm21 = immhi[23:5] : immlo[30:29]`; ADRP page-shift;
      `Label` operand computed from `ip`.
- [x] `(S)` Add/Sub immediate (`sf|op|S`, `imm12@[21:10]`, shift-by-12 `@bit[22]`);
      CMP/CMN aliases via `Rd == ZR`.
- [x] `(M)` Logical immediate (`N|immr|imms`) via `decode_bit_masks` (spec-faithful);
      `MOV (bitmask)` alias via `Rn == ZR`; reserved-value handling.
- [x] `(S)` Move-wide MOVN/MOVZ/MOVK (`hw@[22:21]`); `sf==0 & hw[1]==1 → UNDEFINED`;
      `ImmShiftedMove` operand; MOV aliases via `move_wide_preferred`.
- [x] `(S)` Bitfield SBFM/BFM/UBFM with the LSL/LSR/ASR/SXTB/SXTH/SXTW/UXTB/UXTH/UBFIZ/
      BFXIL/SBFIZ/UBFX/SBFX alias family (alias conditions in code).
- [x] `(S)` Extract EXTR; ROR alias when `Rn == Rm`.
- [x] `(S)` Per-encoding unit tests transcribed from the ARM ARM; bring DP-Immediate green in
      the M9 harness across all three oracles.

**Exit criteria:** all DP-Immediate encodings decode + format correctly vs the oracles;
spec-derived unit tests pass.

---

## M4 — Base-ISA decode group: Branches, Exception & System  `P0`

`decode/branch_sys.rs`. Top key `op0 = 101x`; sub-decoded by `bits[31:29]`.

- [x] `(S)` Conditional branch `B.<cond>` (`imm19`); `NV == AL` rendering.
- [x] `(S)` Unconditional branch immediate B/BL (`imm26`; BL writes X30) → `Label`.
- [x] `(S)` Compare-and-branch CBZ/CBNZ (`imm19`); Test-and-branch TBZ/TBNZ
      (`b5:b40` bit number, `imm14`).
- [x] `(S)` Unconditional branch register BR/BLR/RET (PAuth variants reserved for M15).
- [x] `(S)` Exception generation SVC/HVC/SMC/BRK/HLT/DCPS1-3.
- [x] `(M)` System group:
  - [x] `(S)` HINT family NOP/YIELD/WFE/WFI/SEV/SEVL (by `CRm`/`op2`) + the HINT special-case
        in `decode/mod.rs`.
  - [x] `(S)` Barriers CLREX/DSB/DMB/ISB; `SB`, `DGH`, `CFINV`, `XAFLAG`, `AXFLAG`.
  - [x] `(M)` MSR/MRS with the 15-bit sysreg key → `SystemReg` (sorted-slice name lookup;
        generic `S<o0>_<o1>_c<crn>_c<crm>_<o2>` fallback). SYS/SYSL + `at`/`dc`/`cfp`/`tlbi`
        aliases.
- [x] `(S)` Per-encoding unit tests; bring Branch/Exception/System green in M9 (PAuth branch
      variants tracked in M15).

**Exit criteria:** all base Branch/Exception/System encodings correct vs the oracles.

---

## M5 — Base-ISA decode group: Loads & Stores  `P0`

`decode/ldst.rs`. Top key `op0 = x1x0`; subdivided by `bits[29:28] + [26:24]`. Addressing
modes captured in `Operand::MemImm{mode}` / `MemExt`.

- [x] `(S)` Load register literal LDR/LDRSW/PRFM (`imm19` word offset) → `Label`/`MemImm`.
- [x] `(M)` Load/Store exclusive LDXR/STXR/LDAXR/STLXR/LDXP/STXP (LSE CAS family in M14).
- [x] `(M)` Load/Store pair LDP/STP/LDNP/STNP (`imm7` scaled; offset / pre / post-index)
      → two regs + `MemImm`.
- [x] `(L)` Load/Store register, all sub-forms:
  - [x] `(S)` Unsigned scaled `imm12`.
  - [x] `(S)` Unscaled LDUR/STUR `imm9`.
  - [x] `(S)` Pre-index (`]!`) and post-index (`], #imm`).
  - [x] `(S)` Register offset with extend + scale (`MemExt`).
- [x] `(M)` Advanced-SIMD load/store **multiple** & **single** structures (LD1…LD4 / ST1…ST4)
      → `MultiReg{ regs, count, arr, lane }`; post-index immediate/register variants.
- [x] `(S)` Per-encoding unit tests; bring base Loads/Stores green in M9 (LSE atomics in M14).

**Exit criteria:** all base Loads/Stores encodings correct vs the oracles.

---

## M6 — Base-ISA decode group: Data Processing — Register  `P0`

`decode/dp_reg.rs`. Top key `op0 = x101`; decoded by `bits[28:24]`.

- [x] `(S)` Logical shifted register AND/ORR/EOR/ANDS/BIC/ORN/EON; MOV/MVN/TST aliases.
- [x] `(S)` Add/Sub shifted register (shift `00=LSL..10=ASR`); NEG/NEGS/CMP/CMN aliases.
- [x] `(S)` Add/Sub extended register (extend `000=UXTB..111=SXTX`, shift 0–4).
- [x] `(S)` Add/Sub with carry ADC/SBC; NGC/NGCS aliases; RMIF/SETF8/SETF16.
- [x] `(S)` Conditional compare CCMN/CCMP; conditional select CSEL/CSINC/CSINV/CSNEG
      (+ CSET/CSETM/CINC/CINV/CNEG aliases).
- [x] `(S)` DP 2-source UDIV/SDIV/LSLV/LSRV/ASRV/RORV/CRC32* (PACGA tracked w/ M15).
- [x] `(S)` DP 1-source RBIT/REV16/REV32/REV/CLZ/CLS (PACIA/AUTIA/XPAC tracked w/ M15).
- [x] `(S)` DP 3-source MADD/MSUB/SMADDL/UMADDL/SMULH/UMULH; MUL/MNEG/SMULL/UMULL aliases
      via `Ra == ZR`.
- [x] `(S)` Per-encoding unit tests; bring DP-Register green in M9 (PAuth 1-src/PACGA in M15).

**Exit criteria:** all base DP-Register encodings correct vs the oracles.

---

## M7 — Base-ISA decode group: Scalar FP & Advanced SIMD  `P0`

`decode/simd_fp/` (`scalar_fp`, `simd_arith`, `simd_data`, `crypto`). Top key `op0 = x111`;
decoded by `bit[28] + bits[27:24]`. The large NEON + scalar-FP base surface. Crypto / FP16 /
BF16 / modified-immediate detail split into their own milestones.

- [x] `(M)` Scalar FP compare/convert/DP-1src/DP-2src/DP-3src (FMOV/FCVT/FADD/FSUB/FMUL/
      FDIV/FMADD/FMSUB/FNMADD/FNMSUB/FABS/FNEG/FSQRT/FRINT*); `ftype` width selection.
- [x] `(M)` FP ↔ integer convert (FCVTZS/FCVTZU/SCVTF/UCVTF, fixed-point variants) and FP
      immediate (`vfp_expand_imm`, rendered via the FP-imm formatter helper).
- [x] `(L)` Advanced SIMD 3-same / 3-different / 2-reg-misc / across-lanes with
      `VectorArrangement` operands.
- [x] `(M)` Advanced SIMD copy / permute (TRN1/TRN2/UZP1/UZP2/ZIP1/ZIP2) / table (TBL/TBX) /
      extract (EXT).
- [x] `(M)` Scalar SIMD variants and indexed-element forms (`V0.s[2]` via
      `Operand::IndexedElement`).
- [x] `(S)` Per-encoding unit tests; bring scalar-FP + base Advanced-SIMD green in M9
      (modified-immediate in M12; FP16/BF16 in M13; crypto in M17).

**Exit criteria:** scalar-FP + base Advanced-SIMD encodings correct vs the oracles.

---

## M8 — Default zero-alloc formatter (UAL) + alias/preferred-disassembly  `P0`

**Goal:** `FmtFormatter` producing UAL output into a `core::fmt::Write` / `BufSink`, with the
ARM ARM preferred-disassembly (alias) rules applied in code and gated by
`FormatterOptions::aliases` (default on), mirroring the iced `Code` vs `Mnemonic` split.

- [x] `(M)` `format/mod.rs`: `Formatter` trait (object-safe: `format`, `format_mnemonic`,
      `format_operand`, `options`, `options_mut`), `FormatterOutput` sink trait
      (`write(&mut self, text: &str, kind: TokenKind)`), `FormatterOptions`, `TokenKind`,
      `SymbolResolver` + `SymbolResult`.
- [x] `(S)` Blanket `impl FormatterOutput for T: core::fmt::Write` (ignoring `kind`) so any
      `core::fmt::Write` works in `no_std`.
- [x] `(M)` `format/fmt_writer.rs`: `BufSink(&mut [u8])` with an **overflow flag** (never
      writes past the buffer); integer/hex emit without alloc; FP-imm decimal rendering helper
      (the only FP, isolated here).
- [x] `(L)` `FmtFormatter` single operand-dispatch path (no parallel string/token code):
  - [x] `(S)` Mnemonic emit with padding and `TokenKind::Mnemonic`.
  - [x] `(S)` Register + arrangement-spec (`.4s` full vs `.s` truncated per option).
  - [x] `(S)` Shifted register / shifted immediate (`, lsl #n`, `show_lsl_zero`).
  - [x] `(S)` Extended register (`, uxtw #n` / `sxtx` …).
  - [x] `(S)` Memory bracket fusion: `[base]`, `[base, #imm]`, `[base, #imm]!` (pre),
        `[base], #imm` (post-imm), `[base, Rm, ext #s]`, `[base], Rm` (post-reg).
  - [x] `(S)` Predicate qualifiers `/z` `/m`; SVE/SME decorators.
  - [x] `(S)` Sysreg name (sorted-slice lookup) with the generic `S<o0>_..._<o2>` fallback.
  - [x] `(S)` Signed-magnitude immediates, `#` prefix, configurable hex prefix.
  - [x] `(S)` `Label` rendered as an `Address` token (PC-relative target from `ip`).
- [x] `(M)` Apply alias / preferred-disassembly **in code** per the ARM ARM alias conditions:
      MOV (← ORR/ADD/MOVZ/MOVN/bitmask), CMP/CMN (← SUBS/ADDS), TST (← ANDS), NEG/NEGS,
      MUL/MNEG/SMULL/UMULL (← MADD/MSUB), LSL/LSR/ASR/ROR (← UBFM/SBFM/EXTR),
      UBFIZ/UBFX/SBFIZ/SBFX/BFI/BFXIL, NOP/YIELD/... (← HINT), CSET/CSETM/CINC/CINV/CNEG,
      NGC/NGCS, SXTB/SXTH/SXTW/UXTB/UXTH. Decode always yields the canonical `Code`; alias
      selection sets the resolved `Mnemonic` only. `aliases = false` yields canonical
      mnemonics.
- [x] `(S)` `FormatterOptions::default()` reproduces UAL output (aliases on, `use_sp_not_xzr`,
      separator spacing, first-operand index) so the oracles can match.
- [x] `(S)` No-alloc audit over the formatter via `BufSink`.

**Exit criteria:** `FmtFormatter` renders the base-ISA set to UAL matching the oracles;
`aliases=false` path sane; `BufSink` overflow verified; no-alloc audit green on the format
path.

---

## M9 — Multi-oracle differential harness  `P0`

**Goal:** the correctness gate — reusable for every later milestone. Correctness is defined
against the ARM ARM, cross-checked against **three independent oracles**: LLVM, GNU binutils,
and the binja corpus (as a development guide only).

- [x] `(M)` `tests/llvm_diff.rs`: re-disassemble sampled words with `llvm-mc --disassemble`,
      normalize, and compare against fARM64's default formatter output; skips if `llvm-mc` is
      absent. (binutils cross-check is ad hoc, not a committed test.)
- [x] `(M)` `tests/golden.rs`: parse `refs/arch-arm64-master/disassembler/test_cases.txt`
      **locally** as a guide (never shipped). Skip `//` lines but track the last-seen `//`
      group label for bucketing; split data lines into `insword_hex` + `expected`. Decode at a
      fixed `ADDRESS_TEST`; format with default options. Reports coverage + parity.
- [x] `(M)` Shared normalize pipeline: trim, collapse whitespace, strip trailing ` //`
      comments, expand `{zN.T-zM.T}` ranges, remove brace spaces, strip leading hex zeros,
      decimal→hex, float-normalize (`0.000000`→`0.0`), lowercase; plus token-level
      equivalences (`cs==hs`, `cc==lo`, `xN.d==xN`, `sp.d==sp`, signed/unsigned hex at 8/32/64).
- [x] `(M)` **Documented divergence allow-list** for intentional spec-vs-binja differences
      (e.g. `DecodeBitMasks` `tmask`; the `dgh↔hint`, `cfinv↔msr`, `at↔sys`, `dc↔sys`,
      `tlbi↔sys`, `axflag*` style preferred-mnemonic choices). Each entry cites the ARM ARM
      and the reason. The list is the *only* place a binja mismatch is tolerated; LLVM/binutils
      remain the spec-aligned oracles.
- [x] `(S)` Single parameterized test per oracle; collect failures **bucketed by
      encoding-group label**; report e.g. `LDP_*: 3/8 failed`.
- [x] `(S)` Wire as the **required CI gate** (base groups must be green; extension groups gated
      per their milestone) with a documented, shrinking allow-list of not-yet-implemented
      groups so CI stays green during rollout.

**Exit criteria:** harness runs against all three oracles; bucketed failure report works; CI
gate active; the spec-vs-binja divergence allow-list is documented and cited.

---

## M10 — Iterator ergonomics, errors, API polish  `P1`

- [x] `(S)` `Decoder::new`, `try_new`, `with_options`.
- [x] `(S)` `decode()` wrapper over `decode_into`; `can_decode` (≥4 bytes); `position` /
      `set_position`; `ip` / `set_ip` (keep `ip` consistent on seek); `last_error`.
- [x] `(S)` `impl IntoIterator for Decoder` (consuming) and `for &mut Decoder` (borrowing).
- [x] `(S)` `is_invalid`, `flow_control`, `next_ip` finalized.
- [x] `(S)` Doc the supported-target table and feature flags in `lib.rs`; document the
      no-alloc Tier A / alloc Tier B / std Tier C split.
- [x] `(S)` `#[cfg(feature = "alloc")] format_to_string(...)` convenience (strictly additive).

**Exit criteria:** ergonomic iteration works on all targets; docs describe the portability
tiers.

---

## M11 — InstructionInfo (flow/access) facility  `P1`

- [ ] `(M)` `info.rs`: `instruction_info(&Instruction) -> InstructionInfo` filling
      fixed-capacity inline arrays (`used_registers`, `used_memory`) — **no alloc** on the core
      path; return borrowed slices.
- [ ] `(M)` Per-encoding access metadata (read/write/cond/read-write) derived in the
      hand-written group decoders from each operand's role.
- [ ] `(S)` `flow_control()` mapping from `Code`/`Mnemonic`.
- [ ] `(S)` `#[cfg(feature = "alloc")] InstructionInfoFactory` (allocate-once / refill,
      iced-style).
- [ ] `(S)` `OpAccess` enum (`None/Read/Write/ReadWrite/CondRead/CondWrite`),
      `UsedRegister`, `UsedMemory`.

**Exit criteria:** flow/access correct for the base ISA; no-alloc audit green on the core info
path.

---

## M12 — Extension: Advanced-SIMD details & modified-immediate  `P1`

- [x] `(S)` MOVI/MVNI/FMOV (vector) via `adv_simd_expand_imm` (spec-faithful); `Msl` shift.
- [x] `(M)` Remaining indexed-element multiply/FMA forms; saturating/rounding variants;
      arrangement full-vs-truncated suffix edge cases.
- [x] `(S)` Gate via `FeatureSet` where the ARM ARM requires it (Advanced SIMD is base FP/NEON).
- [x] `(S)` Bring modified-immediate + remaining SIMD groups green in M9.

**Exit criteria:** Advanced-SIMD detail groups correct vs the oracles.

---

## M13 — Extension: FP16 / BF16  `P1`  (cargo: `fp16`, `bf16`)

- [ ] `(S)` FP16 scalar + vector ops; `ftype == 11` half-precision paths.
- [ ] `(S)` BF16 ops (BFCVT/BFDOT/BFMMLA/BFMLAL…).
- [ ] `(S)` `#[cfg(feature = "fp16")]` / `#[cfg(feature = "bf16")]` gating of the group-decoder
      arms + `Code` variants + name-table slices; `FeatureSet` runtime gate.
- [ ] `(S)` Groups green in M9; base-only build still omits these (size check).

**Exit criteria:** FP16/BF16 groups correct; compile-out verified.

---

## M14 — Extension: LSE (atomics)  `P1`  (cargo: `lse`)

- [ ] `(S)` CAS/CASA/CASL/CASAL, CASP family.
- [ ] `(S)` Atomic memory ops LDADD/LDCLR/LDEOR/LDSET/LDSMAX/LDSMIN/LDUMAX/LDUMIN/SWP
      (+ A/L/AL ordering variants) and the ST* aliases.
- [ ] `(S)` Feature gating (cargo `lse` + `FeatureSet`); groups green in M9.

**Exit criteria:** LSE atomics groups correct; compile-out verified.

---

## M15 — Extension: Pointer Authentication (PAuth)  `P1`  (cargo: `pauth`)

- [ ] `(S)` 1-source PACIA/PACIB/PACDA/PACDB/AUTIA/AUTIB/AUTDA/AUTDB/XPACI/XPACD + combined
      variants; PACGA (DP 2-source).
- [ ] `(S)` Authenticated branches BRAA/BRAB/BLRAA/BLRAB/RETAA/RETAB/ERETAA/ERETAB;
      authenticated loads LDRAA/LDRAB.
- [ ] `(S)` Feature gating (cargo `pauth` + `FeatureSet`); groups green in M9.

**Exit criteria:** PAuth groups correct; compile-out verified.

---

## M16 — Extension: MTE (memory tagging)  `P2`  (cargo: `mte`)

- [ ] `(S)` IRG, ADDG/SUBG, GMI, STG/STZG/ST2G/STZ2G/STGP, LDG, SUBP/SUBPS (CMPP alias).
- [ ] `(S)` Feature gating; groups green in M9.

**Exit criteria:** MTE groups correct; compile-out verified.

---

## M17 — Extension: Crypto (AES/SHA/SHA512/SM3/SM4)  `P2`  (cargo: `crypto`)

- [x] `(S)` AES (AESE/AESD/AESMC/AESIMC).
- [x] `(S)` SHA1 / SHA256 (SHA1C/SHA1P/SHA1M/SHA1H/SHA1SU0/SHA1SU1/SHA256H/SHA256H2/
      SHA256SU0/SHA256SU1).
- [x] `(S)` SHA512 / SM3 / SM4 families.
- [x] `(S)` Feature gating; groups green in M9.

**Exit criteria:** crypto groups correct; compile-out verified.

---

## M18 — Extension: SVE / SVE2  `P1`  (cargo: `sve`)  — corpus-dominant

Top key `op0 = 0b0010`; routed to the hand-written SVE decoder in `decode/sve/`
(`sve_int`/`sve_perm`/`sve_fp`/`sve_mem`) **only** when compiled (`sve` feature) and
`Feature::Sve` is accepted (else left `Invalid`). The binja corpus guide is dominated by these.

- [x] `(M)` SVE operand shapes: predicated `Z`/`P` operands with `pred_qual` `/z` `/m`,
      truncated arrangement suffixes, element-size math (`esize` from `size`/`tszh`).
- [x] `(L)` Predicated data-processing (REVD, ADD/SUB/AND/ORR predicated, …).
- [x] `(L)` Gather/scatter & contiguous loads/stores (LD1B/LD1H/…/ST1*) with the `mul vl`
      decorator and vector/scalar+immediate addressing.
- [x] `(M)` Predicate generation/manipulation (PTRUE/WHILELT/WHILE*/PFALSE/…).
- [x] `(L)` SVE2 additions (integer multiply-add, bit-permute, complex, narrowing, …).
- [x] `(S)` New pseudocode derivations → hand-written helpers in `decode/bits.rs`.
- [x] `(S)` cargo `sve` compile-out + `FeatureSet` runtime gate; the `{zN.T-zM.T}` range
      expansion in the harness is exercised heavily here.
- [x] `(M)` Bring SVE/SVE2 groups green incrementally; bucket-triage by group.

**Exit criteria:** SVE/SVE2 groups correct; compile-out verified; `001x` routes only when
enabled.

---

## M19 — Extension: SME / SME2  `P2`  (cargo: `sme`)

Top key `op0 = 0b0000` with `word<31>==1` (shares the reserved group with `UDF`); routed
from `decode_reserved` to `decode/sme/` only when compiled (`sme` feature) and `Feature::Sme`
is accepted.

- [ ] `(M)` `Operand::SmeTile{ tile, slice }` handling; ZA accumulator-array operands;
      `SliceIndicator`.
- [ ] `(M)` ZA tile load/store (LD1B_ZA / ST1*_ZA …) with slice indexing.
- [ ] `(M)` Outer products BFMOPA/FMOPA/SMOPA/UMOPA (and -S variants); MOVA to/from ZA.
- [ ] `(M)` SME2 multi-vector forms if present in the corpus guide.
- [ ] `(S)` cargo `sme` compile-out + `FeatureSet` runtime gate.
- [ ] `(S)` Bring SME/SME2 groups green in M9.

**Exit criteria:** SME/SME2 groups correct; compile-out verified.

---

## M20 — GNU formatter, fuzzing, benches, docs  `P2`

- [ ] `(S)` `format/gnu.rs`: `GnuFormatter` dialect behind `fmt-gnu` (same trait, objdump-style
      policy).
- [ ] `(M)` proptest suite (dev-dep, std/alloc test cfg):
  - [ ] `(S)` decode never panics and always advances exactly 4 bytes for any `u32`
        (`len() == 4` always).
  - [ ] `(S)` `op_count == #non-None operands`; `op_kind(n)` matches `op(n)` variant.
  - [ ] `(S)` formatter never panics; `BufSink` overflow reported, never written past buffer.
- [ ] `(M)` `no-alloc-audit` allocator test over a large word set through `decode_into` +
      `FmtFormatter` + `BufSink` (panics on any allocation = proof of zero heap).
- [ ] `(S)` Criterion benches: `decode_into` throughput; zero-alloc formatter throughput.
- [ ] `(S)` `cargo-fuzz` target: arbitrary 4-byte words → `decode_into` + format; assert no
      panic/UB; oracle parity where applicable.
- [ ] `(M)` Docs: `ARCHITECTURE.md` (the hand-written decode tree + pseudocode helpers;
      document the known spec-vs-binja divergences and the allow-list), `ENCODING.md`, rustdoc
      on the public surface.
- [ ] `(S)` `bench-only` counting-allocator feature wired and documented.

**Exit criteria:** GNU dialect passes a representative subset; proptest + no-alloc audit + fuzz
green; benches recorded; docs published.

---

## M21 — Optional xtask name/enum-table generator  `P1`  (convenience, off the critical path)

**Goal:** repurpose `xtask` as a host-only, optional generator for the **mechanical** lookup
tables only — the large `Code`/`Mnemonic` enums and the `&'static str` register / condition /
sysreg name tables — emitted from a curated, ARM-spec-derived dataset. It generates **no**
decode logic and parses **no** third-party C. The library stays hermetic (no build.rs).

- [ ] `(M)` Curate an ARM-spec-derived dataset (mnemonics, codes, register names, condition
      names, sysreg name↔encoding map) as committed input data for the generator.
- [ ] `(M)` `cargo xtask gen` emits `src/tables/names.rs` and the enum bodies as `// @generated`
      source with stable ordering, deterministic formatting, and a header citing the ARM-spec
      dataset (no third-party attribution).
- [ ] `(S)` Idempotent + diffable output; a `cargo xtask check` mode that verifies the
      committed generated files match a fresh run (CI guard against drift).
- [ ] `(S)` Keep `Code`/`Mnemonic`/`Register` discriminants **append-only**; the generator
      preserves existing ordering and appends new variants.

**Exit criteria:** `cargo xtask gen` reproduces the committed tables byte-for-byte; the snapshot
test still passes; the library builds with no xtask involvement.

---

## M22 — Encoder / round-trip  `P3`  ✅ done

The semantic encoder is implemented in `src/encode/` (`mod.rs` + `bits.rs` + a
per-group file mirroring the decode tree) and exposed as
`encode(&Instruction) -> Result<u32, EncodeError>` and `Instruction::encode`,
re-exported from the crate root. It rebuilds the word from **semantics only**
(`code`/`mnemonic`/operands/`ip`) and never reads `insn.word()`, so a passing
round-trip proves the decode is invertible. See
[API.md](./API.md#encoder-encode--encodeerror).

- [x] `(L)` Encoder covering **all decoder groups** (dp_imm / dp_reg /
      branch_sys / ldst / ldst_simd / simd_fp+crypto / sve / sme); inverse
      pseudocode helpers (`encode_bit_masks`, inverse `vfp_expand_imm` /
      `adv_simd_expand_imm`).
- [x] `(M)` Round-trip harness `tests/roundtrip.rs`: decode → `encode` →
      re-decode over the 42,289-case binja corpus.

**Exit criteria (met):** **100.00% semantic round-trip** — every attempted case
(42,144/42,144) re-decodes to the identical `Instruction`; 98.00% exact-word, the
~2% gap being documented don't-care / should-be-ones fields the decoder discards
(semantically lossless). See [VALIDATION.md](./VALIDATION.md#round-trip-encoder).
A future `ConstantOffsets`-style field-position accessor (for in-place patching)
remains optional and is not required by the round-trip.

---

## Definition of Done (project-level)

A milestone is **Done** only when **all** hold:

1. **Spec conformance.** Every encoding owned by the milestone decodes + formats per the ARM
   ARM, and is **green** under the M9 multi-oracle harness (LLVM + binutils as spec-aligned
   oracles; binja corpus as a guide with mismatches allowed **only** via the documented,
   cited divergence allow-list). The shared CI allow-list of not-yet-implemented groups shrinks
   by exactly those groups.
2. **Spec-derived unit tests.** Per-encoding unit tests transcribed from the ARM ARM pass for
   the milestone's encodings; new pseudocode helpers carry spec-transcribed value tests.
3. **Portability.** Hosted (`std`+`alloc`), `wasm32-unknown-unknown` (default), and
   `aarch64-unknown-none` (`-Z build-std=core`, default) all **build and test** green.
4. **Zero heap.** The `no-alloc-audit` allocator (panics on any allocation) passes over the
   milestone's decode + default-format path.
5. **Type budget.** `size_of::<Operand>() <= 16`, `size_of::<Instruction>() <= 112`, and the
   `Copy` witnesses for `Operand`/`Instruction` static asserts still pass.
6. **API stability.** The `Code`/`Mnemonic`/`Register` discriminant snapshot test passes
   (append-only); no renumbering; the public API surface is unchanged by decode work.
7. **No panics-as-control-flow.** All decode fall-throughs return a typed `DecodeError`; shipped
   code paths contain no reachable `panic!`/`unwrap`/`todo!`; `len()` is always 4.
8. **Lints/docs.** `cargo fmt --check`, `clippy -D warnings`, and public-item docs pass.

---

## Coverage targets

| Layer | Target | Measured by |
|-|-|-|
| Base ISA (DP-imm, branches/sys, ld/st, DP-reg, scalar-FP + base SIMD) | **100% spec-conformant** | M9 multi-oracle harness, bucketed report |
| Advanced-SIMD detail + modified-immediate | 100% group conformant | M9 |
| FP16 / BF16 / LSE / PAuth / MTE / Crypto | 100% of each group conformant | M9, per-feature CI job |
| SVE / SVE2 | 100% of group conformant (corpus-dominant guide) | M9 |
| SME / SME2 | 100% of group conformant | M9 |
| Oracle agreement | LLVM + binutils agree across the corpus; binja diffs only via cited allow-list | M9 final |
| Heap allocations on decode + default-format path | **0** | `no-alloc-audit` |
| Build targets | 3/3 green (hosted, wasm32, aarch64-none) | CI matrix |
| `size_of::<Operand>()` / `size_of::<Instruction>()` | ≤ 16 / ≤ 112 bytes | static asserts |
| Decode robustness | no panic / exactly +4 bytes for **all** 2^32 words | proptest + fuzz |

---

## Critical path (ordering rationale)

`M0 → M1 → M2` is strictly sequential: scaffold, then the public types, then the shared
pseudocode helpers (`decode/bits.rs`) every group decoder depends on. The base-ISA group
decoders **M3–M7** (DP-Immediate, Branch/Exception/System, Loads/Stores, DP-Register, Scalar
FP & Advanced SIMD) are hand-written and can proceed largely in parallel once M2 lands; the
default formatter + alias rules **M8** can be developed alongside them (each group needs
formatting to be testable). The multi-oracle harness **M9** should come up as early as the
first group is decoding so every subsequent group is validated against LLVM + binutils + the
binja guide from day one. **M10/M11** (API polish, InstructionInfo) are independent P1 work
once the base groups + formatter land. Extension milestones **M12–M19** each depend only on
M2 + M7 + M8 + M9 plus their own hand-written decode arms and can be scheduled by ROI:

> Base → FP/NEON → FP16 → LSE → PAuth → SME → MTE → BF16 → SVE/SVE2 → crypto

**M21** (the optional name/enum-table generator) is a P1 convenience that can be done anytime
after M1 — it is **not** on the critical path; the enums can be hand-maintained until then.
**M20** (polish) is last and non-blocking; **M22** (encoder / round-trip) is
complete — 100% semantic round-trip over the corpus.
