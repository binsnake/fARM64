# fARM64 — Public API

> Drafted public surface for the `fARM64` AArch64 disassembler.
> iced-x86-inspired, `#![no_std]` by default, zero-heap on the core decode and
> default-format paths.

This document specifies the **public** API as it is actually implemented. The
hand-written decode core (an original implementation derived from the *Arm
Architecture Reference Manual*) is an internal implementation detail that never
appears in these types. Everything below is a projection of that core.

> Import paths matter. The crate prelude re-exports `Decoder`, `Instruction`, the
> enums, `Feature`/`FeatureSet`, `DecodeError`, `OpKind`/`Operand`, `Register`,
> the `Code`/`Mnemonic` enums, and the **formatter traits** (`Formatter`,
> `FormatterOptions`, `FormatterOutput`, `SymbolResolver`, `SymbolResult`,
> `TokenKind`). The concrete formatters and the buffer sink live in the
> [`format`](#formatter-subsystem) module: use `fARM64::format::{FmtFormatter,
> BufSink}` (and `fARM64::format::GnuFormatter` under `fmt-gnu`).

---

## Table of contents

- [Design constraints that shape the API](#design-constraints-that-shape-the-api)
- [Crate features and portability tiers](#crate-features-and-portability-tiers)
- [`Decoder`](#decoder)
- [`Instruction`](#instruction)
- [`OpKind` and `Operand`](#opkind-and-operand)
- [`Register` family](#register-family)
- [`Code` and `Mnemonic`](#code-and-mnemonic)
- [Supporting enums](#supporting-enums-condition-shift-extend-arrangement)
- [`Feature` / `FeatureSet`](#feature--featureset)
- [`DecodeError`](#decodeerror)
- [Encoder (`encode` / `EncodeError`)](#encoder-encode--encodeerror)
- [Formatter subsystem](#formatter-subsystem)
- [`InstructionInfo` flow/access facility](#instructioninfo-flowaccess-facility)
- [End-to-end usage examples](#end-to-end-usage-examples)
- [iced-x86 features intentionally dropped or changed](#iced-x86-features-intentionally-dropped-or-changed)

---

## Design constraints that shape the API

Three rules constrain every signature below; read them before the types.

1. **Zero-heap core.** `Decoder`, `decode`/`decode_into`, every enum, every name
   table, the default `FmtFormatter`, and the core `InstructionInfo` path touch
   no allocator. There is no `alloc` dependency on these paths. `String`/`Vec`
   returns exist only behind the `alloc` feature and are always additive
   conveniences — never the primary API.
2. **`Instruction` is a `Copy` value type.** No internal pointers, no borrows,
   target size `<= 40` bytes (goal 32). You can store it in arrays, send it
   across threads, and `memcpy` it freely. The decoder never hands out a
   reference into its own buffer.
3. **Names are `&'static str`.** Registers, mnemonics, system registers, and
   arrangement suffixes all come from `const`/`static` tables. `name()` methods
   are `const fn` where the discriminant alone determines the string, and never
   allocate.

A64 is fixed-width: every instruction is exactly 4 bytes, little-endian,
4-byte-aligned, and `next_ip == ip + 4` always. The API drops anything x86 needs
for variable-length decoding (instruction length fields beyond a `const 4`,
prefixes, segment overrides, 16/32/64-bit mode selection).

---

## Crate features and portability tiers

| Feature | Default | Adds | Tier |
|-|-|-|-|
| *(none)* | yes | `no_std`, **no alloc**: `Decoder`, `Instruction`, all enums, `FmtFormatter`, `BufSink`, core `InstructionInfo` | A — always works |
| `alloc` | no | `format_to_string`, `String`/token-collecting sinks, `InstructionInfoFactory` | B |
| `std` | no | implies `alloc`; `std::error::Error for DecodeError`, std test/bench helpers | C |
| `fmt-gnu` | no | `GnuFormatter` objdump-style dialect | — |
| `fp16` `lse` `pauth` `sme` `mte` `bf16` `sve` `crypto` | no | compile-in the matching generated table slices and `Code`/`Register` variants | — |
| `no-alloc-audit` | no (test) | installs an allocation-panicking global allocator for the zero-heap proof | — |

Supported targets (CI-enforced from day one): `x86_64-*` (dev/host),
`wasm32-unknown-unknown`, `aarch64-unknown-none` (bare-metal, no CRT), and
generally any target providing `core`. Cargo features decide what is *compiled*;
the runtime [`FeatureSet`](#feature--featureset) decides what is *accepted*.
These two gates are independent.

```rust
#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]
#[cfg(feature = "alloc")]
extern crate alloc;
```

---

## `Decoder`

A borrowing, zero-allocation cursor over a byte slice. It is also an
`Iterator`. It never panics and never reads out of bounds.

```rust
pub struct Decoder<'a> { /* &'a [u8], cursor, ip, options, features */ }

impl<'a> Decoder<'a> {
    /// `data` is the code bytes; `ip` is the runtime address of `data[0]`
    /// (used to resolve PC-relative targets). Never panics.
    pub fn new(data: &'a [u8], ip: u64, options: DecoderOptions) -> Decoder<'a>;

    /// Fallible constructor for symmetry; validates option/feature consistency.
    pub fn try_new(
        data: &'a [u8],
        ip: u64,
        options: DecoderOptions,
    ) -> Result<Decoder<'a>, DecodeError>;

    /// Decode one instruction, advance the cursor by 4, return a `Copy` value.
    /// On error returns an instruction with `code() == Code::Invalid`;
    /// inspect `last_error()` for the reason.
    pub fn decode(&mut self) -> Instruction;

    /// Primary hot-loop method: decode directly into a caller-owned slot.
    /// `decode` is a thin wrapper that calls this and returns `*out`.
    pub fn decode_into(&mut self, out: &mut Instruction);

    /// `>= 4` bytes remain.
    pub fn can_decode(&self) -> bool;

    /// Byte cursor within the slice.
    pub fn position(&self) -> usize;
    /// Seek; keeps `ip` consistent with the new position.
    pub fn set_position(&mut self, pos: usize);

    /// Address of the next instruction to decode.
    pub fn ip(&self) -> u64;
    pub fn set_ip(&mut self, ip: u64);

    /// `DecodeError::None` after a success, otherwise the failure reason of the
    /// most recent `decode`/`decode_into`.
    pub fn last_error(&self) -> DecodeError;

    /// The options this decoder was created with (carries the active
    /// `FeatureSet`).
    pub fn options(&self) -> &DecoderOptions;
}
```

`set_ip` (above) sets the decode address without moving the byte cursor;
`set_position` moves the cursor and keeps `ip` consistent. There is no
`with_ip`/`options_mut`/`features` convenience — construct with `new`/`try_new`
and read `options().features`.

### Iteration

`Decoder` implements `IntoIterator` both by value (consuming) and by `&mut`
(borrowing). Iteration stops when fewer than 4 bytes remain. Iterating yields
every decode result including invalids, so check `is_invalid()` /
`Decoder::last_error()` if you need to distinguish.

```rust
impl<'a> IntoIterator for Decoder<'a> {
    type Item = Instruction;
    type IntoIter = DecoderIntoIter<'a>;
    fn into_iter(self) -> Self::IntoIter;
}

impl<'a, 'd> IntoIterator for &'d mut Decoder<'a> {
    type Item = Instruction;
    type IntoIter = DecoderIter<'a, 'd>;
    fn into_iter(self) -> Self::IntoIter;
}

pub struct DecoderIntoIter<'a> { /* ... */ }
pub struct DecoderIter<'a, 'd> { /* ... */ }
impl<'a> Iterator for DecoderIntoIter<'a> { type Item = Instruction; /* ... */ }
impl<'a, 'd> Iterator for DecoderIter<'a, 'd> { type Item = Instruction; /* ... */ }
```

### `DecoderOptions`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecoderOptions {
    /// Architecture extensions accepted at decode time. Default: `FeatureSet::ALL`
    /// (of the compiled-in extensions), so by default you decode everything that
    /// was compiled. Narrow it to reject extension encodings as
    /// `DecodeError::FeatureRequired`.
    pub features: FeatureSet,
}

impl Default for DecoderOptions {
    fn default() -> Self;
}
```

---

## `Instruction`

A self-contained `Copy` value. All accessors are infallible; out-of-range
operand indices return the `None`-shaped value (`Operand::None`,
`Register::None`, `0`).

It derives `PartialEq` but **not** `Eq`/`Hash`, because the inline operand array
can hold an `Operand::FpImm(f32)` payload (no total order / hash). For a total
key, use `(word(), ip())`.

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Instruction {
    /* word: u32, ip: u64, code: Code, mnemonic: Mnemonic,
       op_count: u8, flags: u8, operands: [Operand; MAX_OPERANDS]
       — fields are private */
}

impl Instruction {
    /// Encoding-level identity (one variant per ARM encoding row).
    pub const fn code(&self) -> Code;
    /// Width-/encoding-independent operation, alias-resolved by the decoder when
    /// the encoding has a preferred-disassembly alias.
    pub const fn mnemonic(&self) -> Mnemonic;

    /// The raw 32-bit instruction word (little-endian host value).
    pub const fn word(&self) -> u32;

    /// Always 4 on A64.
    pub const fn len(&self) -> usize;
    pub const fn is_empty(&self) -> bool; // always false; present for lint parity

    /// Address of this instruction.
    pub const fn ip(&self) -> u64;
    /// `ip() + 4`.
    pub const fn next_ip(&self) -> u64;

    /// Number of meaningful operands (0..=MAX_OPERANDS).
    pub const fn op_count(&self) -> usize;
    /// Discriminant of operand `n`; `OpKind::None` if `n >= op_count()`.
    pub fn op_kind(&self, n: usize) -> OpKind;
    /// The rich operand value; `Operand::None` if `n >= op_count()`.
    pub fn op(&self, n: usize) -> Operand;

    // ---- iced-style fast typed accessors (no enum match at the call site) ----

    /// Register of a register-shaped operand; `Register::None` otherwise.
    pub fn op_register(&self, n: usize) -> Register;
    /// Immediate value of operand `n` as `u64` (signed/label values reinterpreted
    /// via `as u64`; logical/unsigned returned directly); `0` otherwise.
    pub fn op_immediate(&self, n: usize) -> u64;

    // ---- flow / classification ----

    pub fn is_invalid(&self) -> bool;
    /// Control-flow class, derived from the (alias-resolved) `Mnemonic` (with the
    /// one `Code::BCond` disambiguation for conditional `B.<cond>`).
    pub fn flow_control(&self) -> FlowControl;
    /// NZCV write behaviour.
    pub fn set_flags(&self) -> FlagEffect;
}
```

To read a memory operand, match `op(n)` against `Operand::MemImm` / `MemExt`
(and the SVE `SveMem` variant) directly — there is no flattened `MemOperand`
helper. The extension a given encoding belongs to is available via
`insn.code().feature()` (and `insn.code().is_base()`).

---

## `OpKind` and `Operand`

`OpKind` is the discriminant view; call it first, then the matching typed
accessor (the iced pattern). `Operand` is the fully-typed, safe value.

`OpKind` and `Operand` are `#[non_exhaustive]` and carry one discriminant per
variant. `Operand` derives `PartialEq` (not `Eq`/`Hash`, due to `FpImm(f32)`).

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OpKind {
    None,
    Register,
    ImmUnsigned,
    ImmSigned,
    ImmLogical,      // bitmask-immediate (decode_bit_masks result)
    ImmShiftedMove,  // MOVZ/MOVN/MOVK imm16 << (16*hw)
    ImmShiftedMsl,   // MOVI/MVNI ", msl #amt" form
    FpImm,
    ShiftAmount,
    Label,           // PC-relative target address
    Cond,
    SysReg,
    SysOp,           // barrier/PSTATE/IC/DC/AT/TLBI keyword operand
    MemImm,          // base + signed offset, optional pre/post index
    MemExt,          // base + extended/shifted index register
    MultiReg,        // {v0.16b - v3.16b} structure lists
    IndexedElement,  // v0.s[2] style indexed element
    SmeTile,         // ZA tile + slice
    ImplSpec,        // raw bytes for the rare un-modelled forms
    SvePattern,      // SVE element-count pattern (pow2/vl1../all)
    SveMul,          // SVE "mul #n" multiplier modifier
    ImmSignedDec,    // SVE signed immediate, mixed-radix rendering
    SveMem,          // SVE load/store/prefetch addressing
    SmeTileSlice,    // SME ZA tile-slice / whole-array operand
}
```

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Operand {
    None,

    /// A single register, possibly with an arrangement, lane index, an
    /// in-operand shift/extend, or an SVE predicate qualifier. Optionals are
    /// `None` when not present — no "is this field valid" booleans.
    Reg {
        reg: Register,
        arr: Option<VectorArrangement>,
        lane: Option<u8>,
        shift: Option<(ShiftType, u8)>,
        extend: Option<ExtendType>,
        pred: Option<PredQual>,
    },

    ImmUnsigned(u64),
    ImmSigned(i64),
    /// Logical/bitmask immediate already expanded to its full value.
    ImmLogical(u64),
    /// MOVZ/MOVN/MOVK shifted move immediate.
    ImmShiftedMove { imm: u16, lsl: u8 },
    /// MOVI/MVNI mask-shift-left (`, msl #amt`) form.
    ImmShiftedMsl { imm: u16, msl: u8 },
    /// Floating-point immediate, reconstructed via bit-cast (no FP math).
    FpImm(f32),
    ShiftAmount(u8),
    /// PC-relative target, already resolved to an absolute address.
    Label(u64),
    Cond(Condition),
    SysReg(SystemReg),
    /// Fixed system-instruction keyword (barrier option / PSTATE field / BTI
    /// target / `cN` / IC/DC/AT/TLBI/CFP/CPP/DVP op name).
    SysOp(SysToken),

    /// `[base]`, `[base, #imm]`, `[base, #imm]!`, `[base], #imm`.
    MemImm { base: Register, imm: i64, mode: MemIndexMode },
    /// `[base, index, extend #shift]`.
    MemExt { base: Register, index: Register, extend: ExtendType, shift: u8 },

    /// Structure list, e.g. `{v0.16b, v1.16b, v2.16b}`.
    MultiReg { regs: [Register; 4], count: u8,
               arr: Option<VectorArrangement>, lane: Option<u8> },
    /// Indexed element with a register index.
    IndexedElement { reg: Register, arr: Option<VectorArrangement>,
                     index: Register, imm: i64 },
    /// SME `za`/`zaN` tile with a slice indicator.
    SmeTile { tile: u16, slice: SliceIndicator },

    /// Escape hatch: raw bytes for forms not yet modelled structurally.
    ImplSpec([u8; 5]),

    // ---- SVE / SME extensions ----
    /// SVE element-count pattern (`pow2`/`vl1..vl256`/`mul3`/`mul4`/`all`/`#n`).
    SvePattern(u8),
    /// SVE `mul #n` multiplier modifier (1..=16).
    SveMul(u8),
    /// SVE signed immediate (positive hex, negative decimal — see ENCODING.md).
    ImmSignedDec(i64),
    /// SVE load/store/prefetch addressing (mode tags the sub-form).
    SveMem { base: Register, offset: Register, arr: Option<VectorArrangement>,
             extend: ExtendType, imm: i32, amount: u8, mode: SveMemMode },
    /// SME ZA tile-slice (`z0v.b[w14, #5]`) or whole-array (`za[w12, #1]`).
    SmeTileSlice { reg: Register, slice: SliceIndicator,
                   arr: Option<VectorArrangement>, sel: Register,
                   imm: i16, has_imm: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MemIndexMode { Offset, PreIndex, PostImm, PostReg }

/// SVE predicate qualifier: `/z` (zeroing) or `/m` (merging).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PredQual { None, Zeroing, Merging }

/// SME tile slice direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SliceIndicator { None, Horizontal, Vertical }

/// Which SVE addressing sub-form `Operand::SveMem` carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SveMemMode {
    ScalarImmMulVl, ScalarImm, ScalarImmDec, VecImm,
    VecScalar, ScalarVec, VecVec,
}
```

`SysToken` (in `fARM64::sysop`) is a compact index into the `&'static str` table
of system-instruction keyword operands.

---

## `Register` family

Register-31 ambiguity is resolved at decode time: a register is stored as
`Sp`/`Wsp`/`Xzr`/`Wzr`, never as raw "31". Callers never see the raw number.

```rust
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u16)]
#[non_exhaustive]
pub enum Register {
    None,
    // 32-bit GP
    W0, W1, /* .. */ W30, Wzr, Wsp,
    // 64-bit GP
    X0, X1, /* .. */ X30, Xzr, Sp,
    // scalar FP/SIMD views
    B0, /* .. */ B31,
    H0, /* .. */ H31,
    S0, /* .. */ S31,
    D0, /* .. */ D31,
    Q0, /* .. */ Q31,
    // vector
    V0, /* .. */ V31,
    // SVE scalable vector / predicate
    Z0, /* .. */ Z31,
    P0, /* .. */ P15,
    // prefetch pseudo-operand register class
    Pf0, /* .. */ Pf31,
}

impl Register {
    /// Width of the value held: 32/64/128 for GP/SIMD views, `0` for the
    /// scalable Z/P/prefetch slots.
    pub const fn width_bits(self) -> u16;
    /// Architectural number 0..=31 (or 0..=15 for P). `Sp`/`Xzr` -> 31, etc.
    pub const fn number(self) -> u8;
    /// 64-bit GP view of this register (`W3 -> X3`, `Wzr -> Xzr`).
    pub const fn as_x(self) -> Register;
    /// 32-bit GP view of this register (`X3 -> W3`, `Sp -> Wsp`).
    pub const fn as_w(self) -> Register;
    pub const fn is_simd(self) -> bool;   // B/H/S/D/Q/V
    pub const fn is_sve(self) -> bool;    // Z/P
    /// Register class of this register.
    pub const fn class(self) -> RegClass;
    /// Lowercase canonical name from the const table; never allocates.
    pub const fn name(self) -> &'static str;
}

/// Resolves the ARM ARM's per-operand SP-vs-ZR choice for reg 31.
pub const fn gp_register(use_sp: bool, width: RegWidth, n: u8) -> Register;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RegClass { None, Gp, ScalarFp, Vector, Sve, Predicate, Prefetch }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RegWidth { W32, X64 }
```

`Register` includes prefetch pseudo-register slots `Pf0..Pf31` (the prefetch
operation class), in addition to GP (`W`/`X`/`Wsp`/`Sp`/`Wzr`/`Xzr`), scalar
`B/H/S/D/Q`, vector `V`, SVE `Z`, and predicate `P`.

---

## `Code` and `Mnemonic`

Two enums, in the iced style. `Code` is encoding-level identity (drives
dispatch). `Mnemonic` is the human-facing operation. Both are
`#[repr(u16)] #[non_exhaustive]` with an append-only discriminant policy so
adding ARM revisions never breaks downstream `match`es. `Code`, its
`mnemonic()`/`feature()` accessors, and the variant docs are emitted together by
the in-crate `codes!` macro (one declarative row per encoding), so they cannot
drift.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
#[non_exhaustive]
pub enum Code {
    Invalid,
    BCond, BUncond, Cbz64, Tbnz, BlImm, Blr, Br, Ret, Udf,
    // … one variant per recognized ARM ARM encoding row.
}

impl Code {
    /// Operation this encoding maps to.
    pub const fn mnemonic(self) -> Mnemonic;
    /// Gating extension (`Feature::Base` for the base ISA).
    pub const fn feature(self) -> Feature;
    /// True if this encoding is part of the base ISA (no feature gate).
    pub const fn is_base(self) -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
#[non_exhaustive]
pub enum Mnemonic {
    Invalid,
    Add, Sub, Ldr, Str, B, Bl, Movz, Revd,
    // … one variant per operation name.
}

impl Mnemonic {
    /// Lowercase canonical name from the const table; never allocates.
    pub const fn name(self) -> &'static str;
}
```

Why both: `Code` lets you dispatch the exact encoding; `Mnemonic` lets you ask
high-level questions (`insn.mnemonic() == Mnemonic::B`) without matching
thousands of `Code` arms. Alias selection (`MOV`<-`ORR`, `CMP`<-`SUBS`,
`MUL`<-`MADD`, `LSL`/`LSR`/`ASR`<-`UBFM`/`SBFM`, `NOP`<-`HINT`, …) sets
`mnemonic()`; `code()` always stays canonical. Aliasing is toggled by
[`FormatterOptions::aliases`](#formatteroptions) (default on).

---

## Supporting enums (condition, shift, extend, arrangement)

```rust
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Condition { Eq, Ne, Cs, Cc, Mi, Pl, Vs, Vc, Hi, Ls, Ge, Lt, Gt, Le, Al, Nv }

impl Condition {
    /// ARM preferred spelling (`cs`/`cc`); the differential comparator also
    /// accepts `hs`/`lo`.
    pub const fn name(self) -> &'static str;
    pub const fn invert(self) -> Condition;
}

/// True shifts only. Extend types live in `ExtendType` for type clarity.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ShiftType { None, Lsl, Lsr, Asr, Ror, Msl }

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ExtendType { None, Uxtb, Uxth, Uxtw, Uxtx, Sxtb, Sxth, Sxtw, Sxtx }

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum VectorArrangement {
    None,
    V8B, V16B, V4H, V8H, V2S, V4S, V1D, V2D,
    // scalable element specifiers for SVE/SME
    B, H, S, D, Q,
}

impl VectorArrangement {
    pub const fn element_bits(self) -> u16;
    pub const fn element_count(self) -> u8;     // 0 for scalable forms
    /// `.4s` when `full`, `.s` when not (SVE truncated suffix).
    pub const fn suffix(self, full: bool) -> &'static str;
}
```

---

## `Feature` / `FeatureSet`

`FeatureSet` carries two `u64` words: `features0` gates decode-time structural
admission and `features1` gates pseudocode-time behaviour, kept separate because
the ARM ARM treats those questions independently. Both fields are public.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
#[non_exhaustive]
pub enum Feature {
    Base, Fp16, Bf16, Lse, PAuth, Mte, Sve, Sme, Crypto,
    Tme, Trf, Wfxt, Frintts,
    // … completed by codegen
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FeatureSet {
    pub features0: u64, // decode-time admission
    pub features1: u64, // pcode-time availability
}

impl FeatureSet {
    /// Base ISA only (both words zero).
    pub const BASE: Self;
    /// Every extension accepted (both words all-ones).
    pub const ALL: Self;
    /// Alias of `BASE`.
    pub const NONE: Self;

    /// `true` if `f` is present (`Base` is always present). Checks `features0`.
    pub fn has(self, f: Feature) -> bool;
    /// A copy with `f` enabled in both words.
    pub fn with(self, f: Feature) -> Self;
}

impl Default for FeatureSet {
    /// `FeatureSet::ALL` — decode everything out of the box.
    fn default() -> Self;
}
```

---

## `DecodeError`

Plain enum covering the ARM ARM decode outcomes (reserved / unallocated /
UNDEFINED, `SEE`-elsewhere redirections, constraint violations), plus a Rust-side
`FeatureRequired`. `Display` is provided in `core`; `std::error::Error` is
`std`-gated.

```rust
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DecodeError {
    None,             // DECODE_STATUS_OK
    Reserved,         // -1
    Unmatched,        // -2
    Unallocated,      // -3
    Undefined,        // -4
    EndOfInstruction, // -5
    Lost,             // -6
    Unreachable,      // -7
    AssertFailed,     // -8
    ErrorOperands,    // -9
    /// The encoding was structurally valid but its extension was not in the
    /// active `FeatureSet`.
    FeatureRequired(Feature),
}

impl core::fmt::Display for DecodeError { /* ... */ }
#[cfg(feature = "std")]
impl std::error::Error for DecodeError {}
```

`EndOfInstruction` is accepted (treated as success) when the encoding is a
`HINT`, per the ARM ARM's HINT special case.

---

## Encoder (`encode` / `EncodeError`)

The inverse of decode: reconstruct the raw 32-bit little-endian A64 word from
the **semantics** of an `Instruction` — its `code()`, `mnemonic()`, operands and
`ip()`. The encoder **never reads `insn.word()`**; it rebuilds the word purely
from the public projection, which is precisely what proves the decode is
invertible. Like the decoder it is `no_std`, zero-alloc and **total**: it never
panics, surfacing every failure as an `EncodeError`. Dispatch is on
`Instruction::code()` (the canonical encoding identity) routed to a per-group
encoder; all decoder groups are covered (DP-immediate, DP-register,
branch/system, loads & stores, SIMD ld/st, scalar-FP / Advanced-SIMD / crypto,
SVE, SME).

Both a free function and an inherent method are provided, re-exported from the
crate root (`use fARM64::{encode, EncodeError}`):

```rust
/// Encode `insn` back into its 32-bit little-endian A64 word.
/// Reconstructs from semantics only (code/mnemonic/operands/ip); never reads
/// `insn.word()`. Returns `EncodeError` for anything it cannot encode; never panics.
pub fn encode(insn: &Instruction) -> Result<u32, EncodeError>;

impl Instruction {
    /// Convenience method; identical to `encode(self)`.
    pub fn encode(&self) -> Result<u32, EncodeError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EncodeError {
    /// This `Code` / group is not implemented by the encoder yet.
    Unsupported,
    /// An operand was missing, of the wrong kind, or inconsistent with the
    /// encoding (register out of range, mnemonic/operand-layout mismatch, …).
    InvalidOperand,
    /// An immediate has no valid encoding in the instruction's field(s) (not a
    /// representable logical/bitmask immediate, out-of-range shift, PC-relative
    /// target out of reach, …).
    InvalidImmediate,
    /// The instruction is the invalid sentinel (`Code::Invalid`).
    Invalid,
}

impl core::fmt::Display for EncodeError { /* ... */ }
#[cfg(feature = "std")]
impl std::error::Error for EncodeError {}
```

### Round-trip example

```rust
use fARM64::{Decoder, DecoderOptions};

// `ADD W0, W1, #1`, little-endian.
let code = [0x20, 0x04, 0x00, 0x11];
let mut dec = Decoder::new(&code, 0, DecoderOptions::default());
let insn = dec.decode();

let word: u32 = insn.encode().expect("encodable");
assert_eq!(word, u32::from_le_bytes(code));
```

### Semantic round-trip vs exact word

The encoder's correctness contract is **semantic**: `encode(insn)` is guaranteed
to re-decode to an `Instruction` value-identical to `insn` (same `code`,
`mnemonic`, operands, flow/flags). It is *not* guaranteed to reproduce the exact
input word bit-for-bit. The two agree for the vast majority of encodings, but a
small set of architectural don't-care / should-be-one fields are discarded by the
decoder (they carry no semantics), so the encoder emits the canonical value
instead — value-identical and re-decoding identically, but not the same raw bits.
These fields are: `SMULH`/`UMULH` `Ra`, load/store-exclusive `Rs`/`Rt2`, `IC`
`Rt`, `DUP` (general) index bits, and `FCMP`/`FCMPE #0.0` `Rm`. See
[VALIDATION.md](./VALIDATION.md#round-trip-encoder) for the measured numbers
(100.00% semantic, 98.00% exact-word over the corpus).

So: compare `encode(decode(w))` to the *re-decoded instruction*, not blindly to
`w`, when an exact-bit match is not required. If a caller does need the original
bits, `insn.word()` still carries them.

---

## Formatter subsystem

Output is push-based into a sink, so the default path allocates nothing. Any
`core::fmt::Write` is a sink via a blanket impl; `BufSink` wraps a fixed
`&mut [u8]` with overflow tracking. `String` is a token-collecting sink only
under `alloc`.

```rust
/// Object-safe. Zero-alloc: all text flows through `out`.
pub trait Formatter {
    fn format(&self, insn: &Instruction, out: &mut dyn FormatterOutput);
    fn format_mnemonic(&self, insn: &Instruction, out: &mut dyn FormatterOutput);
    fn format_operand(&self, insn: &Instruction, n: usize, out: &mut dyn FormatterOutput);
    fn options(&self) -> &FormatterOptions;
    fn options_mut(&mut self) -> &mut FormatterOptions;
}

/// The token / output sink.
pub trait FormatterOutput {
    fn write(&mut self, text: &str, kind: TokenKind);
}

/// Blanket impl: any `core::fmt::Write` is a (text-only) sink (kind ignored).
impl<W: core::fmt::Write> FormatterOutput for W { /* ... */ }

/// Fixed-buffer sink with overflow tracking; never writes past `buf`.
pub struct BufSink<'b> { /* buf: &'b mut [u8], len: usize, overflow: bool */ }
impl<'b> BufSink<'b> {
    pub fn new(buf: &'b mut [u8]) -> Self;
    pub fn as_str(&self) -> &str;       // the written prefix (valid UTF-8)
    pub fn overflowed(&self) -> bool;
    pub fn len(&self) -> usize;
}
impl<'b> FormatterOutput for BufSink<'b> { /* ... */ }

#[cfg(feature = "alloc")]
impl FormatterOutput for alloc::string::String { /* ignores kind */ }
```

```rust
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TokenKind {
    Mnemonic, Register, Number, Float, Address,
    Punctuation, BeginMemory, EndMemory, OperandSeparator,
    Decorator, SysReg,
}
```

### `FormatterOptions`

Defaults reproduce canonical ARM UAL output (the spec's preferred disassembly).

```rust
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FormatterOptions {
    /// Emit preferred-disassembly aliases (MOV/CMP/MUL/LSL/NOP …). Default true.
    pub aliases: bool,
    pub uppercase_mnemonics: bool,
    pub uppercase_registers: bool,
    /// Render reg-31 as `sp`/`wsp` instead of `xzr`/`wzr` where applicable.
    pub use_sp_not_xzr: bool,
    /// e.g. `"0x"`.
    pub hex_prefix: &'static str,
    pub signed_immediates: bool,
    /// Print `lsl #0` explicitly (off by default to match UAL).
    pub show_lsl_zero: bool,
    pub space_after_operand_separator: bool,
    /// Mnemonic field padding for column alignment.
    pub first_operand_char_index: u8,
}

impl Default for FormatterOptions {
    /// Reproduces canonical ARM UAL output.
    fn default() -> Self;
}
```

### Default formatter and convenience

```rust
/// Default no_std, zero-alloc UAL formatter — the differential-oracle target.
pub struct FmtFormatter { /* options */ }
impl FmtFormatter {
    pub fn new() -> Self;
    pub fn with_options(opts: FormatterOptions) -> Self;
}
impl Formatter for FmtFormatter { /* ... */ }
impl Default for FmtFormatter { fn default() -> Self; }

/// objdump/GNU dialect, same trait, alternate policy.
#[cfg(feature = "fmt-gnu")]
pub struct GnuFormatter { /* ... */ }
#[cfg(feature = "fmt-gnu")]
impl Formatter for GnuFormatter { /* ... */ }

/// Strictly additive convenience; never on the default path.
#[cfg(feature = "alloc")]
pub fn format_to_string(fmt: &dyn Formatter, insn: &Instruction) -> alloc::string::String;
```

### Symbol resolution

The `SymbolResolver` / `SymbolResult` types are defined and re-exported (alloc-
free: `SymbolResult` borrows a name). The default `FmtFormatter` does not yet
expose a `format_symbolic` entry point — symbol resolution is a defined hook
awaiting formatter wiring.

```rust
pub trait SymbolResolver {
    fn symbol(
        &mut self,
        insn: &Instruction,
        operand: usize,
        address: u64,
    ) -> Option<SymbolResult<'_>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SymbolResult<'a> {
    /// The symbol name (borrowed; no allocation).
    pub name: &'a str,
    /// Offset from the symbol base.
    pub offset: i64,
}
```

---

## `InstructionInfo` flow/access facility

The core path fills fixed-capacity inline arrays — no allocation. An
`alloc`-gated factory mirrors iced's allocate-once/refill pattern.

> Status: the types below are defined and re-exported, but
> `info::instruction_info` is currently a stub (`todo!()`) and will panic if
> called. `Instruction::flow_control()` and `Instruction::set_flags()` (on the
> `Instruction` itself) are the implemented flow/flag accessors today. The
> `instruction_info` free function lives in the `fARM64::info` module (it is not
> in the crate prelude).

```rust
#[derive(Debug, Clone, Copy)]
pub struct InstructionInfo { /* fixed inline arrays + counts */ }

/// Zero-alloc: returns owned fixed-capacity info. (Not yet implemented.)
pub fn instruction_info(insn: &Instruction) -> InstructionInfo;

impl InstructionInfo {
    pub fn used_registers(&self) -> &[UsedRegister];
    pub fn used_memory(&self) -> &[UsedMemory];
    pub fn flow_control(&self) -> FlowControl;
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct UsedRegister { pub register: Register, pub access: OpAccess }

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct UsedMemory {
    pub base: Register,
    pub index: Register,
    pub offset: i64,
    pub access: OpAccess,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OpAccess { None, Read, Write, ReadWrite, CondRead, CondWrite }

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FlowControl {
    Next, UnconditionalBranch, ConditionalBranch,
    IndirectBranch, Call, IndirectCall, Return, Exception,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FlagEffect { None, Sets, SetsNormal, SetsFloat }

/// Allocate-once / refill, mirroring iced. `alloc` only.
#[cfg(feature = "alloc")]
pub struct InstructionInfoFactory { /* ... */ }
#[cfg(feature = "alloc")]
impl InstructionInfoFactory {
    pub fn new() -> Self;
    pub fn info(&mut self, insn: &Instruction) -> &InstructionInfo;
}
```

---

## End-to-end usage examples

### 1. Decode a buffer and print (simplest path, `alloc`)

```rust
use fARM64::{Decoder, DecoderOptions};
use fARM64::format::{format_to_string, FmtFormatter};

# #[cfg(feature = "alloc")]
fn dump(code: &[u8], base: u64) {
    let fmt = FmtFormatter::new();
    let mut dec = Decoder::new(code, base, DecoderOptions::default());
    for insn in &mut dec {
        // e.g. "add w20, w20, w29, uxtb" / "revd z29.q, p4/m, z29.q"
        println!("{:#018x}  {}", insn.ip(), format_to_string(&fmt, &insn));
    }
}
```

### 2. Zero-alloc `decode_into` + `BufSink` loop (no_std / bare-metal)

```rust
use fARM64::{Decoder, DecoderOptions, Instruction};
use fARM64::format::{BufSink, FmtFormatter, Formatter};

/// Decode and format every instruction in `code` without touching an allocator.
fn each_line(code: &[u8], base: u64, mut sink: impl FnMut(u64, &str, bool)) {
    let fmt = FmtFormatter::new();
    let mut dec = Decoder::new(code, base, DecoderOptions::default());
    let mut insn = Instruction::default();
    let mut buf = [0u8; 128];

    while dec.can_decode() {
        dec.decode_into(&mut insn);
        let mut out = BufSink::new(&mut buf);
        if insn.is_invalid() {
            // dec.last_error() carries the DecodeError::* reason
        }
        fmt.format(&insn, &mut out);
        sink(insn.ip(), out.as_str(), out.overflowed());
    }
}
```

### 3. Custom formatter options (uppercase, explicit `lsl #0`, no aliases)

```rust
use fARM64::{Decoder, DecoderOptions, FormatterOptions};
use fARM64::format::{BufSink, FmtFormatter, Formatter};

fn raw_uppercase(word: u32, ip: u64) -> ([u8; 64], usize) {
    let opts = FormatterOptions {
        aliases: false,            // show ORR instead of MOV, SUBS instead of CMP
        uppercase_mnemonics: true,
        show_lsl_zero: true,
        ..FormatterOptions::default()
    };
    let fmt = FmtFormatter::with_options(opts);

    let bytes = word.to_le_bytes();
    let insn = Decoder::new(&bytes, ip, DecoderOptions::default()).decode();

    let mut buf = [0u8; 64];
    let mut out = BufSink::new(&mut buf);
    fmt.format(&insn, &mut out);
    let n = out.len();
    (buf, n)
}
```

### 4. Token sink (syntax highlighting)

```rust
use fARM64::{Decoder, DecoderOptions, TokenKind};
use fARM64::format::{FmtFormatter, Formatter, FormatterOutput};

struct Highlighter;
impl FormatterOutput for Highlighter {
    fn write(&mut self, text: &str, kind: TokenKind) {
        let color = match kind {
            TokenKind::Mnemonic => "\x1b[1;36m",
            TokenKind::Register => "\x1b[33m",
            TokenKind::Number | TokenKind::Float | TokenKind::Address => "\x1b[35m",
            _ => "\x1b[0m",
        };
        print!("{color}{text}\x1b[0m");
    }
}

fn highlight(word: u32) {
    let insn = Decoder::new(&word.to_le_bytes(), 0, DecoderOptions::default()).decode();
    FmtFormatter::new().format(&insn, &mut Highlighter);
    println!();
}
```

### 5. Flow classification (implemented today)

```rust
use fARM64::{Decoder, DecoderOptions, FlowControl};

fn scan_until_return(code: &[u8], base: u64) {
    let mut dec = Decoder::new(code, base, DecoderOptions::default());
    while dec.can_decode() {
        let insn = dec.decode();
        match insn.flow_control() {
            FlowControl::Call | FlowControl::IndirectCall => { /* call site */ }
            FlowControl::Return => break,
            _ => {}
        }
    }
}
```

> The richer per-register access analysis (`info::instruction_info` →
> `used_registers`/`used_memory`) is defined but not yet implemented — see the
> [`InstructionInfo`](#instructioninfo-flowaccess-facility) status note.

### 6. Restricting accepted extensions at runtime

```rust
use fARM64::{Decoder, DecoderOptions, FeatureSet, Feature, DecodeError};

// Compile-in SVE (cargo feature), but reject SVE encodings at runtime here.
fn base_plus_lse_only(word: u32) -> Result<(), DecodeError> {
    let features = FeatureSet::BASE.with(Feature::Lse);
    let opts = DecoderOptions { features };
    let mut dec = Decoder::new(&word.to_le_bytes(), 0, opts);
    let insn = dec.decode();
    if insn.is_invalid() {
        return Err(dec.last_error()); // e.g. DecodeError::FeatureRequired(Feature::Sve)
    }
    Ok(())
}
```

---

## iced-x86 features intentionally dropped or changed

A64 is a different ISA from x86; the API keeps iced's *shape* but drops what does
not apply and adds AArch64-specific concepts.

| iced-x86 concept | fARM64 | Why |
|-|-|-|
| Variable instruction length, `Instruction::len()` field | `len()` is `const 4` | A64 is fixed 32-bit. |
| `Decoder::new(bitness, …)`; 16/32/64-bit modes | no bitness parameter | A64 is always 64-bit. |
| Prefixes, segment/REX/EVEX state, `Instruction::segment_prefix()` | dropped | No such concept on A64. |
| `MemorySize`, scaled-index `* 1/2/4/8` | `Operand::MemImm`/`MemExt`/`SveMem` (`ExtendType` + `shift` + `MemIndexMode`) | A64 uses extend+shift register addressing and named index modes. |
| `FlagsModified`/EFLAGS bitset | `FlagEffect` enum | A64 has a single 4-bit NZCV write classification. |
| Separate `Mnemonic` only | `Code` **and** `Mnemonic` (kept), plus alias resolution in the formatter | Same split as iced; alias conditions applied per the ARM ARM. |
| Many formatter dialects (Intel/AT&T/masm/nasm/gas) | `FmtFormatter` (UAL) default + optional `GnuFormatter` | A64 has fewer mainstream syntaxes; UAL is the spec's preferred disassembly. |
| `Register` as flat numeric, raw stack pointer | SP/ZR resolved at decode; never raw reg-31; SVE `Z`/predicate `P`, scalar `B/H/S/D/Q`, vector `V` views | A64 register model. |
| Encoder/`BlockEncoder` shipped | semantic **encoder shipped** (`encode`/`Instruction::encode` → `Result<u32, EncodeError>`, validated by decode→encode→decode round-trip); no `BlockEncoder`/relocation layer | Round-trip proves the decode is invertible; block-level relocation is out of scope. |
| `alloc`/`String` assumed available | **no_std + no-alloc is the default**; `String`/`Vec` are `alloc`-gated extras | Hard portability requirement (wasm32, bare-metal). |

**Added vs iced:** `FeatureSet`/`Feature` runtime extension gating with
`DecodeError::FeatureRequired`; `Operand` carries arrangement / lane / predicate
qualifier / SME-tile shapes; `VectorArrangement`, `PredQual`, `SliceIndicator`,
and `SmeTile` operands; `BufSink` fixed-buffer formatting as a first-class sink.

---

## Provenance and licensing

The decode core behind this API is an **original, hand-written implementation**
derived from the *Arm Architecture Reference Manual*. It is **not** a derivative of
Binary Ninja's `arch-arm64` or any other disassembler, and carries no
required-attribution obligation. `fARM64` is licensed `MIT`.

Correctness is defined against the ARM ARM and cross-checked against multiple
oracles (LLVM `llvm-mc`/`llvm-objdump`, GNU binutils `objdump`, and Binary Ninja's
`test_cases.txt` corpus used only as a development guide). Where an oracle diverges
from the spec, `fARM64` follows the spec; see `ENCODING.md` for the decode tree,
the ARM pseudocode helpers, and the documented divergence allow-list.
