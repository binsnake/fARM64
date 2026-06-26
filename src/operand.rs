//! The rich, safe, `Copy` [`Operand`] enum and its [`OpKind`] discriminant view.
//!
//! A safe, structured operand: SP/ZR are pre-resolved into the [`Register`]
//! value, and "is this field valid?" booleans are replaced by the variant
//! itself. Every variant is `Copy` and allocation-free.

use crate::enums::{Condition, ExtendType, ShiftType, VectorArrangement};
use crate::register::Register;
use crate::sysop::SysToken;
use crate::sysreg::SystemReg;

/// Addressing mode of a memory operand ([`Operand::MemImm`]).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemIndexMode {
    /// `[base, #imm]` — base + immediate offset, base unchanged.
    Offset,
    /// `[base, #imm]!` — pre-index: base updated before access.
    PreIndex,
    /// `[base]!` — writeback with no displacement (the MOPS `CPY*`/`SET*`
    /// address operands). Renders the bracket and a trailing `!`, no `, #imm`.
    PreNoOffset,
    /// `[base], #imm` — post-index by immediate: base updated after access.
    PostImm,
    /// `[base], <Xm>` — post-index by register (some SIMD load/store forms).
    PostReg,
}

/// SVE predicate qualifier on a register operand (`/z` zeroing, `/m` merging).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PredQual {
    /// No qualifier (governing/source predicate used plain).
    None,
    /// Zeroing predication (`/z`).
    Zeroing,
    /// Merging predication (`/m`).
    Merging,
}

/// Addressing mode of an SVE memory operand ([`Operand::SveMem`]).
///
/// SVE load/store/prefetch addressing does not fit the base-ISA [`MemImm`] /
/// [`MemExt`] shapes: it adds a `MUL VL`-scaled immediate, a 64-bit scalar plus
/// **scalable-vector** index (gather/scatter), and a scalable-vector base. This
/// enum tags which [`Operand::SveMem`] sub-form a value carries so the formatter
/// can render the exact Binary Ninja / ARM spelling.
///
/// [`MemImm`]: Operand::MemImm
/// [`MemExt`]: Operand::MemExt
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SveMemMode {
    /// `[<Xn|SP>{, #<imm>, MUL VL}]` — scalar base plus a VL-scaled immediate
    /// (the immediate counts vector-length multiples). `[<Xn|SP>]` when the
    /// immediate is zero.
    ScalarImmMulVl,
    /// `[<Xn|SP>{, #<imm>}]` — scalar base plus a plain (already byte-scaled)
    /// immediate rendered in **signed hex** (`#-0x1`/`#0x40`), used by `LD1RO`
    /// and the broadcasting `LD1R*` forms.
    ScalarImm,
    /// `[<Xn|SP>{, #<imm>}]` — scalar base plus a plain immediate rendered with
    /// the SVE radix convention (negative in **decimal**, positive in hex), used
    /// by `LD1RQ` (whose displayed offset is a real `×16` byte count).
    ScalarImmDec,
    /// `[<Zn>.<T>{, #<imm>}]` — scalable-vector base plus an unsigned immediate
    /// (vector-plus-immediate gather/scatter). `[<Zn>.<T>]` when zero.
    VecImm,
    /// `[<Zn>.<T>, <Xm>]` — scalable-vector base plus a 64-bit scalar offset
    /// (the SVE2 `LDNT1*`/`STNT1*` vector-base gather/scatter).
    VecScalar,
    /// `[<Xn|SP>, <Zm>.<T>{, <mod> #<amt>}]` — 64-bit scalar base plus a
    /// scalable-vector index with an optional `SXTW`/`UXTW`/`LSL` modifier
    /// (gather load / scatter store). The modifier is carried in `extend`
    /// (`Uxtx` renders as `lsl`) and `amount` holds the shift.
    ScalarVec,
    /// `[<Zn>.<T>, <Zm>.<T>{, <mod> #<amt>}]` — scalable-vector base plus a
    /// scalable-vector index, both with the same arrangement, and an optional
    /// `SXTW`/`UXTW`/`LSL` modifier (the SVE `ADR` vector-address form). The
    /// modifier is carried in `extend` (`Uxtx` renders as `lsl`); `amount` holds
    /// the shift and the modifier is shown only when `amount != 0`.
    VecVec,
}

/// SME ZA-tile slice indicator (horizontal / vertical), for [`Operand::SmeTile`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SliceIndicator {
    /// No slice direction.
    None,
    /// Horizontal slice (`ZA0H...`).
    Horizontal,
    /// Vertical slice (`ZA0V...`).
    Vertical,
}

/// A single decoded operand.
///
/// `Copy`. The register-bearing [`Operand::Reg`] variant folds arrangement,
/// lane, shift, extend and predicate-qualifier decorations into one place so
/// callers never juggle parallel option fields.
///
/// Realized size is 16 bytes (see the `static_asserts` in `lib.rs`). It derives
/// `PartialEq` but **not** `Eq`/`Hash`: the [`Operand::FpImm`] payload is an
/// `f32`, which has no total order or hash. This is an intentional, documented
/// departure from a hypothetical `Eq`/`Hash` operand.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Operand {
    /// Empty slot (no operand).
    None,

    /// A register with optional SIMD/SVE decorations.
    Reg {
        /// The (already SP/ZR-resolved) register.
        reg: Register,
        /// Vector/SVE arrangement, if any (`.4s`, `.b`, ...).
        arr: Option<VectorArrangement>,
        /// Element lane index, if this is an indexed element (`v0.s[2]`).
        lane: Option<u8>,
        /// Shift modifier applied to the register, if any.
        shift: Option<(ShiftType, u8)>,
        /// Register-extension applied (extended-register forms).
        extend: Option<ExtendType>,
        /// SVE predicate qualifier (`/z`, `/m`), if applicable.
        pred: Option<PredQual>,
    },

    /// Unsigned immediate, already widened to `u64`.
    ImmUnsigned(u64),
    /// Signed immediate, already sign-extended to `i64`.
    ImmSigned(i64),
    /// Logical (bitmask) immediate value computed by `decode_bitmasks`.
    ImmLogical(u64),
    /// Wide-move immediate: a 16-bit value to be shifted left by `lsl` bits
    /// (`MOVZ`/`MOVN`/`MOVK`).
    ImmShiftedMove {
        /// The 16-bit immediate.
        imm: u16,
        /// Left-shift amount in bits (`0`/`16`/`32`/`48`).
        lsl: u8,
    },
    /// Modified-immediate value rendered with a trailing `, msl #amt` (the
    /// "mask shift left", ones shifted in) used by the `MOVI`/`MVNI` `MSL` forms
    /// (`movi v0.4s, #0x96, msl #0x10`). `imm` is the raw 8-bit immediate.
    ImmShiftedMsl {
        /// The 8-bit immediate.
        imm: u16,
        /// Mask-shift-left amount in bits (`8` or `16`).
        msl: u8,
    },
    /// Floating-point immediate, materialised from its 8-bit encoding via
    /// `vfp_expand_imm` (bit-cast only; no FP arithmetic at decode time).
    FpImm(f32),
    /// A bare shift amount operand (e.g. the `#amt` of `LSL #amt`).
    ShiftAmount(u8),
    /// A PC-relative target, pre-resolved to an absolute address.
    Label(u64),
    /// A condition code operand (`CCMP`, ...).
    Cond(Condition),
    /// A system register reference.
    SysReg(SystemReg),
    /// A fixed system-instruction keyword operand: a barrier option, `csync`, a
    /// PSTATE field, a `BTI` target, a `cN` token, or an `IC`/`DC`/`AT`/`TLBI`/
    /// `CFP`/`CPP`/`DVP` operation name (see [`crate::sysop::SysToken`]).
    SysOp(SysToken),

    /// Memory operand with an immediate displacement.
    MemImm {
        /// Base register (SP-resolved where appropriate).
        base: Register,
        /// Signed displacement.
        imm: i64,
        /// Addressing mode.
        mode: MemIndexMode,
    },
    /// Memory operand with a register index plus extend/shift.
    MemExt {
        /// Base register.
        base: Register,
        /// Index register.
        index: Register,
        /// Index extension type.
        extend: ExtendType,
        /// Index left-shift amount.
        shift: u8,
    },

    /// A list of consecutive vector registers (`{v0.4s, v1.4s}`, SIMD
    /// load/store structures).
    MultiReg {
        /// The registers, in order; only `count` are valid.
        regs: [Register; 4],
        /// Number of valid entries in `regs` (`1..=4`).
        count: u8,
        /// Shared arrangement for the list.
        arr: Option<VectorArrangement>,
        /// Shared lane index for single-structure forms.
        lane: Option<u8>,
    },

    /// An indexed element operand carrying a register index expression
    /// (some SVE gather/scatter and indexed forms).
    IndexedElement {
        /// The data register.
        reg: Register,
        /// Arrangement of `reg`.
        arr: Option<VectorArrangement>,
        /// Index register.
        index: Register,
        /// Immediate index / offset.
        imm: i64,
    },

    /// An SME ZA tile (with optional slice direction).
    SmeTile {
        /// Packed tile identifier.
        tile: u16,
        /// Slice direction.
        slice: SliceIndicator,
    },

    /// An implementation-specific / opaque operand payload (rare forms that do
    /// not fit the structured variants); five raw bytes, no heap.
    ImplSpec([u8; 5]),

    /// An SVE element-count pattern operand (`pow2`, `vl1`..`vl256`, `mul3`,
    /// `mul4`, `all`, or a raw `#0xN` for the unnamed values), used by
    /// `CNTB`/`INCB`/`SQINCB`/... The payload is the raw 5-bit `pattern` field;
    /// the formatter renders the keyword or, for `0b01110..=0b11100`, the value
    /// as `#0xN` (matching the ARM ARM / Binary Ninja rendering).
    SvePattern(u8),

    /// An SVE multiplier modifier rendered as `mul #0xN` (the trailing
    /// `, MUL #<imm>` of `INC`/`DEC`/`CNT` element-count forms). The payload is
    /// the multiplier `imm` (`1..=16`).
    SveMul(u8),

    /// A signed immediate rendered with the SVE radix convention: a non-negative
    /// value prints in hex (`#0x..`), a negative value prints in **decimal**
    /// (`#-12`). Used by SVE `INDEX`/`CMP #imm`/`ADDVL`/`SMAX #imm`/`DUP #imm`
    /// and friends, where Binary Ninja mixes hex (positive) and decimal
    /// (negative) magnitudes.
    ImmSignedDec(i64),

    /// An SVE memory addressing operand (load / store / prefetch).
    ///
    /// Covers the SVE-specific addressing shapes that do not fit [`MemImm`] /
    /// [`MemExt`]: the `MUL VL`-scaled immediate, the scalar-plus-scalable-vector
    /// gather/scatter index, and the scalable-vector base. [`mode`](SveMemMode)
    /// selects which sub-form is active and therefore which fields are
    /// meaningful. The whole value is `Copy` and fits the 16-byte budget.
    ///
    /// [`MemImm`]: Operand::MemImm
    /// [`MemExt`]: Operand::MemExt
    SveMem {
        /// Base register: a GP register (`Xn|SP`) for the scalar-base modes, or
        /// a scalable-vector register (`Zn`) for the vector-base modes.
        base: Register,
        /// Offset/index register: the 64-bit scalar `Xm` ([`SveMemMode::VecScalar`])
        /// or the scalable-vector index `Zm` ([`SveMemMode::ScalarVec`]).
        /// [`Register::None`] when the mode has no register offset.
        offset: Register,
        /// Arrangement of the scalable-vector component (the base for the
        /// vector-base modes, the index for [`SveMemMode::ScalarVec`]).
        arr: Option<VectorArrangement>,
        /// Index modifier for [`SveMemMode::ScalarVec`] (`Uxtw`/`Sxtw`, or
        /// `Uxtx` which renders as `lsl`). Unused by the other modes.
        extend: ExtendType,
        /// Immediate displacement (meaning per [`mode`](SveMemMode): VL-multiples,
        /// byte offset, or vector element offset).
        imm: i32,
        /// Shift / scale amount for [`SveMemMode::ScalarVec`]; `0xFF` means "no
        /// `#amt` shown" (the unscaled gather/scatter forms).
        amount: u8,
        /// Which SVE addressing sub-form this operand carries.
        mode: SveMemMode,
    },

    /// An SME ZA-array tile *slice* operand, as Binary Ninja renders the SME
    /// `MOVA`, `LD1*`/`ST1*` (ZA array vector) and `LDR`/`STR` (ZA) forms.
    ///
    /// Two surface spellings, selected by [`reg`](Operand::SmeTileSlice::reg):
    ///
    /// * a per-tile slice `z<n><h|v>.<T>[<Ws>{, #<imm>}]` (e.g. `z0v.b[w14,
    ///   #0x5]`, `z3h.s[w13]`) when `reg` is a `Z` register — note Binary Ninja
    ///   prints these tiles with a **`z` prefix**, not `za`; or
    /// * the whole-array select `za[<Ws>{, #<imm>}]` (the `LDR`/`STR` ZA forms)
    ///   when `reg` is [`Register::None`] and [`slice`](Operand::SmeTileSlice::slice)
    ///   is [`SliceIndicator::None`].
    ///
    /// The immediate is shown only when [`has_imm`](Operand::SmeTileSlice::has_imm)
    /// is set (the `.q`/`Q` 128-bit slice forms have no index immediate).
    SmeTileSlice {
        /// The tile register rendered with a `z` prefix (`Z0..Z31`), or
        /// [`Register::None`] for the whole-array `za[...]` form.
        reg: Register,
        /// Slice direction (`h`/`v`); [`SliceIndicator::None`] for `za[...]`.
        slice: SliceIndicator,
        /// Element-size arrangement suffix (`.b`/`.h`/`.s`/`.d`/`.q`), or `None`
        /// for the whole-array `za[...]` form.
        arr: Option<VectorArrangement>,
        /// The `Ws` slice-select GP register (`W12..W15`).
        sel: Register,
        /// The slice index immediate (`#<imm>`), shown only when `has_imm`.
        imm: i16,
        /// Whether to render the `, #<imm>` index (false for the `.q` forms).
        has_imm: bool,
    },

    /// A general-purpose register carrying a writeback `!` suffix, rendered as
    /// `x2!`. Used only by the MOPS `CPY*`/`SET*` family for the size/count
    /// operand (and, for `CPY*`, all three operands' register part). The
    /// register is already SP/ZR-resolved.
    RegBang(Register),

    /// A consecutive 64-bit register pair `<Xt>, <Xt+1>`, rendered as two
    /// comma-separated registers. Used by the `FEAT_D128` `MRRS`/`MSRR` (128-bit
    /// system-register pair move) and generic `SYSP` (system pair) instructions,
    /// where the architecture transfers an even/odd `X` pair through a single
    /// even base register. Both halves are already SP/ZR-resolved (the odd half
    /// is `xzr` when the base is `x30`). Carrying the pair as one operand keeps
    /// these instructions within [`crate::MAX_OPERANDS`].
    RegPair {
        /// The even (low) register of the pair.
        first: Register,
        /// The odd (high) register of the pair (`first` number + 1; `xzr` for 31).
        second: Register,
    },

    /// An SME2 ZA-array vector *slice group* destination, as LLVM renders the
    /// SME2 multi-vector accumulate / multiply-into-ZA instructions:
    /// `za.<T>[<Ws>, <off>{:<off+span-1>}{, vgx2|vgx4}]`.
    ///
    /// Examples: `za.s[w8, 0:3]`, `za.h[w8, 6, vgx2]`, `za.s[w9, 4:7, vgx4]`,
    /// `za.s[w8, 4]`. The element-size [`arr`](Operand::SmeZaSlice::arr) is the
    /// `.<T>` suffix; [`sel`](Operand::SmeZaSlice::sel) is the `W8..W11` (or
    /// `W12..W15`) slice-select register; [`off`](Operand::SmeZaSlice::off) is the
    /// first slice index; [`span`](Operand::SmeZaSlice::span) is how many
    /// consecutive slices the group covers (`1` → `off`, `>1` → `off:off+span-1`);
    /// and [`vg`](Operand::SmeZaSlice::vg) is the multi-vector qualifier
    /// (`0` → none, `2` → `vgx2`, `4` → `vgx4`).
    SmeZaSlice {
        /// Element-size arrangement (`.b`/`.h`/`.s`/`.d`).
        arr: Option<VectorArrangement>,
        /// The `Ws` slice-select GP register (`W8..W11`/`W12..W15`).
        sel: Register,
        /// First slice index.
        off: u8,
        /// Number of consecutive slices in the group (`1`/`2`/`4`); a span `>1`
        /// renders the `off:off+span-1` range.
        span: u8,
        /// Multi-vector qualifier: `0` (none), `2` (`vgx2`), or `4` (`vgx4`).
        vg: u8,
    },

    /// An SME2/SVE2 multi-vector register *group* source, as LLVM renders the
    /// strided/consecutive vector lists of the SME2 multi-vector forms:
    /// `{ z0.b, z1.b }` (a comma list) or `{ z0.b - z3.b }` (a range).
    ///
    /// [`first`](Operand::SveVecGroup::first) is the lowest `Z` register;
    /// [`count`](Operand::SveVecGroup::count) is how many registers (`2` or `4`);
    /// [`arr`](Operand::SveVecGroup::arr) is the shared element-size suffix; and
    /// [`range`](Operand::SveVecGroup::range) selects the ` - ` range rendering
    /// (LLVM uses the range for consecutive 4-register groups and the comma list
    /// for 2-register groups).
    ///
    /// [`stride`](Operand::SveVecGroup::stride) is the register-number step between
    /// successive group members: `1` for the usual *consecutive* group
    /// (`{ z8.s - z11.s }`), or the larger step of the SME2 *strided* multi-vector
    /// load/store lists — `8` for a 2-register strided group (`{ z16.d, z24.d }`)
    /// and `4` for a 4-register strided group (`{ z1.h, z5.h, z9.h, z13.h }`). A
    /// strided group always renders as a comma list (`range == false`).
    SveVecGroup {
        /// The lowest `Z` register of the group.
        first: Register,
        /// Number of registers (`2` or `4`).
        count: u8,
        /// Shared element-size arrangement (`.b`/`.h`/`.s`/`.d`).
        arr: Option<VectorArrangement>,
        /// Render as a ` - ` range (`{ z0.b - z3.b }`) rather than a comma list.
        range: bool,
        /// Register-number step between successive members (`1` consecutive, `8`
        /// or `4` for the strided multi-vector lists).
        stride: u8,
    },

    /// An SME2 / SVE2.1 **predicate-as-counter** governing operand (`PNg`), as
    /// LLVM renders the multi-vector predicate-as-counter forms: `pn8`..`pn15`,
    /// optionally with a `/z` zeroing qualifier (`pn8/z`).
    ///
    /// The predicate-as-counter is the same physical register file as the SVE
    /// predicates `P0`..`P15`, but the 3-bit `PNg` field selects only `P8`..`P15`
    /// and the assembler spells it with a `pn` prefix. The underlying register is
    /// carried in [`reg`](Operand::PredCounter::reg) (`P8`..`P15`); the formatter
    /// rewrites the `p` to `pn`. Loads take `/z`
    /// ([`zeroing`](Operand::PredCounter::zeroing) `= true`); stores and the ALU
    /// `SEL` take no qualifier.
    PredCounter {
        /// The underlying predicate register (`P8`..`P15`), rendered `pn8`..`pn15`.
        reg: Register,
        /// Whether the `/z` zeroing qualifier is shown.
        zeroing: bool,
        /// Optional element-size suffix (`.b`/`.h`/`.s`/`.d`), as on the SVE2.1
        /// `WHILE<cc>` predicate-as-counter result (`pn8.b`). `None` for the SME2
        /// multi-vector governing forms, which render the bare `pn8`.
        arr: Option<crate::enums::VectorArrangement>,
    },

    /// An SVE2.1 `VLx2`/`VLx4` vector-length multiplier decorator, as the trailing
    /// operand of the `WHILE<cc>` predicate-as-counter forms
    /// (`WHILE<cc> <PNd>.<T>, <Xn>, <Xm>, VLx2`). The value is `2` or `4`.
    VlMul(u8),
}

impl Operand {
    /// The [`OpKind`] discriminant of this operand, telling callers which typed
    /// accessor to use.
    #[inline]
    pub const fn kind(self) -> OpKind {
        match self {
            Operand::None => OpKind::None,
            Operand::Reg { .. } => OpKind::Register,
            Operand::ImmUnsigned(_) => OpKind::ImmUnsigned,
            Operand::ImmSigned(_) => OpKind::ImmSigned,
            Operand::ImmLogical(_) => OpKind::ImmLogical,
            Operand::ImmShiftedMove { .. } => OpKind::ImmShiftedMove,
            Operand::ImmShiftedMsl { .. } => OpKind::ImmShiftedMsl,
            Operand::FpImm(_) => OpKind::FpImm,
            Operand::ShiftAmount(_) => OpKind::ShiftAmount,
            Operand::Label(_) => OpKind::Label,
            Operand::Cond(_) => OpKind::Cond,
            Operand::SysReg(_) => OpKind::SysReg,
            Operand::SysOp(_) => OpKind::SysOp,
            Operand::MemImm { .. } => OpKind::MemImm,
            Operand::MemExt { .. } => OpKind::MemExt,
            Operand::MultiReg { .. } => OpKind::MultiReg,
            Operand::IndexedElement { .. } => OpKind::IndexedElement,
            Operand::SmeTile { .. } => OpKind::SmeTile,
            Operand::ImplSpec(_) => OpKind::ImplSpec,
            Operand::SvePattern(_) => OpKind::SvePattern,
            Operand::SveMul(_) => OpKind::SveMul,
            Operand::ImmSignedDec(_) => OpKind::ImmSignedDec,
            Operand::SveMem { .. } => OpKind::SveMem,
            Operand::SmeTileSlice { .. } => OpKind::SmeTileSlice,
            Operand::RegBang(_) => OpKind::RegBang,
            Operand::RegPair { .. } => OpKind::RegPair,
            Operand::SmeZaSlice { .. } => OpKind::SmeZaSlice,
            Operand::SveVecGroup { .. } => OpKind::SveVecGroup,
            Operand::PredCounter { .. } => OpKind::PredCounter,
            Operand::VlMul(_) => OpKind::VlMul,
        }
    }
}

impl Default for Operand {
    #[inline]
    fn default() -> Self {
        Operand::None
    }
}

/// Discriminant naming each operand slot's shape (iced `OpKind` analog), so a
/// caller can pick the matching typed accessor on [`crate::Instruction`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    /// No operand.
    None,
    /// [`Operand::Reg`].
    Register,
    /// [`Operand::ImmUnsigned`].
    ImmUnsigned,
    /// [`Operand::ImmSigned`].
    ImmSigned,
    /// [`Operand::ImmLogical`].
    ImmLogical,
    /// [`Operand::ImmShiftedMove`].
    ImmShiftedMove,
    /// [`Operand::ImmShiftedMsl`].
    ImmShiftedMsl,
    /// [`Operand::FpImm`].
    FpImm,
    /// [`Operand::ShiftAmount`].
    ShiftAmount,
    /// [`Operand::Label`].
    Label,
    /// [`Operand::Cond`].
    Cond,
    /// [`Operand::SysReg`].
    SysReg,
    /// [`Operand::SysOp`].
    SysOp,
    /// [`Operand::MemImm`].
    MemImm,
    /// [`Operand::MemExt`].
    MemExt,
    /// [`Operand::MultiReg`].
    MultiReg,
    /// [`Operand::IndexedElement`].
    IndexedElement,
    /// [`Operand::SmeTile`].
    SmeTile,
    /// [`Operand::ImplSpec`].
    ImplSpec,
    /// [`Operand::SvePattern`].
    SvePattern,
    /// [`Operand::SveMul`].
    SveMul,
    /// [`Operand::ImmSignedDec`].
    ImmSignedDec,
    /// [`Operand::SveMem`].
    SveMem,
    /// [`Operand::SmeTileSlice`].
    SmeTileSlice,
    /// [`Operand::RegBang`].
    RegBang,
    /// [`Operand::RegPair`].
    RegPair,
    /// [`Operand::SmeZaSlice`].
    SmeZaSlice,
    /// [`Operand::SveVecGroup`].
    SveVecGroup,
    /// [`Operand::PredCounter`].
    PredCounter,
    /// [`Operand::VlMul`].
    VlMul,
}
