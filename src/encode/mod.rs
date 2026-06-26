//! The A64 **encoder** — the inverse of [`crate::decode`].
//!
//! Reconstructs the raw 32-bit little-endian instruction word from the
//! *semantics* of an [`Instruction`] (its [`Code`], [`Mnemonic`], operands and
//! [`Instruction::ip`]). It deliberately never reads
//! [`Instruction::word`](crate::Instruction::word): the whole point is to prove
//! the decode is invertible, so the encoder must rebuild the word purely from
//! the public projection.
//!
//! Like the decoder, the encoder is `no_std`, zero-alloc, and **total**: it
//! never panics. Anything it cannot (yet) encode returns [`EncodeError`].
//!
//! Dispatch is on [`Instruction::code`] — the canonical encoding identity —
//! routed to a per-group encoder. Only [`dp_imm`] is implemented so far; the
//! other groups are compiling stubs that return [`EncodeError::Unsupported`]
//! and are filled in by later work.

pub mod bits;

pub mod branch_sys;
pub mod dp_imm;
pub mod dp_reg;
pub mod ldst;
pub mod ldst_simd;
pub mod mops;
pub mod simd_fp;
pub mod sme;
pub mod sve;

use crate::instruction::Instruction;
use crate::mnemonic::Code;

/// Why an [`Instruction`] could not be encoded back to a 32-bit word.
///
/// The encoder is total — every failure surfaces here rather than panicking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum EncodeError {
    /// This [`Code`] / group is not implemented by the encoder yet.
    Unsupported,
    /// An operand was missing, of the wrong kind, or otherwise inconsistent
    /// with the encoding (e.g. a register out of range, or a mnemonic alias
    /// whose operand layout did not match).
    InvalidOperand,
    /// An immediate has no valid encoding in the instruction's field(s) (e.g. a
    /// value that is not a representable logical/bitmask immediate, an
    /// out-of-range shift, or a PC-relative target out of reach).
    InvalidImmediate,
    /// The instruction is the invalid sentinel ([`Code::Invalid`]) and has no
    /// encoding.
    Invalid,
}

impl core::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            EncodeError::Unsupported => "encoding not supported by the encoder yet",
            EncodeError::InvalidOperand => "operand invalid for this encoding",
            EncodeError::InvalidImmediate => "immediate has no valid encoding",
            EncodeError::Invalid => "invalid instruction has no encoding",
        };
        f.write_str(s)
    }
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl std::error::Error for EncodeError {}

/// Encode `insn` back into its 32-bit little-endian A64 instruction word.
///
/// Dispatches on [`Instruction::code`] to the per-group encoder. Reconstructs
/// the word purely from the instruction's semantics (code, mnemonic, operands,
/// `ip`) — it never inspects [`Instruction::word`]. Returns [`EncodeError`] for
/// any instruction it cannot encode; never panics.
#[inline]
pub fn encode(insn: &Instruction) -> Result<u32, EncodeError> {
    let code = insn.code();
    match code {
        Code::Invalid => Err(EncodeError::Invalid),
        c if is_dp_imm(c) => dp_imm::encode(insn),
        c if is_dp_reg(c) => dp_reg::encode(insn),
        c if is_branch_sys(c) => branch_sys::encode(insn),
        c if mops::is_mops(c) => mops::encode(insn),
        c if is_ldst(c) => ldst::encode(insn),
        c if simd_fp::is_simd_fp(c) => simd_fp::encode(insn),
        c if sve::is_sve(c) => sve::encode(insn),
        c if sme::is_sme(c) => sme::encode(insn),
        _ => Err(EncodeError::Unsupported),
    }
}

impl Instruction {
    /// Encode this instruction back into its 32-bit little-endian word.
    ///
    /// See [`encode`]. Reconstructs the word from semantics only; never reads
    /// [`Instruction::word`].
    #[inline]
    pub fn encode(&self) -> Result<u32, EncodeError> {
        encode(self)
    }
}

/// `true` for every [`Code`] produced by the Data Processing -- Immediate group
/// ([`crate::decode::dp_imm`]); the set this encoder fully covers.
#[inline]
fn is_dp_imm(code: Code) -> bool {
    use Code::*;
    matches!(
        code,
        Adr | Adrp
            | AddImm32 | AddImm64 | AddsImm32 | AddsImm64
            | SubImm32 | SubImm64 | SubsImm32 | SubsImm64
            | AddgImm | SubgImm
            | AndImm32 | AndImm64 | OrrImm32 | OrrImm64
            | EorImm32 | EorImm64 | AndsImm32 | AndsImm64
            | Movn32 | Movn64 | Movz32 | Movz64 | Movk32 | Movk64
            | Sbfm32 | Sbfm64 | Bfm32 | Bfm64 | Ubfm32 | Ubfm64
            | Extr32 | Extr64
            // Min/max (immediate) — FEAT_CSSC.
            | SmaxImm32 | SmaxImm64 | SminImm32 | SminImm64
            | UmaxImm32 | UmaxImm64 | UminImm32 | UminImm64
    )
}

/// `true` for every [`Code`] produced by the Data Processing -- Register group
/// ([`crate::decode::dp_reg`]); the set this encoder fully covers.
#[inline]
fn is_dp_reg(code: Code) -> bool {
    use Code::*;
    matches!(
        code,
        // Logical (shifted register).
        AndShifted32 | AndShifted64 | BicShifted32 | BicShifted64 | OrrShifted32 | OrrShifted64
            | OrnShifted32 | OrnShifted64 | EorShifted32 | EorShifted64 | EonShifted32
            | EonShifted64 | AndsShifted32 | AndsShifted64 | BicsShifted32 | BicsShifted64
        // Add/subtract (shifted register).
            | AddShifted32 | AddShifted64 | AddsShifted32 | AddsShifted64 | SubShifted32
            | SubShifted64 | SubsShifted32 | SubsShifted64
        // Add/subtract (extended register).
            | AddExtended32 | AddExtended64 | AddsExtended32 | AddsExtended64 | SubExtended32
            | SubExtended64 | SubsExtended32 | SubsExtended64
        // Add/subtract (with carry).
            | Adc32 | Adc64 | Adcs32 | Adcs64 | Sbc32 | Sbc64 | Sbcs32 | Sbcs64
        // Flag manipulation.
            | Rmif | Setf8 | Setf16
        // Conditional compare (register / immediate).
            | CcmnReg32 | CcmnReg64 | CcmnImm32 | CcmnImm64 | CcmpReg32 | CcmpReg64 | CcmpImm32
            | CcmpImm64
        // Conditional select.
            | Csel32 | Csel64 | Csinc32 | Csinc64 | Csinv32 | Csinv64 | Csneg32 | Csneg64
        // Data-processing (3 source).
            | Madd32 | Madd64 | Msub32 | Msub64 | Smaddl | Smsubl | Smulh | Umaddl | Umsubl
            | Umulh | Maddpt | Msubpt
        // Add/subtract checked-pointer (FEAT_CPA scalar).
            | Addpt | Subpt
        // Data-processing (2 source).
            | Udiv32 | Udiv64 | Sdiv32 | Sdiv64 | Lslv32 | Lslv64 | Lsrv32 | Lsrv64 | Asrv32
            | Asrv64 | Rorv32 | Rorv64 | Crc32b | Crc32h | Crc32w | Crc32x | Crc32cb | Crc32ch
            | Crc32cw | Crc32cx | SubpDp | SubpsDp | IrgDp | GmiDp | Pacga
        // Min/max (register) — FEAT_CSSC.
            | SmaxReg32 | SmaxReg64 | SminReg32 | SminReg64
            | UmaxReg32 | UmaxReg64 | UminReg32 | UminReg64
        // Data-processing (1 source).
            | Rbit32 | Rbit64 | Rev1632 | Rev1664 | Rev3232 | Rev3264 | Rev32Bit | Rev64Bit
            | Clz32 | Clz64 | Cls32 | Cls64
        // Data-processing (1 source) — FEAT_CSSC.
            | Abs32 | Abs64 | Cnt32 | Cnt64 | Ctz32 | Ctz64
            | PaciaDp | PacibDp | PacdaDp | PacdbDp | AutiaDp | AutibDp | AutdaDp | AutdbDp
            | PacizaDp | PacizbDp | PacdzaDp | PacdzbDp | AutizaDp | AutizbDp | AutdzaDp
            | AutdzbDp | XpaciDp | XpacdDp
    )
}

/// `true` for every [`Code`] produced by the Loads-and-Stores group
/// ([`crate::decode::ldst`] / [`crate::decode::ldst_simd`]); the set this encoder
/// fully covers. Delegates to [`ldst::is_ldst`].
#[inline]
fn is_ldst(code: Code) -> bool {
    ldst::is_ldst(code)
}

/// `true` for every [`Code`] produced by the Branches / Exception-generating /
/// System group ([`crate::decode::branch_sys`]) plus the reserved `UDF`
/// ([`crate::decode`]); the set this encoder fully covers.
#[inline]
fn is_branch_sys(code: Code) -> bool {
    use Code::*;
    matches!(
        code,
        // Conditional / unconditional branch (immediate). `BcCond` is the
        // FEAT_HBC hinted conditional branch; the `*sppc` are the FEAT_PAuth_LR
        // PC-relative authenticate/return branches (decoded in dp_imm, but
        // encoded here next to the other branch-with-label forms).
        BCond | BcCond | BUncond | BlImm
            | Retaasppc | Retabsppc | Autiasppc | Autibsppc
        // Compare / test and branch.
            | Cbz32 | Cbz64 | Cbnz32 | Cbnz64 | Tbz | Tbnz
        // FEAT_CMPBR compare-and-branch (register / immediate).
            | Cbgt | Cbge | Cbhi | Cbhs | Cbeq | Cbne | Cblt | Cblo
            | Cbbgt | Cbbge | Cbbhi | Cbbhs | Cbbeq | Cbbne
            | Cbhgt | Cbhge | Cbhhi | Cbhhs | Cbheq | Cbhne
        // Unconditional branch (register) + PAuth.
            | Br | Blr | Ret | Eret | Drps | Braaz | Brabz | Blraaz | Blrabz | Braa | Brab
            | Blraa | Blrab | Retaa | Retab | Eretaa | Eretab
        // Exception generation.
            | Svc | Hvc | Smc | Brk | Hlt | Tcancel | Dcps1 | Dcps2 | Dcps3
        // System: hints.
            | Nop | Yield | Wfe | Wfi | Sev | Sevl | Esb | Psb | Csdb | Bti | Tsb | HintGeneric
        // System: WFET/WFIT.
            | Wfet | Wfit
        // System: barriers.
            | Clrex | Dmb | Dsb | Isb | Sb | Tcommit
        // System: MSR (immediate) PSTATE and bare PSTATE ops.
            | MsrImm | Cfinv | Xaflag | Axflag | Smstart | Smstop
        // System: MSR/MRS (register).
            | MsrReg | Mrs
        // System: SYS/SYSL.
            | Sys | Sysl
        // System: TSTART/TTEST.
            | Tstart | Ttest
        // System: FEAT_D128 pair forms (MRRS/MSRR/SYSP/TLBIP).
            | Mrrs | Msrr | Sysp
        // K4: TCHANGE translation-table change (register / immediate).
            | TchangefReg | TchangebReg | TchangefImm | TchangebImm
        // Reserved: UDF.
            | Udf
    )
}
