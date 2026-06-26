//! SVE/SVE2 encoder implementation (compiled only with `feature = "sve"`).
//!
//! Inverts the four decode submodules ([`crate::decode::sve`]): integer
//! ([`int`]), permute/predicate ([`perm`]), floating-point ([`fp`]) and memory
//! ([`mem`]). The public [`encode`] dispatches on [`Instruction::code`] to the
//! family encoders; [`is_sve`] reports whether a code belongs to the group.

mod fp;
mod int;
mod mem;
mod perm;

use crate::encode::EncodeError;
use crate::enums::VectorArrangement as VA;
use crate::instruction::Instruction;
use crate::mnemonic::Code;
use crate::operand::{Operand, PredQual};
use crate::register::RegClass;

/// Result alias used throughout the SVE encoder.
pub(crate) type R = Result<u32, EncodeError>;

/// `true` for every [`Code`] produced by the SVE/SVE2 decoder.
#[inline]
pub fn is_sve(code: Code) -> bool {
    matches!(code, Code::RevdZPZ | Code::SveRevdZpzZero)
        || int::is_int(code)
        || perm::is_perm(code)
        || fp::is_fp(code)
        || mem::is_mem(code)
}

/// Encode an SVE/SVE2 instruction by inverting its decoder.
pub fn encode(insn: &Instruction) -> R {
    let code = insn.code();
    if let Some(w) = int::enc(insn, code)? {
        return Ok(w);
    }
    if let Some(w) = perm::enc(insn, code)? {
        return Ok(w);
    }
    if let Some(w) = fp::enc(insn, code)? {
        return Ok(w);
    }
    if let Some(w) = mem::enc(insn, code)? {
        return Ok(w);
    }
    Err(EncodeError::Unsupported)
}

// ---------------------------------------------------------------------------
// Shared field / operand readers.
// ---------------------------------------------------------------------------

/// The architectural register number of operand `n` (any register kind).
#[inline]
pub(crate) fn reg(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    match insn.op(n) {
        Operand::Reg { reg, .. } => Ok(reg.number() as u32),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The register number of operand `n`, requiring class `class`.
#[inline]
fn reg_cls(insn: &Instruction, n: usize, class: RegClass) -> Result<u32, EncodeError> {
    match insn.op(n) {
        Operand::Reg { reg, .. } if reg.class() == class => Ok(reg.number() as u32),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// A `Z` (scalable-vector) register number at operand `n`.
#[inline]
pub(crate) fn z(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    reg_cls(insn, n, RegClass::Sve)
}

/// A predicate `P` register number at operand `n`.
#[inline]
pub(crate) fn p(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    reg_cls(insn, n, RegClass::Predicate)
}

/// A general-purpose register number at operand `n`.
#[inline]
pub(crate) fn g(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    reg_cls(insn, n, RegClass::Gp)
}

/// A scalar-FP (`B/H/S/D`) register number at operand `n`.
#[inline]
pub(crate) fn sfp(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    reg_cls(insn, n, RegClass::ScalarFp)
}

/// The lane index of an indexed `Z` operand at `n`.
#[inline]
pub(crate) fn lane(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    match insn.op(n) {
        Operand::Reg {
            lane: Some(l), ..
        } => Ok(l as u32),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// An unsigned immediate value at operand `n` (signed values reinterpreted).
#[inline]
pub(crate) fn imm(insn: &Instruction, n: usize) -> Result<u64, EncodeError> {
    match insn.op(n) {
        Operand::ImmUnsigned(v) | Operand::ImmLogical(v) => Ok(v),
        Operand::ImmSigned(v) | Operand::ImmSignedDec(v) => Ok(v as u64),
        Operand::ShiftAmount(v) => Ok(v as u64),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// A signed immediate value at operand `n`.
#[inline]
pub(crate) fn simm(insn: &Instruction, n: usize) -> Result<i64, EncodeError> {
    match insn.op(n) {
        Operand::ImmSigned(v) | Operand::ImmSignedDec(v) => Ok(v),
        Operand::ImmUnsigned(v) | Operand::ImmLogical(v) => Ok(v as i64),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The 2-bit `size` field for the element arrangement of `Z`/`P` operand `n`.
#[inline]
pub(crate) fn esize(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    match insn.op(n) {
        Operand::Reg {
            arr: Some(a), ..
        } => arr_size(a),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The 2-bit `size` field for a [`VectorArrangement`].
#[inline]
pub(crate) fn arr_size(a: VA) -> Result<u32, EncodeError> {
    match a {
        VA::Sb => Ok(0),
        VA::Sh => Ok(1),
        VA::Ss => Ok(2),
        VA::Sd => Ok(3),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The element-size index + width for an arrangement (`.b`=0/8 .. `.d`=3/64).
#[inline]
pub(crate) fn esize_of(a: VA) -> Result<(u32, u32), EncodeError> {
    let idx = arr_size(a)?;
    Ok((idx, 8u32 << idx))
}

/// The governing-predicate qualifier of operand `n`, if it is a predicate.
#[inline]
pub(crate) fn pred_qual(insn: &Instruction, n: usize) -> Option<PredQual> {
    match insn.op(n) {
        Operand::Reg { pred, .. } => pred,
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Shift-amount inverses (mirror sve_int's right/left_shift_amount).
// ---------------------------------------------------------------------------

/// Inverse of `right_shift_amount`: given a right-shift `amount` at element
/// arrangement `a`, return the 5-bit `tsz` and 3-bit `imm3`.
pub(crate) fn enc_right_shift(a: VA, amount: u32) -> Result<(u32, u32), EncodeError> {
    let (idx, es) = esize_of(a)?;
    if amount == 0 || amount > 2 * es {
        return Err(EncodeError::InvalidImmediate);
    }
    let val = 2 * es - amount; // in [es, 2*es-1]
    let tsz = val >> 3;
    let imm3 = val & 7;
    if tsz == 0 || (31 - (tsz & 0xf).leading_zeros()) != idx {
        return Err(EncodeError::InvalidImmediate);
    }
    Ok((tsz, imm3))
}

/// Inverse of `left_shift_amount`: given a left-shift `amount` at arrangement
/// `a`, return the 5-bit `tsz` and 3-bit `imm3`.
pub(crate) fn enc_left_shift(a: VA, amount: u32) -> Result<(u32, u32), EncodeError> {
    let (idx, es) = esize_of(a)?;
    let val = amount + es;
    if val < es || val >= 2 * es {
        return Err(EncodeError::InvalidImmediate);
    }
    let tsz = val >> 3;
    let imm3 = val & 7;
    if tsz == 0 || (31 - (tsz & 0xf).leading_zeros()) != idx {
        return Err(EncodeError::InvalidImmediate);
    }
    Ok((tsz, imm3))
}

// ---------------------------------------------------------------------------
// SVE pattern / multiplier tail (inverse of push_pattern_mul).
// ---------------------------------------------------------------------------

/// Recover `(pattern, imm4)` from the optional `SvePattern{, SveMul}` tail of
/// an INC/DEC/CNT element-count form, starting at operand `start`. The decoder
/// elides the pattern when it is `all` (0x1f) and the multiplier is 1, and
/// elides the multiplier when it is 1. `imm4` is the raw field (`mul - 1`).
pub(crate) fn read_pattern_mul(insn: &Instruction, start: usize) -> Result<(u32, u32), EncodeError> {
    let mut pattern = 0x1fu32; // default `all`
    let mut mul = 1u32;
    let mut i = start;
    if let Operand::SvePattern(pat) = insn.op(i) {
        pattern = pat as u32;
        i += 1;
    }
    if let Operand::SveMul(m) = insn.op(i) {
        mul = m as u32;
        i += 1;
    }
    let _ = i;
    if mul == 0 || mul > 16 {
        return Err(EncodeError::InvalidImmediate);
    }
    Ok((pattern & 0x1f, mul - 1))
}

/// Recover the SVE predicate `pattern` from operand `n`, defaulting to `all`
/// (0x1f) when the operand is absent (the decoder elides `all`).
pub(crate) fn read_pattern_opt(insn: &Instruction, n: usize) -> u32 {
    match insn.op(n) {
        Operand::SvePattern(pat) => (pat as u32) & 0x1f,
        _ => 0x1f,
    }
}

// ---------------------------------------------------------------------------
// Small constructors for the encoded word.
// ---------------------------------------------------------------------------

/// Shift a field `value` left into position `lo`.
#[inline]
pub(crate) fn fld(value: u32, lo: u32) -> u32 {
    value << lo
}
