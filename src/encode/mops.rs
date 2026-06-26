//! Encoder for FEAT_MOPS — Memory Copy / Memory Set — the exact inverse of
//! [`crate::decode::mops`].
//!
//! Reconstructs the word purely from the canonical [`Code`] and the three
//! register operands; it never reads [`crate::instruction::Instruction::word`].
//! The instruction layout is
//!
//! ```text
//! [0 0 0 1 1][o][0 1][ op1 ][ Rn ][ op2 ][0 1][ Rs ][ Rd ]
//! ```
//!
//! where `(o, op1, op2)` come from [`mops_fields`] and `Rn`/`Rs`/`Rd` from the
//! operands (see [`crate::decode::mops`] for the per-family operand order).

use crate::encode::EncodeError;
use crate::instruction::Instruction;
use crate::mnemonic::Code;
use crate::operand::{MemIndexMode, Operand};

type R = Result<u32, EncodeError>;

/// `Code -> (family/o, op1, op2)` for every MOPS encoding; `None` for any other
/// code. Shared by [`is_mops`] and [`encode`].
fn mops_fields(code: Code) -> Option<(u32, u32, u32)> {
    use Code::*;
    Some(match code {
        Cpyfp => (0, 0, 0),
        Cpyfpwt => (0, 0, 1),
        Cpyfprt => (0, 0, 2),
        Cpyfpt => (0, 0, 3),
        Cpyfpwn => (0, 0, 4),
        Cpyfpwtwn => (0, 0, 5),
        Cpyfprtwn => (0, 0, 6),
        Cpyfptwn => (0, 0, 7),
        Cpyfprn => (0, 0, 8),
        Cpyfpwtrn => (0, 0, 9),
        Cpyfprtrn => (0, 0, 10),
        Cpyfptrn => (0, 0, 11),
        Cpyfpn => (0, 0, 12),
        Cpyfpwtn => (0, 0, 13),
        Cpyfprtn => (0, 0, 14),
        Cpyfptn => (0, 0, 15),
        Cpyfm => (0, 2, 0),
        Cpyfmwt => (0, 2, 1),
        Cpyfmrt => (0, 2, 2),
        Cpyfmt => (0, 2, 3),
        Cpyfmwn => (0, 2, 4),
        Cpyfmwtwn => (0, 2, 5),
        Cpyfmrtwn => (0, 2, 6),
        Cpyfmtwn => (0, 2, 7),
        Cpyfmrn => (0, 2, 8),
        Cpyfmwtrn => (0, 2, 9),
        Cpyfmrtrn => (0, 2, 10),
        Cpyfmtrn => (0, 2, 11),
        Cpyfmn => (0, 2, 12),
        Cpyfmwtn => (0, 2, 13),
        Cpyfmrtn => (0, 2, 14),
        Cpyfmtn => (0, 2, 15),
        Cpyfe => (0, 4, 0),
        Cpyfewt => (0, 4, 1),
        Cpyfert => (0, 4, 2),
        Cpyfet => (0, 4, 3),
        Cpyfewn => (0, 4, 4),
        Cpyfewtwn => (0, 4, 5),
        Cpyfertwn => (0, 4, 6),
        Cpyfetwn => (0, 4, 7),
        Cpyfern => (0, 4, 8),
        Cpyfewtrn => (0, 4, 9),
        Cpyfertrn => (0, 4, 10),
        Cpyfetrn => (0, 4, 11),
        Cpyfen => (0, 4, 12),
        Cpyfewtn => (0, 4, 13),
        Cpyfertn => (0, 4, 14),
        Cpyfetn => (0, 4, 15),
        Setp => (0, 6, 0),
        Setpt => (0, 6, 1),
        Setpn => (0, 6, 2),
        Setptn => (0, 6, 3),
        Setm => (0, 6, 4),
        Setmt => (0, 6, 5),
        Setmn => (0, 6, 6),
        Setmtn => (0, 6, 7),
        Sete => (0, 6, 8),
        Setet => (0, 6, 9),
        Seten => (0, 6, 10),
        Setetn => (0, 6, 11),
        Cpyp => (1, 0, 0),
        Cpypwt => (1, 0, 1),
        Cpyprt => (1, 0, 2),
        Cpypt => (1, 0, 3),
        Cpypwn => (1, 0, 4),
        Cpypwtwn => (1, 0, 5),
        Cpyprtwn => (1, 0, 6),
        Cpyptwn => (1, 0, 7),
        Cpyprn => (1, 0, 8),
        Cpypwtrn => (1, 0, 9),
        Cpyprtrn => (1, 0, 10),
        Cpyptrn => (1, 0, 11),
        Cpypn => (1, 0, 12),
        Cpypwtn => (1, 0, 13),
        Cpyprtn => (1, 0, 14),
        Cpyptn => (1, 0, 15),
        Cpym => (1, 2, 0),
        Cpymwt => (1, 2, 1),
        Cpymrt => (1, 2, 2),
        Cpymt => (1, 2, 3),
        Cpymwn => (1, 2, 4),
        Cpymwtwn => (1, 2, 5),
        Cpymrtwn => (1, 2, 6),
        Cpymtwn => (1, 2, 7),
        Cpymrn => (1, 2, 8),
        Cpymwtrn => (1, 2, 9),
        Cpymrtrn => (1, 2, 10),
        Cpymtrn => (1, 2, 11),
        Cpymn => (1, 2, 12),
        Cpymwtn => (1, 2, 13),
        Cpymrtn => (1, 2, 14),
        Cpymtn => (1, 2, 15),
        Cpye => (1, 4, 0),
        Cpyewt => (1, 4, 1),
        Cpyert => (1, 4, 2),
        Cpyet => (1, 4, 3),
        Cpyewn => (1, 4, 4),
        Cpyewtwn => (1, 4, 5),
        Cpyertwn => (1, 4, 6),
        Cpyetwn => (1, 4, 7),
        Cpyern => (1, 4, 8),
        Cpyewtrn => (1, 4, 9),
        Cpyertrn => (1, 4, 10),
        Cpyetrn => (1, 4, 11),
        Cpyen => (1, 4, 12),
        Cpyewtn => (1, 4, 13),
        Cpyertn => (1, 4, 14),
        Cpyetn => (1, 4, 15),
        Setgp => (1, 6, 0),
        Setgpt => (1, 6, 1),
        Setgpn => (1, 6, 2),
        Setgptn => (1, 6, 3),
        Setgm => (1, 6, 4),
        Setgmt => (1, 6, 5),
        Setgmn => (1, 6, 6),
        Setgmtn => (1, 6, 7),
        Setge => (1, 6, 8),
        Setget => (1, 6, 9),
        Setgen => (1, 6, 10),
        Setgetn => (1, 6, 11),
        _ => return None,
    })
}

/// `Code -> op2` for the FEAT_MOPS memory-set-with-tag *option* forms
/// (`SETGO*`). These share `(family, op1) == (1, 6)` but use `word<11:10> == 00`
/// (not `01`) and a fixed `word<20:16> == 11111` value field; only `[Xd]!, Xn!`
/// are operands. `None` for any other code.
fn setg_option_op2(code: Code) -> Option<u32> {
    use Code::*;
    Some(match code {
        SetgopMops => 0,
        SetgoptMops => 1,
        SetgopnMops => 2,
        SetgoptnMops => 3,
        SetgomMops => 4,
        SetgomtMops => 5,
        SetgomnMops => 6,
        SetgomtnMops => 7,
        SetgoeMops => 8,
        SetgoetMops => 9,
        SetgoenMops => 10,
        SetgoetnMops => 11,
        _ => return None,
    })
}

/// `true` if `code` is a FEAT_MOPS Memory Copy / Memory Set encoding.
#[inline]
pub fn is_mops(code: Code) -> bool {
    mops_fields(code).is_some() || setg_option_op2(code).is_some()
}

/// The 5-bit register number from a plain `Reg` operand.
#[inline]
fn reg(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    match insn.op(n) {
        Operand::Reg { reg, .. } => Ok(reg.number() as u32),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The 5-bit register number from a `RegBang` (`Xn!`) operand.
#[inline]
fn reg_bang(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    match insn.op(n) {
        Operand::RegBang(reg) => Ok(reg.number() as u32),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The 5-bit base register from a `[Xn]!` (`PreNoOffset`) memory operand.
#[inline]
fn mem_bang(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    match insn.op(n) {
        Operand::MemImm {
            base,
            imm: 0,
            mode: MemIndexMode::PreNoOffset,
        } => Ok(base.number() as u32),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Encode a Memory Copy / Memory Set (`FEAT_MOPS`) instruction.
pub fn encode(insn: &Instruction) -> R {
    // SETG*-option forms: `[Xd]!, Xn!` only, `word<11:10>==00`, value field `xzr`.
    if let Some(op2) = setg_option_op2(insn.code()) {
        let rd = mem_bang(insn, 0)?;
        let rs = reg_bang(insn, 1)?;
        // word<11:10> == 00 (the option marker) distinguishes these from the
        // value-register SETG forms (word<11:10> == 01).
        let word = (0b00011 << 27)
            | (1 << 26) // SETG family
            | (0b01 << 24)
            | (0b110 << 21) // op1 = 6
            | (0b11111 << 16) // value field = xzr
            | (op2 << 12)
            | (rs << 5)
            | rd;
        return Ok(word);
    }

    let (family, op1, op2) = mops_fields(insn.code()).ok_or(EncodeError::Unsupported)?;

    // Operand order differs between the copy and set families (see the decoder).
    let is_set = op1 == 6;
    let (rd, rn, rs) = if is_set {
        // `[Xd]!, Xs!, Xn` -> dst, size!, value.
        let rd = mem_bang(insn, 0)?;
        let rs = reg_bang(insn, 1)?;
        let rn = reg(insn, 2)?;
        (rd, rn, rs)
    } else {
        // `[Xd]!, [Xs]!, Xn!` -> dst, src, size!.
        let rd = mem_bang(insn, 0)?;
        let rn = mem_bang(insn, 1)?;
        let rs = reg_bang(insn, 2)?;
        (rd, rn, rs)
    };

    let word = (0b00011 << 27)
        | (family << 26)
        | (0b01 << 24)
        | (op1 << 21)
        | (rn << 16)
        | (op2 << 12)
        | (0b01 << 10)
        | (rs << 5)
        | rd;
    Ok(word)
}
