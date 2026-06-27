//! Encoder for the Apple IMPLEMENTATION-DEFINED instructions (AMX / GXF).
//!
//! The exact inverse of [`crate::decode::apple`]: rebuilds the 32-bit word from
//! the [`Code`] (op number) and operands. AMX words are
//! `0x0020_1000 | (op << 5) | operand`; GXF is `GEXIT` (`0x0020_1400`) and
//! `GENTER #imm5` (`0x0020_1420 | imm5`). Never panics; unsupported shapes
//! surface as [`EncodeError`].

use crate::encode::EncodeError;
use crate::instruction::Instruction;
use crate::mnemonic::Code;
use crate::operand::Operand;

/// `true` for every [`Code`] in the Apple AMX / GXF group.
#[inline]
pub(crate) fn is_apple(code: Code) -> bool {
    use Code::*;
    matches!(
        code,
        AmxLdx
            | AmxLdy
            | AmxStx
            | AmxSty
            | AmxLdz
            | AmxStz
            | AmxLdzi
            | AmxStzi
            | AmxExtrx
            | AmxExtry
            | AmxFma64
            | AmxFms64
            | AmxFma32
            | AmxFms32
            | AmxMac16
            | AmxFma16
            | AmxFms16
            | AmxSet
            | AmxClr
            | AmxVecint
            | AmxVecfp
            | AmxMatint
            | AmxMatfp
            | AmxGenlut
            | Genter
            | Gexit
    )
}

/// The general-purpose register number (0..=31) of operand 0.
#[inline]
fn amx_reg(insn: &Instruction) -> Result<u32, EncodeError> {
    match insn.op(0) {
        Operand::Reg { reg, .. } => Ok(reg.number() as u32),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Encode an Apple AMX / GXF instruction back to its 32-bit word.
pub(crate) fn encode(insn: &Instruction) -> Result<u32, EncodeError> {
    use Code::*;
    let code = insn.code();

    // GXF: not part of the AMX op-number space (these set word<10>).
    match code {
        Gexit => return Ok(0x0020_1400),
        Genter => {
            let imm5 = match insn.op(0) {
                Operand::ImmUnsigned(v) => v,
                _ => return Err(EncodeError::InvalidOperand),
            };
            if imm5 > 0x1f {
                return Err(EncodeError::InvalidImmediate);
            }
            return Ok(0x0020_1420 | imm5 as u32);
        }
        _ => {}
    }

    // AMX: 0x0020_1000 | (op << 5) | operand.
    let (op, operand): (u32, u32) = match code {
        AmxLdx => (0, amx_reg(insn)?),
        AmxLdy => (1, amx_reg(insn)?),
        AmxStx => (2, amx_reg(insn)?),
        AmxSty => (3, amx_reg(insn)?),
        AmxLdz => (4, amx_reg(insn)?),
        AmxStz => (5, amx_reg(insn)?),
        AmxLdzi => (6, amx_reg(insn)?),
        AmxStzi => (7, amx_reg(insn)?),
        AmxExtrx => (8, amx_reg(insn)?),
        AmxExtry => (9, amx_reg(insn)?),
        AmxFma64 => (10, amx_reg(insn)?),
        AmxFms64 => (11, amx_reg(insn)?),
        AmxFma32 => (12, amx_reg(insn)?),
        AmxFms32 => (13, amx_reg(insn)?),
        AmxMac16 => (14, amx_reg(insn)?),
        AmxFma16 => (15, amx_reg(insn)?),
        AmxFms16 => (16, amx_reg(insn)?),
        AmxSet => (17, 0),
        AmxClr => (17, 1),
        AmxVecint => (18, amx_reg(insn)?),
        AmxVecfp => (19, amx_reg(insn)?),
        AmxMatint => (20, amx_reg(insn)?),
        AmxMatfp => (21, amx_reg(insn)?),
        AmxGenlut => (22, amx_reg(insn)?),
        _ => return Err(EncodeError::Unsupported),
    };
    Ok(0x0020_1000 | (op << 5) | (operand & 0x1f))
}
