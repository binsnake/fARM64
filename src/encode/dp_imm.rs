//! Encoder for Data Processing -- Immediate — the exact inverse of
//! [`crate::decode::dp_imm`].
//!
//! Dispatches on [`Instruction::code`] (the canonical encoding identity), then
//! branches on [`Instruction::mnemonic`] to recover the operand layout the
//! decoder produced (the preferred-disassembly alias, if any) and packs the raw
//! fields in reverse. It reconstructs the word purely from the instruction's
//! semantics — it never reads [`Instruction::word`].
//!
//! ## How aliases are inverted
//!
//! The decoder rewrites canonical encodings into aliases via field math; the
//! encoder undoes that math. For example a decoded `LSL <Wd>, <Wn>, #n` is the
//! `UBFM` encoding with `immr = (-n) MOD 32`, `imms = 31 - n`; given the alias
//! shift `n` (operand 2) we recompute `immr = (32 - n) MOD 32` and
//! `imms = 31 - n`. Every alias in this group is inverted the same way: read the
//! alias operands, derive the canonical `(immr, imms)` / register fields, pack.
//!
//! ## ip-relative forms
//!
//! `ADR`/`ADRP` carry an absolute [`Operand::Label`] target. The encoder
//! recovers the original PC-relative immediate from `target` and the
//! instruction's [`Instruction::ip`] (ADRP additionally pages both), exactly
//! reversing the decoder's base+immediate computation.

use crate::encode::bits::encode_bit_masks;
use crate::encode::EncodeError;
use crate::instruction::Instruction;
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::Operand;

type R = Result<u32, EncodeError>;

/// Encode a Data Processing -- Immediate instruction.
#[inline]
pub fn encode(insn: &Instruction) -> R {
    use Code::*;
    match insn.code() {
        Adr | Adrp => enc_pc_rel(insn),
        AddImm32 | AddImm64 | SubImm32 | SubImm64 | AddsImm32 | AddsImm64 | SubsImm32
        | SubsImm64 => enc_addsub_imm(insn),
        AddgImm | SubgImm => enc_addsub_tags(insn),
        SmaxImm32 | SmaxImm64 | SminImm32 | SminImm64 | UmaxImm32 | UmaxImm64 | UminImm32
        | UminImm64 => enc_minmax_imm(insn),
        AndImm32 | AndImm64 | OrrImm32 | OrrImm64 | EorImm32 | EorImm64 | AndsImm32
        | AndsImm64 => enc_logical_imm(insn),
        Movn32 | Movn64 | Movz32 | Movz64 | Movk32 | Movk64 => enc_move_wide(insn),
        Sbfm32 | Sbfm64 | Bfm32 | Bfm64 | Ubfm32 | Ubfm64 => enc_bitfield(insn),
        Extr32 | Extr64 => enc_extract(insn),
        _ => Err(EncodeError::Unsupported),
    }
}

// ---------------------------------------------------------------------------
// Small field/operand helpers.
// ---------------------------------------------------------------------------

/// The 5-bit register number of operand `n`, or an error if it is not a plain
/// register. SP-vs-ZR is irrelevant for the *number* (both are 31), so the
/// encoding-defined role is satisfied by just emitting the field.
#[inline]
fn reg_num(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    match insn.op(n) {
        Operand::Reg { reg, .. } => Ok(reg.number() as u32),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The unsigned immediate value of an `ImmUnsigned`/`ImmSigned`/`ImmLogical`
/// operand `n` (signed values reinterpreted as `u64`).
#[inline]
fn imm_u(insn: &Instruction, n: usize) -> Result<u64, EncodeError> {
    match insn.op(n) {
        Operand::ImmUnsigned(v) | Operand::ImmLogical(v) => Ok(v),
        Operand::ImmSigned(v) => Ok(v as u64),
        Operand::ShiftAmount(v) => Ok(v as u64),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// `true` if the instruction's code is one of the 64-bit (`sf == 1`) variants
/// in this group; used to choose `datasize`/`sf`.
#[inline]
fn is_64(code: Code) -> bool {
    use Code::*;
    matches!(
        code,
        AddImm64
            | SubImm64
            | AddsImm64
            | SubsImm64
            | AddgImm
            | SubgImm
            | AndImm64
            | OrrImm64
            | EorImm64
            | AndsImm64
            | Movn64
            | Movz64
            | Movk64
            | Sbfm64
            | Bfm64
            | Ubfm64
            | Extr64
            | SmaxImm64
            | SminImm64
            | UmaxImm64
            | UminImm64
    )
}

// ---------------------------------------------------------------------------
// 00x : PC-relative addressing (ADR / ADRP).
// ---------------------------------------------------------------------------

/// `ADR`/`ADRP`. Recover the 21-bit `immhi:immlo` from the absolute label and
/// `ip` (ADRP pages both base and target and shifts the immediate by 12).
fn enc_pc_rel(insn: &Instruction) -> R {
    let rd = reg_num(insn, 0)?;
    let target = match insn.op(1) {
        Operand::Label(v) => v,
        _ => return Err(EncodeError::InvalidOperand),
    };
    let ip = insn.ip();
    let is_adrp = insn.code() == Code::Adrp;

    let imm21: u32 = if is_adrp {
        // base = page(ip); imm = (page(target) - page(ip)) >> 12, as a signed
        // 21-bit value.
        let base = ip & !0xFFF;
        let tgt_page = target & !0xFFF;
        let diff = (tgt_page.wrapping_sub(base)) as i64;
        let imm = diff >> 12;
        if (imm >> 20) != 0 && (imm >> 20) != -1 {
            return Err(EncodeError::InvalidImmediate);
        }
        (imm as u32) & 0x1f_ffff
    } else {
        let diff = (target.wrapping_sub(ip)) as i64;
        if (diff >> 20) != 0 && (diff >> 20) != -1 {
            return Err(EncodeError::InvalidImmediate);
        }
        (diff as u32) & 0x1f_ffff
    };

    let immlo = imm21 & 0x3; // bits <1:0>
    let immhi = (imm21 >> 2) & 0x7_ffff; // bits <20:2>
    let op = if is_adrp { 1u32 } else { 0u32 };
    let word = (op << 31) | (immlo << 29) | (0b10000 << 24) | (immhi << 5) | rd;
    Ok(word)
}

// ---------------------------------------------------------------------------
// 010 : Add/subtract (immediate).
// ---------------------------------------------------------------------------

/// `ADD`/`ADDS`/`SUB`/`SUBS` (immediate) and the `MOV`/`CMP`/`CMN` aliases.
fn enc_addsub_imm(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();
    let sf = if is_64(code) { 1u32 } else { 0 };
    let (op, s) = match code {
        AddImm32 | AddImm64 => (0u32, 0u32),
        AddsImm32 | AddsImm64 => (0, 1),
        SubImm32 | SubImm64 => (1, 0),
        _ => (1, 1), // SubsImm*
    };

    let (rd, rn, imm12, sh) = match insn.mnemonic() {
        // MOV (to/from SP): ADD Rd, Rn, #0 with no shift.
        Mnemonic::Mov => {
            let rd = reg_num(insn, 0)?;
            let rn = reg_num(insn, 1)?;
            (rd, rn, 0u32, 0u32)
        }
        // CMP (SUBS) / CMN (ADDS): Rd is ZR(31), operands are [Rn, imm].
        Mnemonic::Cmp | Mnemonic::Cmn => {
            let rn = reg_num(insn, 0)?;
            let (imm12, sh) = imm_shifted(insn, 1)?;
            (31u32, rn, imm12, sh)
        }
        // Canonical ADD/ADDS/SUB/SUBS: [Rd, Rn, imm].
        _ => {
            let rd = reg_num(insn, 0)?;
            let rn = reg_num(insn, 1)?;
            let (imm12, sh) = imm_shifted(insn, 2)?;
            (rd, rn, imm12, sh)
        }
    };

    if imm12 > 0xfff {
        return Err(EncodeError::InvalidImmediate);
    }
    let word = (sf << 31)
        | (op << 30)
        | (s << 29)
        | (0b10001 << 24)
        | (sh << 22)
        | (imm12 << 10)
        | (rn << 5)
        | rd;
    Ok(word)
}

/// Read an add/sub `ImmShiftedMove` (binja renders the imm12 unshifted with a
/// trailing `lsl #0xc`) at operand `n`, returning `(imm12, sh)` where `sh` is
/// the shift selector bit (`1` for `lsl #12`).
fn imm_shifted(insn: &Instruction, n: usize) -> Result<(u32, u32), EncodeError> {
    match insn.op(n) {
        Operand::ImmShiftedMove { imm, lsl } => {
            let sh = match lsl {
                0 => 0u32,
                12 => 1,
                _ => return Err(EncodeError::InvalidImmediate),
            };
            Ok((imm as u32, sh))
        }
        // Tolerate a plain immediate operand (no shift).
        Operand::ImmUnsigned(v) | Operand::ImmLogical(v) => {
            if v > 0xfff {
                return Err(EncodeError::InvalidImmediate);
            }
            Ok((v as u32, 0))
        }
        Operand::ImmSigned(v) => {
            if !(0..=0xfff).contains(&v) {
                return Err(EncodeError::InvalidImmediate);
            }
            Ok((v as u32, 0))
        }
        _ => Err(EncodeError::InvalidOperand),
    }
}

// ---------------------------------------------------------------------------
// 011 : Add/subtract (immediate, with tags) — ADDG / SUBG.
// ---------------------------------------------------------------------------

/// `ADDG`/`SUBG`. Operands: [Rd, Rn, #(uimm6<<4 already), #uimm4].
fn enc_addsub_tags(insn: &Instruction) -> R {
    let op = if insn.code() == Code::SubgImm { 1u32 } else { 0 };
    let rd = reg_num(insn, 0)?;
    let rn = reg_num(insn, 1)?;
    let uimm6_scaled = imm_u(insn, 2)?;
    let uimm4 = imm_u(insn, 3)?;
    if uimm6_scaled & 0xf != 0 || (uimm6_scaled >> 4) > 0x3f || uimm4 > 0xf {
        return Err(EncodeError::InvalidImmediate);
    }
    let uimm6 = (uimm6_scaled >> 4) as u32;
    let uimm4 = uimm4 as u32;
    // sf=1, S=0, op2=0 fixed. Base pattern: 1 op 0 100011 0 uimm6 00 uimm4 Rn Rd
    let word = (1 << 31)
        | (op << 30)
        | (0b100011 << 23)
        | (uimm6 << 16)
        | (uimm4 << 10)
        | (rn << 5)
        | rd;
    Ok(word)
}

/// `SMAX`/`SMIN`/`UMAX`/`UMIN` (immediate, FEAT_CSSC). Operands: [Rd, Rn, #imm8]
/// where the imm8 is signed for SMAX/SMIN and unsigned for UMAX/UMIN. Inverse of
/// [`crate::decode::dp_imm`]'s `decode_minmax_imm`. Pattern:
/// `sf 0 0 100011 1 00 opc imm8 Rn Rd` with `opc<1>` = min, `opc<0>` = unsigned.
fn enc_minmax_imm(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();
    let sf = if is_64(code) { 1u32 } else { 0 };
    let (opc, is_signed) = match code {
        SmaxImm32 | SmaxImm64 => (0b00u32, true),
        UmaxImm32 | UmaxImm64 => (0b01, false),
        SminImm32 | SminImm64 => (0b10, true),
        _ => (0b11, false), // UminImm*
    };

    let rd = reg_num(insn, 0)?;
    let rn = reg_num(insn, 1)?;
    let imm8 = if is_signed {
        // Signed imm8: accept -128..=127 (and tolerate the 0..=255 raw form).
        let v = match insn.op(2) {
            Operand::ImmSigned(v) => v,
            Operand::ImmUnsigned(u) | Operand::ImmLogical(u) => u as i64,
            _ => return Err(EncodeError::InvalidOperand),
        };
        if !(-128..=255).contains(&v) {
            return Err(EncodeError::InvalidImmediate);
        }
        (v as u32) & 0xff
    } else {
        let v = imm_u(insn, 2)?;
        if v > 0xff {
            return Err(EncodeError::InvalidImmediate);
        }
        v as u32
    };

    // sf 0 0 100011 1 00 opc imm8 Rn Rd. Fixed: op(30)=0, S(29)=0,
    // word<28:23>=100011, word<22>=1, word<21:20>=00.
    let word = (sf << 31)
        | (0b100011 << 23)
        | (1 << 22)
        | (opc << 18)
        | (imm8 << 10)
        | (rn << 5)
        | rd;
    Ok(word)
}

// ---------------------------------------------------------------------------
// 100 : Logical (immediate).
// ---------------------------------------------------------------------------

/// `AND`/`ORR`/`EOR`/`ANDS` (logical immediate) and the `MOV`/`TST` aliases.
fn enc_logical_imm(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();
    let sf = if is_64(code) { 1u32 } else { 0 };
    let datasize = if sf == 1 { 64u32 } else { 32 };
    let opc = match code {
        AndImm32 | AndImm64 => 0b00u32,
        OrrImm32 | OrrImm64 => 0b01,
        EorImm32 | EorImm64 => 0b10,
        _ => 0b11, // ANDS
    };

    let (rd, rn, value) = match insn.mnemonic() {
        // TST Rn, #imm : ANDS with Rd == ZR(31). Operands [Rn, imm].
        Mnemonic::Tst => {
            let rn = reg_num(insn, 0)?;
            let v = imm_u(insn, 1)?;
            (31u32, rn, v)
        }
        // MOV Rd, #imm : ORR with Rn == ZR(31). Operands [Rd, imm].
        Mnemonic::Mov => {
            let rd = reg_num(insn, 0)?;
            let v = imm_u(insn, 1)?;
            (rd, 31u32, v)
        }
        // Canonical AND/ORR/EOR/ANDS: [Rd, Rn, imm].
        _ => {
            let rd = reg_num(insn, 0)?;
            let rn = reg_num(insn, 1)?;
            let v = imm_u(insn, 2)?;
            (rd, rn, v)
        }
    };

    let value = if datasize == 32 { value & 0xffff_ffff } else { value };
    let (n, immr, imms) = encode_bit_masks(value, datasize).ok_or(EncodeError::InvalidImmediate)?;

    let word = (sf << 31)
        | (opc << 29)
        | (0b100100 << 23)
        | (n << 22)
        | (immr << 16)
        | (imms << 10)
        | (rn << 5)
        | rd;
    Ok(word)
}

// ---------------------------------------------------------------------------
// 101 : Move wide (immediate).
// ---------------------------------------------------------------------------

/// `MOVN`/`MOVZ`/`MOVK` and the `MOV` alias (which decodes to a MOVZ/MOVN field
/// triple). The canonical forms carry an `ImmShiftedMove{imm16, lsl}`; the MOV
/// alias carries a fully-materialized `ImmSigned` value that we must factor back
/// into `(imm16, hw)`.
fn enc_move_wide(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();
    let sf = if is_64(code) { 1u32 } else { 0 };
    let datasize = if sf == 1 { 64u32 } else { 32 };
    let opc = match code {
        Movn32 | Movn64 => 0b00u32,
        Movz32 | Movz64 => 0b10,
        _ => 0b11, // MOVK
    };

    let rd = reg_num(insn, 0)?;

    let (imm16, hw) = match insn.op(1) {
        // Canonical MOVN/MOVZ/MOVK: raw imm16 with its LSL.
        Operand::ImmShiftedMove { imm, lsl } => {
            let hw = lsl / 16;
            (imm as u32, hw as u32)
        }
        // MOV alias: a fully materialized value. Factor it back to imm16 << hw*16
        // (MOVZ) or NOT(imm16 << hw*16) (MOVN), per the original opc.
        Operand::ImmSigned(_) | Operand::ImmUnsigned(_) => {
            let raw = imm_u(insn, 1)?;
            let raw = if datasize == 32 { raw & 0xffff_ffff } else { raw };
            match opc {
                0b10 => factor_movz(raw, datasize)?,
                0b00 => factor_movn(raw, datasize)?,
                _ => return Err(EncodeError::InvalidOperand),
            }
        }
        _ => return Err(EncodeError::InvalidOperand),
    };

    // 32-bit forms must have hw<1> == 0.
    if sf == 0 && (hw & 0b10) != 0 {
        return Err(EncodeError::InvalidImmediate);
    }
    if imm16 > 0xffff {
        return Err(EncodeError::InvalidImmediate);
    }

    let word = (sf << 31)
        | (opc << 29)
        | (0b100101 << 23)
        | (hw << 21)
        | (imm16 << 5)
        | rd;
    Ok(word)
}

/// Factor a MOVZ value into `(imm16, hw)`: the value must be a 16-bit field at a
/// 16-bit-aligned position with all other bits zero.
fn factor_movz(value: u64, datasize: u32) -> Result<(u32, u32), EncodeError> {
    let lanes = datasize / 16;
    for hw in 0..lanes {
        let shift = hw * 16;
        if (value & !(0xffffu64 << shift)) == 0 {
            return Ok((((value >> shift) & 0xffff) as u32, hw));
        }
    }
    Err(EncodeError::InvalidImmediate)
}

/// Factor a MOVN value into `(imm16, hw)`: `NOT(value)` (truncated to datasize)
/// must be a single aligned 16-bit field.
fn factor_movn(value: u64, datasize: u32) -> Result<(u32, u32), EncodeError> {
    let inv = if datasize == 32 {
        (!value) & 0xffff_ffff
    } else {
        !value
    };
    let lanes = datasize / 16;
    for hw in 0..lanes {
        let shift = hw * 16;
        if (inv & !(0xffffu64 << shift)) == 0 {
            return Ok((((inv >> shift) & 0xffff) as u32, hw));
        }
    }
    Err(EncodeError::InvalidImmediate)
}

// ---------------------------------------------------------------------------
// 110 : Bitfield (SBFM / BFM / UBFM) and the alias family.
// ---------------------------------------------------------------------------

/// `SBFM`/`BFM`/`UBFM` plus every bitfield alias. Recovers `(immr, imms)` by
/// inverting the decoder's per-alias math.
fn enc_bitfield(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();
    let sf = if is_64(code) { 1u32 } else { 0 };
    let n = sf; // N must equal sf for the bitfield group.
    let datasize = if sf == 1 { 64u32 } else { 32 };
    let max = datasize - 1;
    let opc = match code {
        Sbfm32 | Sbfm64 => 0b00u32,
        Bfm32 | Bfm64 => 0b01,
        _ => 0b10, // UBFM
    };

    let (rn, immr, imms) = recover_bitfield_fields(insn, datasize, max, opc)?;

    if immr > max || imms > max {
        return Err(EncodeError::InvalidImmediate);
    }
    let rd = reg_num(insn, 0)?;
    let word = (sf << 31)
        | (opc << 29)
        | (0b100110 << 23)
        | (n << 22)
        | (immr << 16)
        | (imms << 10)
        | (rn << 5)
        | rd;
    Ok(word)
}

/// Recover `(rn, immr, imms)` from the (alias-resolved) operands of a bitfield
/// instruction. `opc` selects SBFM(00)/BFM(01)/UBFM(10).
fn recover_bitfield_fields(
    insn: &Instruction,
    datasize: u32,
    max: u32,
    opc: u32,
) -> Result<(u32, u32, u32), EncodeError> {
    let m = insn.mnemonic();
    match m {
        // --- sign/zero extends: immr=0, imms in {7,15,31}. Rn is operand 1. ---
        Mnemonic::Sxtb | Mnemonic::Uxtb => Ok((reg_num(insn, 1)?, 0, 7)),
        Mnemonic::Sxth | Mnemonic::Uxth => Ok((reg_num(insn, 1)?, 0, 15)),
        Mnemonic::Sxtw => Ok((reg_num(insn, 1)?, 0, 31)),

        // --- ASR (SBFM) / LSR (UBFM): immr = #shift, imms = max. ---
        Mnemonic::Asr | Mnemonic::Lsr => {
            let rn = reg_num(insn, 1)?;
            let shift = imm_u(insn, 2)? as u32;
            Ok((rn, shift, max))
        }

        // --- LSL (UBFM): immr = (datasize - n) MOD datasize, imms = max - n. ---
        Mnemonic::Lsl => {
            let rn = reg_num(insn, 1)?;
            let shift = imm_u(insn, 2)? as u32;
            if shift > max {
                return Err(EncodeError::InvalidImmediate);
            }
            let immr = (datasize - shift) % datasize;
            let imms = max - shift;
            Ok((rn, immr, imms))
        }

        // --- SBFIZ / UBFIZ Rd, Rn, #lsb, #width ---
        Mnemonic::Sbfiz | Mnemonic::Ubfiz => {
            let rn = reg_num(insn, 1)?;
            let lsb = imm_u(insn, 2)? as u32;
            let width = imm_u(insn, 3)? as u32;
            if width == 0 {
                return Err(EncodeError::InvalidImmediate);
            }
            let immr = (datasize - lsb) % datasize;
            let imms = width - 1;
            Ok((rn, immr, imms))
        }

        // --- SBFX / UBFX / BFXIL Rd, Rn, #lsb, #width ---
        Mnemonic::Sbfx | Mnemonic::Ubfx | Mnemonic::Bfxil => {
            let rn = reg_num(insn, 1)?;
            let lsb = imm_u(insn, 2)? as u32;
            let width = imm_u(insn, 3)? as u32;
            if width == 0 {
                return Err(EncodeError::InvalidImmediate);
            }
            let immr = lsb;
            let imms = lsb + width - 1;
            Ok((rn, immr, imms))
        }

        // --- BFI Rd, Rn, #lsb, #width ---
        Mnemonic::Bfi => {
            let rn = reg_num(insn, 1)?;
            let lsb = imm_u(insn, 2)? as u32;
            let width = imm_u(insn, 3)? as u32;
            if width == 0 {
                return Err(EncodeError::InvalidImmediate);
            }
            let immr = (datasize - lsb) % datasize;
            let imms = width - 1;
            Ok((rn, immr, imms))
        }

        // --- BFC Rd, #lsb, #width : Rn == ZR(31), operands [Rd, lsb, width] ---
        Mnemonic::Bfc => {
            let lsb = imm_u(insn, 1)? as u32;
            let width = imm_u(insn, 2)? as u32;
            if width == 0 {
                return Err(EncodeError::InvalidImmediate);
            }
            let immr = (datasize - lsb) % datasize;
            let imms = width - 1;
            Ok((31, immr, imms))
        }

        // --- canonical SBFM/BFM/UBFM Rd, Rn, #immr, #imms ---
        _ => {
            // Guard against an unexpected alias mnemonic landing on a bitfield
            // code without a matching arm above.
            let _ = opc;
            let rn = reg_num(insn, 1)?;
            let immr = imm_u(insn, 2)? as u32;
            let imms = imm_u(insn, 3)? as u32;
            Ok((rn, immr, imms))
        }
    }
}

// ---------------------------------------------------------------------------
// 111 : Extract (EXTR) and the ROR alias.
// ---------------------------------------------------------------------------

/// `EXTR` and its `ROR` alias (when Rn == Rm).
fn enc_extract(insn: &Instruction) -> R {
    let sf = if is_64(insn.code()) { 1u32 } else { 0 };
    let n = sf;
    let rd = reg_num(insn, 0)?;

    let (rn, rm, imms) = match insn.mnemonic() {
        // ROR Rd, Rn, #lsb : Rm == Rn, imms = lsb. Operands [Rd, Rn, imm].
        Mnemonic::Ror => {
            let rn = reg_num(insn, 1)?;
            let imms = imm_u(insn, 2)? as u32;
            (rn, rn, imms)
        }
        // Canonical EXTR Rd, Rn, Rm, #lsb. Operands [Rd, Rn, Rm, imm].
        _ => {
            let rn = reg_num(insn, 1)?;
            let rm = reg_num(insn, 2)?;
            let imms = imm_u(insn, 3)? as u32;
            (rn, rm, imms)
        }
    };

    if sf == 0 && (imms & 0b10_0000) != 0 {
        return Err(EncodeError::InvalidImmediate);
    }
    if imms > 0x3f {
        return Err(EncodeError::InvalidImmediate);
    }
    // 00100111 sf=.. N0 0 Rm imms Rn Rd ; base group 100111, op21=00, o0=0.
    let word = (sf << 31)
        | (0b00100111 << 23)
        | (n << 22)
        | (rm << 16)
        | (imms << 10)
        | (rn << 5)
        | rd;
    Ok(word)
}

#[cfg(test)]
mod tests {
    use crate::features::FeatureSet;
    use crate::instruction::Instruction;

    /// Decode a word then re-encode it and require the exact same word back.
    fn rt(word: u32) {
        let mut insn = Instruction::default();
        crate::decode::decode_into(word, 0x1000, FeatureSet::ALL, &mut insn);
        assert!(!insn.is_invalid(), "word {word:#010x} failed to decode");
        let got = insn
            .encode()
            .unwrap_or_else(|e| panic!("encode of {word:#010x} ({:?}) failed: {e:?}", insn.code()));
        assert_eq!(
            got, word,
            "round-trip mismatch for {word:#010x}: re-encoded {got:#010x} (code={:?}, mnem={:?})",
            insn.code(),
            insn.mnemonic()
        );
    }

    #[test]
    fn dp_imm_known_words() {
        // ADD w0, w1, #1
        rt(0x1100_0420);
        // ADD x0, x1, #1
        rt(0x9100_0420);
        // SUB x0, x1, #1
        rt(0xD100_0420);
        // ADD x0, x1, #1, lsl #12
        rt(0x9140_0420);
        // ADDS / CMP forms: CMP x1, #1  (SUBS xzr, x1, #1)
        rt(0xF100_043F);
        // MOV x0, sp (ADD x0, sp, #0)
        rt(0x9100_03E0);
        // AND x0, x0, #0xff
        rt(0x9240_1C00);
        // ORR -> MOV x0, #0xffffffffffffffff style: ORR x0, xzr, #imm
        // (use ORR w0, w0, #1)
        rt(0x3200_0000);
        // TST x0, #1  (ANDS xzr, x0, #1)
        rt(0xF240_001F);
        // MOVZ x0, #1
        rt(0xD280_0020);
        // MOVZ x0, #1, lsl #16
        rt(0xD2A0_0020);
        // MOVK w0, #1
        rt(0x7280_0020);
        // MOVN x0, #0 -> MOV x0, #-1
        rt(0x9280_0000);
        // SBFM / ASR x0, x1, #4
        rt(0x9344_FC20);
        // UBFM / LSL x0, x1, #4
        rt(0xD37C_EC20);
        // LSR x0, x1, #4
        rt(0xD344_FC20);
        // BFM / BFI x0, x1, #4, #4
        rt(0xB37C_0C20);
        // EXTR x0, x1, x2, #4  (ROR if rn==rm)
        rt(0x93C2_1020);
        // ADR x0, .+0
        rt(0x1000_0000);
        // ADRP x0, .+0
        rt(0x9000_0000);
    }

    #[test]
    fn dp_imm_cssc_minmax() {
        rt(0x91C01420); // smax x0, x1, #5
        rt(0x11C3F483); // smax w3, w4, #-3
        rt(0x91CA00C5); // smin x5, x6, #-128
        rt(0x11C9FD07); // smin w7, w8, #127
        rt(0x91C7FD49); // umax x9, x10, #255
        rt(0x11C4018B); // umax w11, w12, #0
        rt(0x91CF21CD); // umin x13, x14, #200
        rt(0x11CC060F); // umin w15, w16, #1
    }
}
