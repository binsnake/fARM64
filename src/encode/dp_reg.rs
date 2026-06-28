//! Encoder for Data Processing -- Register — the exact inverse of
//! [`crate::decode::dp_reg`].
//!
//! Dispatches on [`Instruction::code`] (the canonical encoding identity), then
//! branches on [`Instruction::mnemonic`] to recover the operand layout the
//! decoder produced (the preferred-disassembly alias, if any) and packs the raw
//! fields in reverse. It reconstructs the word purely from the instruction's
//! semantics — it never reads [`Instruction::word`].
//!
//! ## Sub-classes covered
//!
//! Every sub-class of the decoder is inverted here: logical (shifted register)
//! with `MOV`/`MVN`/`TST`; add/subtract (shifted) with `CMP`/`CMN`/`NEG`/`NEGS`;
//! add/subtract (extended) with `CMP`/`CMN` and the `LSL`/`UXTX` option
//! spelling; add/subtract (with carry) with `NGC`/`NGCS`; conditional compare
//! (register and immediate); conditional select with `CSET`/`CSETM`/`CINC`/
//! `CINV`/`CNEG` (the condition is inverted back); data-processing 3-source with
//! `MUL`/`MNEG`/`SMULL`/`SMNEGL`/`UMULL`/`UMNEGL` and the `SMULH`/`UMULH` /
//! widening shapes; data-processing 2-source (`UDIV`/`SDIV`, the `LSL`/`LSR`/
//! `ASR`/`ROR` variable-shift aliases, `CRC32*`, `SUBP`/`SUBPS`, `IRG`/`GMI`,
//! `PACGA`); data-processing 1-source (`RBIT`/`REV*`/`CLZ`/`CLS` and the PAuth
//! `PAC*`/`AUT*`/`XPAC*` forms); and `RMIF`/`SETF8`/`SETF16`.
//!
//! ## How aliases are inverted
//!
//! The decoder rewrites canonical encodings into aliases (`CSET` is `CSINC`
//! with `Rm==Rn==ZR` and an inverted condition, `MOV` is `ORR` with `Rn==ZR`,
//! ...). The encoder restores the canonical register/condition fields and packs
//! them, mirroring the decoder's per-alias math.

use crate::encode::EncodeError;
use crate::enums::{Condition, ShiftType};
use crate::instruction::Instruction;
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::Operand;

type R = Result<u32, EncodeError>;

/// Encode a Data Processing -- Register instruction.
#[inline]
pub fn encode(insn: &Instruction) -> R {
    use Code::*;
    match insn.code() {
        // Logical (shifted register).
        AndShifted32 | AndShifted64 | BicShifted32 | BicShifted64 | OrrShifted32 | OrrShifted64
        | OrnShifted32 | OrnShifted64 | EorShifted32 | EorShifted64 | EonShifted32
        | EonShifted64 | AndsShifted32 | AndsShifted64 | BicsShifted32 | BicsShifted64 => {
            enc_logical_shifted(insn)
        }
        // Add/subtract (shifted register).
        AddShifted32 | AddShifted64 | AddsShifted32 | AddsShifted64 | SubShifted32
        | SubShifted64 | SubsShifted32 | SubsShifted64 => enc_addsub_shifted(insn),
        // Add/subtract (extended register).
        AddExtended32 | AddExtended64 | AddsExtended32 | AddsExtended64 | SubExtended32
        | SubExtended64 | SubsExtended32 | SubsExtended64 => enc_addsub_extended(insn),
        // Add/subtract (with carry).
        Adc32 | Adc64 | Adcs32 | Adcs64 | Sbc32 | Sbc64 | Sbcs32 | Sbcs64 => {
            enc_addsub_carry(insn)
        }
        // Flag manipulation.
        Rmif => enc_rmif(insn),
        Setf8 | Setf16 => enc_setf(insn),
        // Conditional compare (register / immediate).
        CcmnReg32 | CcmnReg64 | CcmnImm32 | CcmnImm64 | CcmpReg32 | CcmpReg64 | CcmpImm32
        | CcmpImm64 => enc_cond_compare(insn),
        // Conditional select.
        Csel32 | Csel64 | Csinc32 | Csinc64 | Csinv32 | Csinv64 | Csneg32 | Csneg64 => {
            enc_cond_select(insn)
        }
        // Data-processing (3 source).
        Madd32 | Madd64 | Msub32 | Msub64 | Smaddl | Smsubl | Smulh | Umaddl | Umsubl | Umulh
        | Maddpt | Msubpt => enc_dp_3source(insn),
        // Add/subtract checked-pointer (FEAT_CPA scalar).
        Addpt | Subpt => enc_addsub_pt(insn),
        // Data-processing (2 source).
        Udiv32 | Udiv64 | Sdiv32 | Sdiv64 | Lslv32 | Lslv64 | Lsrv32 | Lsrv64 | Asrv32
        | Asrv64 | Rorv32 | Rorv64 | Crc32b | Crc32h | Crc32w | Crc32x | Crc32cb | Crc32ch
        | Crc32cw | Crc32cx | SubpDp | SubpsDp | IrgDp | GmiDp | Pacga | SmaxReg32 | SmaxReg64
        | SminReg32 | SminReg64 | UmaxReg32 | UmaxReg64 | UminReg32 | UminReg64 => {
            enc_dp_2source(insn)
        }
        // Data-processing (1 source).
        Rbit32 | Rbit64 | Rev1632 | Rev1664 | Rev3232 | Rev3264 | Rev32Bit | Rev64Bit | Clz32
        | Clz64 | Cls32 | Cls64 | Abs32 | Abs64 | Cnt32 | Cnt64 | Ctz32 | Ctz64 => {
            enc_dp_1source_basic(insn)
        }
        PaciaDp | PacibDp | PacdaDp | PacdbDp | AutiaDp | AutibDp | AutdaDp | AutdbDp
        | PacizaDp | PacizbDp | PacdzaDp | PacdzbDp | AutizaDp | AutizbDp | AutdzaDp
        | AutdzbDp | XpaciDp | XpacdDp
        | Paciasppc | Pacibsppc | Pacnbiasppc | Pacnbibsppc | Autiasppcr | Autibsppcr => {
            enc_dp_1source_pauth(insn)
        }
        _ => Err(EncodeError::Unsupported),
    }
}

// ---------------------------------------------------------------------------
// Small field/operand helpers.
// ---------------------------------------------------------------------------

/// The 5-bit register number of operand `n`, or an error if it is not a plain
/// register. SP-vs-ZR is encoding-defined; the *number* is 31 for both.
#[inline]
fn reg_num(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    match insn.op(n) {
        Operand::Reg { reg, .. } => Ok(reg.number() as u32),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The unsigned immediate value of an immediate operand `n`.
#[inline]
fn imm_u(insn: &Instruction, n: usize) -> Result<u64, EncodeError> {
    match insn.op(n) {
        Operand::ImmUnsigned(v) | Operand::ImmLogical(v) => Ok(v),
        Operand::ImmSigned(v) => Ok(v as u64),
        Operand::ShiftAmount(v) => Ok(v as u64),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The condition of operand `n`.
#[inline]
fn cond_of(insn: &Instruction, n: usize) -> Result<Condition, EncodeError> {
    match insn.op(n) {
        Operand::Cond(c) => Ok(c),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// `true` if the instruction's code is one of the 64-bit (`sf == 1`) variants
/// that follow the simple `*32`/`*64` naming convention in this group.
#[inline]
fn is_64(code: Code) -> bool {
    use Code::*;
    matches!(
        code,
        AndShifted64
            | BicShifted64
            | OrrShifted64
            | OrnShifted64
            | EorShifted64
            | EonShifted64
            | AndsShifted64
            | BicsShifted64
            | AddShifted64
            | AddsShifted64
            | SubShifted64
            | SubsShifted64
            | AddExtended64
            | AddsExtended64
            | SubExtended64
            | SubsExtended64
            | Adc64
            | Adcs64
            | Sbc64
            | Sbcs64
            | CcmnReg64
            | CcmnImm64
            | CcmpReg64
            | CcmpImm64
            | Csel64
            | Csinc64
            | Csinv64
            | Csneg64
            | Madd64
            | Msub64
            | Udiv64
            | Sdiv64
            | Lslv64
            | Lsrv64
            | Asrv64
            | Rorv64
            | Rbit64
            | Rev1664
            | Rev3264
            | Rev64Bit
            | Clz64
            | Cls64
            | Abs64
            | Cnt64
            | Ctz64
            | SmaxReg64
            | SminReg64
            | UmaxReg64
            | UminReg64
    )
}

/// Extract a plain GP register operand's `(number, shift)`, where `shift` is the
/// folded `Option<(ShiftType, amt)>` decoration (a `LSL #0` is folded to
/// `None`).
#[inline]
fn reg_with_shift(
    insn: &Instruction,
    n: usize,
) -> Result<(u32, Option<(ShiftType, u8)>), EncodeError> {
    match insn.op(n) {
        Operand::Reg { reg, shift, .. } => Ok((reg.number() as u32, shift)),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Recover the `(shift_field, imm6)` of a shifted-register operand `n`, given the
/// operand-size `sf` (32-bit forms forbid `imm6<5> == 1`). A folded `None` shift
/// means `LSL #0`.
#[inline]
fn recover_shift(insn: &Instruction, n: usize, sf: u32) -> Result<(u32, u32), EncodeError> {
    let (_, shift) = reg_with_shift(insn, n)?;
    let (st, amt) = shift.unwrap_or((ShiftType::Lsl, 0));
    let sh = match st {
        ShiftType::Lsl => 0u32,
        ShiftType::Lsr => 1,
        ShiftType::Asr => 2,
        ShiftType::Ror => 3,
        _ => return Err(EncodeError::InvalidOperand),
    };
    let imm6 = amt as u32;
    if imm6 > 0x3f {
        return Err(EncodeError::InvalidImmediate);
    }
    if sf == 0 && (imm6 & 0b10_0000) != 0 {
        return Err(EncodeError::InvalidImmediate);
    }
    Ok((sh, imm6))
}

// ---------------------------------------------------------------------------
// Logical (shifted register): AND/BIC/ORR/ORN/EOR/EON/ANDS/BICS.
// ---------------------------------------------------------------------------

/// `AND`/`BIC`/`ORR`/`ORN`/`EOR`/`EON`/`ANDS`/`BICS` (shifted register) and the
/// `MOV`/`MVN`/`TST` aliases.
fn enc_logical_shifted(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();
    let sf = if is_64(code) { 1u32 } else { 0 };
    let (opc, n) = match code {
        AndShifted32 | AndShifted64 => (0b00u32, 0u32),
        BicShifted32 | BicShifted64 => (0b00, 1),
        OrrShifted32 | OrrShifted64 => (0b01, 0),
        OrnShifted32 | OrnShifted64 => (0b01, 1),
        EorShifted32 | EorShifted64 => (0b10, 0),
        EonShifted32 | EonShifted64 => (0b10, 1),
        AndsShifted32 | AndsShifted64 => (0b11, 0),
        _ => (0b11, 1), // BICS
    };

    let (rd, rn, rm, sh, imm6) = match insn.mnemonic() {
        // MOV Rd, Rm : ORR Rd, ZR, Rm (LSL #0). Operands [Rd, Rm].
        Mnemonic::Mov => {
            let rd = reg_num(insn, 0)?;
            let rm = reg_num(insn, 1)?;
            (rd, 31u32, rm, 0u32, 0u32)
        }
        // MVN Rd, Rm{,shift} : ORN Rd, ZR, Rm{,shift}. Operands [Rd, Rm].
        Mnemonic::Mvn => {
            let rd = reg_num(insn, 0)?;
            let rm = reg_num(insn, 1)?;
            let (sh, imm6) = recover_shift(insn, 1, sf)?;
            (rd, 31u32, rm, sh, imm6)
        }
        // TST Rn, Rm{,shift} : ANDS ZR, Rn, Rm{,shift}. Operands [Rn, Rm].
        Mnemonic::Tst => {
            let rn = reg_num(insn, 0)?;
            let rm = reg_num(insn, 1)?;
            let (sh, imm6) = recover_shift(insn, 1, sf)?;
            (31u32, rn, rm, sh, imm6)
        }
        // Canonical: [Rd, Rn, Rm{,shift}].
        _ => {
            let rd = reg_num(insn, 0)?;
            let rn = reg_num(insn, 1)?;
            let rm = reg_num(insn, 2)?;
            let (sh, imm6) = recover_shift(insn, 2, sf)?;
            (rd, rn, rm, sh, imm6)
        }
    };

    let word = (sf << 31)
        | (opc << 29)
        | (0b01010 << 24)
        | (sh << 22)
        | (n << 21)
        | (rm << 16)
        | (imm6 << 10)
        | (rn << 5)
        | rd;
    Ok(word)
}

// ---------------------------------------------------------------------------
// Add/subtract (shifted register): ADD/ADDS/SUB/SUBS.
// ---------------------------------------------------------------------------

/// `ADD`/`ADDS`/`SUB`/`SUBS` (shifted register) and the `CMP`/`CMN`/`NEG`/`NEGS`
/// aliases.
fn enc_addsub_shifted(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();
    let sf = if is_64(code) { 1u32 } else { 0 };
    let (op, s) = match code {
        AddShifted32 | AddShifted64 => (0u32, 0u32),
        AddsShifted32 | AddsShifted64 => (0, 1),
        SubShifted32 | SubShifted64 => (1, 0),
        _ => (1, 1), // SUBS
    };

    let (rd, rn, rm, sh, imm6) = match insn.mnemonic() {
        // CMP/CMN Rn, Rm{,shift} : SUBS/ADDS ZR, Rn, Rm. Operands [Rn, Rm].
        Mnemonic::Cmp | Mnemonic::Cmn => {
            let rn = reg_num(insn, 0)?;
            let rm = reg_num(insn, 1)?;
            let (sh, imm6) = recover_shift(insn, 1, sf)?;
            (31u32, rn, rm, sh, imm6)
        }
        // NEG/NEGS Rd, Rm{,shift} : SUB/SUBS Rd, ZR, Rm. Operands [Rd, Rm].
        Mnemonic::Neg | Mnemonic::Negs => {
            let rd = reg_num(insn, 0)?;
            let rm = reg_num(insn, 1)?;
            let (sh, imm6) = recover_shift(insn, 1, sf)?;
            (rd, 31u32, rm, sh, imm6)
        }
        // Canonical: [Rd, Rn, Rm{,shift}].
        _ => {
            let rd = reg_num(insn, 0)?;
            let rn = reg_num(insn, 1)?;
            let rm = reg_num(insn, 2)?;
            let (sh, imm6) = recover_shift(insn, 2, sf)?;
            (rd, rn, rm, sh, imm6)
        }
    };

    // shift == 11 is reserved for add/sub; recover_shift never yields it for a
    // ShiftType, but Ror would map to 3 — guard against it here.
    if sh == 0b11 {
        return Err(EncodeError::InvalidOperand);
    }
    let word = (sf << 31)
        | (op << 30)
        | (s << 29)
        | (0b01011 << 24)
        | (sh << 22)
        | (rm << 16)
        | (imm6 << 10)
        | (rn << 5)
        | rd;
    Ok(word)
}

// ---------------------------------------------------------------------------
// Add/subtract (extended register): ADD/ADDS/SUB/SUBS.
// ---------------------------------------------------------------------------

/// `ADD`/`ADDS`/`SUB`/`SUBS` (extended register) and the `CMP`/`CMN` aliases,
/// inverting the `LSL`/`UXTX` option spelling.
fn enc_addsub_extended(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();
    let sf = if is_64(code) { 1u32 } else { 0 };
    let (op, s) = match code {
        AddExtended32 | AddExtended64 => (0u32, 0u32),
        AddsExtended32 | AddsExtended64 => (0, 1),
        SubExtended32 | SubExtended64 => (1, 0),
        _ => (1, 1), // SUBS
    };
    // The default option (size-matching) is UXTW(010) for 32-bit, UXTX(011) for
    // 64-bit; this is the option a folded `LSL` (extend == None) maps back to.
    let default_option = if sf == 1 { 0b011u32 } else { 0b010 };

    let (rd, rn, rm, option, imm3) = match insn.mnemonic() {
        // CMP/CMN: SUBS/ADDS with Rd==ZR. Operands [Rn, Rm].
        Mnemonic::Cmp | Mnemonic::Cmn => {
            let rn = reg_num(insn, 0)?;
            let (rm, option, imm3) = recover_extended_rm(insn, 1, default_option)?;
            (31u32, rn, rm, option, imm3)
        }
        // Canonical ADD/ADDS/SUB/SUBS: [Rd, Rn, Rm].
        _ => {
            let rd = reg_num(insn, 0)?;
            let rn = reg_num(insn, 1)?;
            let (rm, option, imm3) = recover_extended_rm(insn, 2, default_option)?;
            (rd, rn, rm, option, imm3)
        }
    };

    if imm3 > 4 {
        return Err(EncodeError::InvalidImmediate);
    }
    let word = (sf << 31)
        | (op << 30)
        | (s << 29)
        | (0b01011 << 24)
        | (1 << 21) // extended-register selector bit
        | (rm << 16)
        | (option << 13)
        | (imm3 << 10)
        | (rn << 5)
        | rd;
    Ok(word)
}

/// Recover `(rm, option, imm3)` from an extended-register `Rm` operand. A folded
/// `LSL` (no `extend` decoration) maps back to `default_option`; otherwise the
/// `extend` decoration carries the option directly. The amount is in `shift`.
#[inline]
fn recover_extended_rm(
    insn: &Instruction,
    n: usize,
    default_option: u32,
) -> Result<(u32, u32, u32), EncodeError> {
    match insn.op(n) {
        Operand::Reg {
            reg, shift, extend, ..
        } => {
            let rm = reg.number() as u32;
            let amt = shift.map(|(_, a)| a as u32).unwrap_or(0);
            let option = match extend {
                Some(ext) => ext.as_bits() as u32,
                // No extend keyword: the decoder rendered it as `LSL`, i.e. the
                // option equals the size-matching default.
                None => default_option,
            };
            Ok((rm, option, amt))
        }
        _ => Err(EncodeError::InvalidOperand),
    }
}

// ---------------------------------------------------------------------------
// Add/subtract (with carry): ADC/ADCS/SBC/SBCS.
// ---------------------------------------------------------------------------

/// `ADC`/`ADCS`/`SBC`/`SBCS` and the `NGC`/`NGCS` aliases.
fn enc_addsub_carry(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();
    let sf = if is_64(code) { 1u32 } else { 0 };
    let (op, s) = match code {
        Adc32 | Adc64 => (0u32, 0u32),
        Adcs32 | Adcs64 => (0, 1),
        Sbc32 | Sbc64 => (1, 0),
        _ => (1, 1), // SBCS
    };

    let (rd, rn, rm) = match insn.mnemonic() {
        // NGC/NGCS Rd, Rm : SBC/SBCS Rd, ZR, Rm. Operands [Rd, Rm].
        Mnemonic::Ngc | Mnemonic::Ngcs => {
            let rd = reg_num(insn, 0)?;
            let rm = reg_num(insn, 1)?;
            (rd, 31u32, rm)
        }
        _ => {
            let rd = reg_num(insn, 0)?;
            let rn = reg_num(insn, 1)?;
            let rm = reg_num(insn, 2)?;
            (rd, rn, rm)
        }
    };

    let word = (sf << 31)
        | (op << 30)
        | (s << 29)
        | (0b11010000 << 21)
        | (rm << 16)
        | (rn << 5)
        | rd;
    Ok(word)
}

/// `ADDPT`/`SUBPT` (FEAT_CPA, scalar): `<Xd|SP>, <Xn|SP>, <Xm>{, LSL #amt}`.
/// `sf=1 op(30) S(29)=0 11010 000 Rm 001 imm3 Rn Rd`; `imm3=<12:10>` is the LSL
/// amount (0..7) on `Xm`.
fn enc_addsub_pt(insn: &Instruction) -> R {
    use Code::*;
    let op = if matches!(insn.code(), Subpt) { 1u32 } else { 0 };
    let rd = reg_num(insn, 0)?;
    let rn = reg_num(insn, 1)?;
    let (rm, shift) = reg_with_shift(insn, 2)?;
    let amt = match shift {
        None => 0u32,
        Some((ShiftType::Lsl, a)) => a as u32,
        _ => return Err(EncodeError::InvalidOperand),
    };
    if amt > 7 {
        return Err(EncodeError::InvalidImmediate);
    }
    let word = (1u32 << 31)
        | (op << 30)
        | (0b11010000 << 21)
        | (rm << 16)
        | (0b001 << 13)
        | (amt << 10)
        | (rn << 5)
        | rd;
    Ok(word)
}

// ---------------------------------------------------------------------------
// Flag manipulation: RMIF / SETF8 / SETF16.
// ---------------------------------------------------------------------------

/// `RMIF <Xn>, #<shift>, #<mask>`.
fn enc_rmif(insn: &Instruction) -> R {
    let rn = reg_num(insn, 0)?;
    let imm6 = imm_u(insn, 1)?;
    let mask = imm_u(insn, 2)?;
    if imm6 > 0x3f || mask > 0xf {
        return Err(EncodeError::InvalidImmediate);
    }
    // sf=1, op=0, S=1, word<28:21>=11010000; opcode2 fixed 000001 at word<15:10>;
    // o2 (word<4>)==0.
    let word = (1u32 << 31)
        | (1 << 29) // S
        | (0b1101_0000 << 21)
        | ((imm6 as u32) << 15)
        | (0b0_0001 << 10)
        | (rn << 5)
        | (mask as u32);
    Ok(word)
}

/// `SETF8`/`SETF16 <Wn>`.
fn enc_setf(insn: &Instruction) -> R {
    let sz = if insn.code() == Code::Setf16 { 1u32 } else { 0 };
    let rn = reg_num(insn, 0)?;
    // sf=0, op=0, S=1, word<28:21>=11010000; opcode2 (word<20:15>)==0; sz at
    // word<14>; o3 (word<4>)==0; mask (word<3:0>)==1101.
    let word = (1 << 29) // S
        | (0b1101_0000 << 21)
        | (sz << 14)
        | (0b00_0010 << 10)
        | (rn << 5)
        | 0b0_1101;
    Ok(word)
}

// ---------------------------------------------------------------------------
// Conditional compare (register and immediate): CCMN/CCMP.
// ---------------------------------------------------------------------------

/// `CCMN`/`CCMP` (register and immediate). Operands: `[Rn, Rm|imm5, #nzcv, cond]`.
fn enc_cond_compare(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();
    let sf = if is_64(code) { 1u32 } else { 0 };
    let (op, imm_form) = match code {
        CcmnReg32 | CcmnReg64 => (0u32, 0u32),
        CcmnImm32 | CcmnImm64 => (0, 1),
        CcmpReg32 | CcmpReg64 => (1, 0),
        _ => (1, 1), // CCMP immediate
    };

    let rn = reg_num(insn, 0)?;
    let rm = if imm_form == 1 {
        let v = imm_u(insn, 1)?;
        if v > 0x1f {
            return Err(EncodeError::InvalidImmediate);
        }
        v as u32
    } else {
        reg_num(insn, 1)?
    };
    let nzcv = imm_u(insn, 2)?;
    if nzcv > 0xf {
        return Err(EncodeError::InvalidImmediate);
    }
    let cond = cond_of(insn, 3)?.as_u4() as u32;

    // sf op 1 11010010 Rm cond imm_form 0 Rn 0 nzcv ; S(word<29>)==1, o2/o3==0.
    let word = (sf << 31)
        | (op << 30)
        | (0b1_11010010 << 21)
        | (rm << 16)
        | (cond << 12)
        | (imm_form << 11)
        | (rn << 5)
        | (nzcv as u32);
    Ok(word)
}

// ---------------------------------------------------------------------------
// Conditional select: CSEL/CSINC/CSINV/CSNEG.
// ---------------------------------------------------------------------------

/// `CSEL`/`CSINC`/`CSINV`/`CSNEG` and the `CSET`/`CSETM`/`CINC`/`CINV`/`CNEG`
/// aliases (whose condition is the *inverse* of the encoded one).
fn enc_cond_select(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();
    let sf = if is_64(code) { 1u32 } else { 0 };
    let (op, o2) = match code {
        Csel32 | Csel64 => (0u32, 0u32),
        Csinc32 | Csinc64 => (0, 1),
        Csinv32 | Csinv64 => (1, 0),
        _ => (1, 1), // CSNEG
    };

    let (rd, rn, rm, cond) = match insn.mnemonic() {
        // CSET/CSETM Rd, cond : Rm==Rn==ZR, encoded cond = invert(cond).
        // Operands [Rd, cond].
        Mnemonic::Cset | Mnemonic::Csetm => {
            let rd = reg_num(insn, 0)?;
            let c = cond_of(insn, 1)?.invert().as_u4() as u32;
            (rd, 31u32, 31u32, c)
        }
        // CINC/CINV/CNEG Rd, Rn, cond : Rm==Rn, encoded cond = invert(cond).
        // Operands [Rd, Rn, cond].
        Mnemonic::Cinc | Mnemonic::Cinv | Mnemonic::Cneg => {
            let rd = reg_num(insn, 0)?;
            let rn = reg_num(insn, 1)?;
            let c = cond_of(insn, 2)?.invert().as_u4() as u32;
            (rd, rn, rn, c)
        }
        // Canonical CSEL/CSINC/CSINV/CSNEG: [Rd, Rn, Rm, cond].
        _ => {
            let rd = reg_num(insn, 0)?;
            let rn = reg_num(insn, 1)?;
            let rm = reg_num(insn, 2)?;
            let c = cond_of(insn, 3)?.as_u4() as u32;
            (rd, rn, rm, c)
        }
    };

    // sf op 0 11010100 Rm cond 0 o2 Rn Rd ; S(word<29>)==0, o1(word<11>)==0.
    let word = (sf << 31)
        | (op << 30)
        | (0b0_11010100 << 21)
        | (rm << 16)
        | (cond << 12)
        | (o2 << 10)
        | (rn << 5)
        | rd;
    Ok(word)
}

// ---------------------------------------------------------------------------
// Data-processing (3 source): MADD/MSUB/SMADDL/.../UMULH.
// ---------------------------------------------------------------------------

/// `MADD`/`MSUB`/`SMADDL`/`SMSUBL`/`SMULH`/`UMADDL`/`UMSUBL`/`UMULH` and the
/// `MUL`/`MNEG`/`SMULL`/`SMNEGL`/`UMULL`/`UMNEGL` aliases.
fn enc_dp_3source(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();
    // Only the plain MADD/MSUB have a 32-bit variant; every widening / high-mul
    // form (SMADDL/.../UMULH) is encoded with sf == 1.
    let sf = match code {
        Madd32 | Msub32 => 0u32,
        _ => 1,
    };
    let (op31, o0) = match code {
        Madd32 | Madd64 => (0b000u32, 0u32),
        Msub32 | Msub64 => (0b000, 1),
        Smaddl => (0b001, 0),
        Smsubl => (0b001, 1),
        Smulh => (0b010, 0),
        Maddpt => (0b011, 0),
        Msubpt => (0b011, 1),
        Umaddl => (0b101, 0),
        Umsubl => (0b101, 1),
        _ => (0b110, 0), // UMULH
    };

    let (rd, rn, rm, ra) = match code {
        // SMULH/UMULH: [Rd, Rn, Rm], Ra fixed ZR.
        Smulh | Umulh => {
            let rd = reg_num(insn, 0)?;
            let rn = reg_num(insn, 1)?;
            let rm = reg_num(insn, 2)?;
            (rd, rn, rm, 31u32)
        }
        // The widen + plain forms share the alias (Ra==ZR -> MUL/MNEG/...).
        _ => match insn.mnemonic() {
            Mnemonic::Mul
            | Mnemonic::Mneg
            | Mnemonic::Smull
            | Mnemonic::Smnegl
            | Mnemonic::Umull
            | Mnemonic::Umnegl => {
                let rd = reg_num(insn, 0)?;
                let rn = reg_num(insn, 1)?;
                let rm = reg_num(insn, 2)?;
                (rd, rn, rm, 31u32)
            }
            // Canonical four-register form: [Rd, Rn, Rm, Ra].
            _ => {
                let rd = reg_num(insn, 0)?;
                let rn = reg_num(insn, 1)?;
                let rm = reg_num(insn, 2)?;
                let ra = reg_num(insn, 3)?;
                (rd, rn, rm, ra)
            }
        },
    };

    // sf 00 11011 op31 Rm o0 Ra Rn Rd ; op54(word<30:29>)==00.
    let word = (sf << 31)
        | (0b11011 << 24)
        | (op31 << 21)
        | (rm << 16)
        | (o0 << 15)
        | (ra << 10)
        | (rn << 5)
        | rd;
    Ok(word)
}

// ---------------------------------------------------------------------------
// Data-processing (2 source): UDIV/SDIV/LSLV.../CRC32*/SUBP/IRG/GMI/PACGA.
// ---------------------------------------------------------------------------

/// `UDIV`/`SDIV`/`LSLV`/`LSRV`/`ASRV`/`RORV` (with `LSL`/`LSR`/`ASR`/`ROR`
/// aliases) plus `CRC32*`, `SUBP`/`SUBPS`, `IRG`/`GMI` and `PACGA`.
fn enc_dp_2source(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();

    // S is 1 only for SUBPS; everything else sets S = 0.
    let (sf, s, opcode): (u32, u32, u32) = match code {
        Udiv32 => (0, 0, 0b000010),
        Udiv64 => (1, 0, 0b000010),
        Sdiv32 => (0, 0, 0b000011),
        Sdiv64 => (1, 0, 0b000011),
        Lslv32 => (0, 0, 0b001000),
        Lslv64 => (1, 0, 0b001000),
        Lsrv32 => (0, 0, 0b001001),
        Lsrv64 => (1, 0, 0b001001),
        Asrv32 => (0, 0, 0b001010),
        Asrv64 => (1, 0, 0b001010),
        Rorv32 => (0, 0, 0b001011),
        Rorv64 => (1, 0, 0b001011),
        SubpDp => (1, 0, 0b000000),
        SubpsDp => (1, 1, 0b000000),
        IrgDp => (1, 0, 0b000100),
        GmiDp => (1, 0, 0b000101),
        Pacga => (1, 0, 0b001100),
        Crc32b => (0, 0, 0b010000),
        Crc32h => (0, 0, 0b010001),
        Crc32w => (0, 0, 0b010010),
        Crc32x => (1, 0, 0b010011),
        Crc32cb => (0, 0, 0b010100),
        Crc32ch => (0, 0, 0b010101),
        Crc32cw => (0, 0, 0b010110),
        Crc32cx => (1, 0, 0b010111),
        SmaxReg32 => (0, 0, 0b011000),
        SmaxReg64 => (1, 0, 0b011000),
        UmaxReg32 => (0, 0, 0b011001),
        UmaxReg64 => (1, 0, 0b011001),
        SminReg32 => (0, 0, 0b011010),
        SminReg64 => (1, 0, 0b011010),
        UminReg32 => (0, 0, 0b011011),
        _ => (1, 0, 0b011011), // UminReg64
    };

    // Operand recovery varies by form; IRG may drop its third operand.
    let (rd, rn, rm) = match code {
        // IRG <Xd|SP>, <Xn|SP>{, <Xm>}: the optional Xm is dropped when ZR.
        IrgDp => {
            let rd = reg_num(insn, 0)?;
            let rn = reg_num(insn, 1)?;
            let rm = if insn.op_count() >= 3 {
                reg_num(insn, 2)?
            } else {
                31u32
            };
            (rd, rn, rm)
        }
        _ => {
            let rd = reg_num(insn, 0)?;
            let rn = reg_num(insn, 1)?;
            let rm = reg_num(insn, 2)?;
            (rd, rn, rm)
        }
    };

    // sf S 0 11010110 Rm opcode Rn Rd ; op(word<30>)==0.
    let word = (sf << 31)
        | (s << 29)
        | (0b0_11010110 << 21)
        | (rm << 16)
        | (opcode << 10)
        | (rn << 5)
        | rd;
    Ok(word)
}

// ---------------------------------------------------------------------------
// Data-processing (1 source): RBIT/REV16/REV/REV32/CLZ/CLS.
// ---------------------------------------------------------------------------

/// `RBIT`/`REV16`/`REV`/`REV32`/`CLZ`/`CLS` (the non-PAuth 1-source forms).
fn enc_dp_1source_basic(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();
    let sf = if is_64(code) { 1u32 } else { 0 };
    // opcode (word<15:10>) and opcode2 (word<20:16>) == 00000.
    let opcode = match code {
        Rbit32 | Rbit64 => 0b000000u32,
        Rev1632 | Rev1664 => 0b000001,
        // 32-bit REV uses opcode 000010; 64-bit REV32 uses opcode 000010.
        Rev32Bit | Rev3232 | Rev3264 => 0b000010,
        Rev64Bit => 0b000011,
        Clz32 | Clz64 => 0b000100,
        Cls32 | Cls64 => 0b000101,
        // FEAT_CSSC.
        Ctz32 | Ctz64 => 0b000110,
        Cnt32 | Cnt64 => 0b000111,
        _ => 0b001000, // ABS (FEAT_CSSC)
    };

    let rd = reg_num(insn, 0)?;
    let rn = reg_num(insn, 1)?;

    // sf 1 0 11010110 00000 opcode Rn Rd ; opcode2(word<20:16>)==00000, S==0.
    // Fixed bits above opcode: bit30=1, bit29=0(S), word<28:21>=11010110.
    let word =
        (sf << 31) | (1 << 30) | (0b11010110 << 21) | (opcode << 10) | (rn << 5) | rd;
    Ok(word)
}

/// The pointer-authentication 1-source forms (`PAC*`/`AUT*`/`XPAC*`). The `Z`
/// forms carry only the destination (encoded with `Rn == 11111`).
fn enc_dp_1source_pauth(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();

    // FEAT_PAuth_LR forms: implicit LR destination (`Rd == 11110`); `Rn` is the
    // SP (`11111`) for the no-operand `PAC*SPPC` forms, or the modifier `Xm` for
    // `AUTI*SPPCR <Xm>`. `opcode2 == 00001` (set below), opcode<5> == 1.
    if let Some((opcode, rn)) = match code {
        Paciasppc => Some((0b101000u32, 31u32)),
        Pacibsppc => Some((0b101001, 31)),
        Pacnbiasppc => Some((0b100000, 31)),
        Pacnbibsppc => Some((0b100001, 31)),
        Autiasppcr => Some((0b100100, reg_num(insn, 0)?)),
        Autibsppcr => Some((0b100101, reg_num(insn, 0)?)),
        _ => None,
    } {
        let word = (1u32 << 31)
            | (1 << 30)
            | (0b11010110 << 21)
            | (1 << 16)
            | (opcode << 10)
            | (rn << 5)
            | 0b11110;
        return Ok(word);
    }

    let (opcode, z) = match code {
        PaciaDp => (0b000000u32, false),
        PacibDp => (0b000001, false),
        PacdaDp => (0b000010, false),
        PacdbDp => (0b000011, false),
        AutiaDp => (0b000100, false),
        AutibDp => (0b000101, false),
        AutdaDp => (0b000110, false),
        AutdbDp => (0b000111, false),
        PacizaDp => (0b001000, true),
        PacizbDp => (0b001001, true),
        PacdzaDp => (0b001010, true),
        PacdzbDp => (0b001011, true),
        AutizaDp => (0b001100, true),
        AutizbDp => (0b001101, true),
        AutdzaDp => (0b001110, true),
        AutdzbDp => (0b001111, true),
        XpaciDp => (0b010000, true),
        _ => (0b010001, true), // XPACD
    };

    let rd = reg_num(insn, 0)?;
    let rn = if z {
        // Z / XPAC forms have an implicit Rn == ZR (no source operand).
        31u32
    } else {
        reg_num(insn, 1)?
    };

    // sf=1 1 0 11010110 00001 opcode Rn Rd ; opcode2(word<20:16>)==00001, S==0.
    // Fixed bits: bit31=1(sf), bit30=1, bit29=0(S), word<28:21>=11010110, bit16=1.
    let word =
        (1u32 << 31) | (1 << 30) | (0b11010110 << 21) | (1 << 16) | (opcode << 10) | (rn << 5) | rd;
    Ok(word)
}

#[cfg(test)]
mod tests {
    use crate::features::FeatureSet;
    use crate::instruction::Instruction;

    /// Decode a word then re-encode it and require the exact same word back.
    #[track_caller]
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
    fn dp_reg_logical_shifted() {
        rt(0x8AC7E6A3); // and x3, x21, x7, ror #0x39
        rt(0x2A1503E7); // mov w7, w21 (orr)
        rt(0xAA731FFA); // mvn x26, x19, lsr #0x7
        rt(0xEA863DFF); // tst x15, x6, asr #0xf
        rt(0xCA08037C); // eor x28, x27, x8 (lsl #0 elided)
    }

    #[test]
    fn dp_reg_addsub_shifted() {
        rt(0x8B4BB429); // add x9, x1, x11, lsr #0x2d
        rt(0xEB8030BF); // cmp x5, x0, asr #0xc
        rt(0xAB88807F); // cmn x3, x8, asr #0x20
        rt(0xCB9017FA); // neg x26, x16, asr #0x5
        rt(0xEB102BFE); // negs x30, x16, lsl #0xa
    }

    #[test]
    fn dp_reg_addsub_extended() {
        rt(0x8B3B6186); // add x6, x12, x27, uxtx
        rt(0x8B304F9F); // add sp, x28, w16, uxtw #0x3
        rt(0xEB2A61FF); // cmp x15, x10, uxtx
        rt(0x8B206FFF); // add sp, sp, x0, lsl #0x3
        rt(0x8B2063FF); // add sp, sp, x0 (lsl #0 elided)
        rt(0x0B33604F); // add w15, w2, w19, uxtx
        rt(0x4B2DEB78); // sub w24, w27, w13, sxtx #0x2
    }

    #[test]
    fn dp_reg_addsub_carry() {
        rt(0x9A0E0358); // adc x24, x26, x14
        rt(0xDA1B03E2); // ngc x2, x27
        rt(0xFA0703F1); // ngcs x17, x7
    }

    #[test]
    fn dp_reg_flag_manip() {
        rt(0xBA1785C9); // rmif x14, #0x2f, #0x9
        rt(0xBA000589); // rmif x12, #0x0, #0x9
        rt(0x3A000A4D); // setf8 w18
        rt(0x3A00494D); // setf16 w10
    }

    #[test]
    fn dp_reg_cond_compare() {
        rt(0xBA45B044); // ccmn x2, x5, #0x4, lt
        rt(0xBA537B2F); // ccmn x25, #0x13, #0xf, vc
        rt(0xFA439341); // ccmp x26, x3, #0x1, ls
        rt(0x7A4B5A46); // ccmp w18, #0xb, #0x6, pl
    }

    #[test]
    fn dp_reg_cond_select() {
        rt(0x9A95C0C0); // csel x0, x6, x21, gt
        rt(0x9A9F97E4); // cset x4, hi
        rt(0x5A9F53EE); // csetm w14, mi
        rt(0x9A8BC574); // cinc x20, x11, le
        rt(0xDA9B5367); // cinv x7, x27, mi
        rt(0xDA9E97D1); // cneg x17, x30, hi
    }

    #[test]
    fn dp_reg_3source() {
        rt(0x9B0D31FC); // madd x28, x15, x13, x12
        rt(0x9B057C24); // mul x4, x1, x5
        rt(0x1B0CFC7C); // mneg w28, w3, w12
        rt(0x9B2B0AF3); // smaddl x19, w23, w11, x2
        rt(0x9B337F45); // smull x5, w26, w19
        rt(0x9B2CFC4D); // smnegl x13, w2, w12
        rt(0x9B477C3B); // smulh x27, x1, x7 (Ra=ZR canonical)
        rt(0x9BB07F03); // umull x3, w24, w16
        rt(0x9BDD7D42); // umulh x2, x10, x29 (Ra=ZR canonical)
    }

    #[test]
    fn dp_reg_2source() {
        rt(0x9ADF09D9); // udiv x25, x14, xzr
        rt(0x9ACB2231); // lsl x17, x17, x11
        rt(0x1AD22518); // lsr w24, w8, w18
        rt(0x9ADB2B14); // asr x20, x24, x27
        rt(0x1AD82F76); // ror w22, w27, w24
        rt(0x1AC74347); // crc32b w7, w26, w7
        rt(0x9AC24E71); // crc32x w17, w19, x2
        rt(0x9ADB004D); // subp x13, x2, x27
        rt(0xBAD601FA); // subps x26, x15, x22
        rt(0xBAC202FF); // subps xzr, x23, x2
        rt(0x9AC112F1); // irg x17, x23, x1
        rt(0x9ADF130C); // irg x12, x24 (Xm dropped)
        rt(0x9ADB14C8); // gmi x8, x6, x27
        rt(0x9AD03137); // pacga x23, x9, x16
    }

    #[test]
    fn dp_reg_cssc() {
        // 1-source: ABS / CNT / CTZ.
        rt(0xDAC02020); // abs x0, x1
        rt(0x5AC02083); // abs w3, w4
        rt(0xDAC01CC5); // cnt x5, x6
        rt(0x5AC01D07); // cnt w7, w8
        rt(0xDAC01949); // ctz x9, x10
        rt(0x5AC0198B); // ctz w11, w12
        // 2-source register min/max.
        rt(0x9AC26020); // smax x0, x1, x2
        rt(0x1ACF61CD); // smax w13, w14, w15
        rt(0x9AD26A30); // smin x16, x17, x18
        rt(0x9AD56693); // umax x19, x20, x21
        rt(0x9AD86EF6); // umin x22, x23, x24
    }

    #[test]
    fn dp_reg_1source() {
        rt(0xDAC1030C); // pacia x12, x24
        rt(0xDAC1128D); // autia x13, x20
        rt(0xDAC123ED); // paciza x13
        rt(0xDAC133F8); // autiza x24
        rt(0xDAC143F2); // xpaci x18
    }
}
