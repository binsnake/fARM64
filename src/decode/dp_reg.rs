//! Data Processing -- Register (ARM ARM C4.1.94).
//!
//! Hand-written decoder dispatched here from [`crate::decode::decode_into`] when
//! `op0 = word<28:25>` selects `x101`. The group is sub-classed on
//! `word<28:21>` (and `word<30>` for the 1-source vs 2-source split):
//!
//! | bits | class |
//! |-|-|
//! | `xxx01010` | Logical (shifted register) |
//! | `xxx01011`, `word<21>==0` | Add/subtract (shifted register) |
//! | `xxx01011`, `word<21>==1` | Add/subtract (extended register) |
//! | `xxx11010`, `word<23:21>==000` | Add/subtract (with carry) |
//! | `xxx11010`, `word<23:21>==010` | Conditional compare (reg / imm) |
//! | `xxx11010`, `word<23:21>==100` | Conditional select |
//! | `xxx11010`, `word<23:21>==110`, `word<30>==0` | Data-processing (2 source) |
//! | `xxx11010`, `word<23:21>==110`, `word<30>==1` | Data-processing (1 source) |
//! | `xxx11011` | Data-processing (3 source) |
//!
//! Preferred-disassembly aliases (ARM ARM alias conditions) are emitted on the
//! `mnemonic` while `code` stays the canonical encoding identity, matching the
//! Binary Ninja differential corpus: `MOV`/`MVN`/`TST` (logical), `CMP`/`CMN`/
//! `NEG`/`NEGS` (add/sub), `NGC`/`NGCS` (carry), `CSET`/`CSETM`/`CINC`/`CINV`/
//! `CNEG` (conditional select), `MUL`/`MNEG`/`SMULL`/`SMNEGL`/`UMULL`/`UMNEGL`
//! (3-source), and `LSL`/`LSR`/`ASR`/`ROR` (2-source variable shifts).

use crate::decode::bits::{bit, bits};
use crate::enums::{Condition, ExtendType, ShiftType};
use crate::features::{Feature, FeatureSet};
use crate::instruction::Instruction;
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::Operand;
use crate::register::{gp_register, RegWidth};

/// Build a plain (undecorated) GP register operand.
#[inline]
fn reg(use_sp: bool, width: RegWidth, n: u32) -> Operand {
    Operand::Reg {
        reg: gp_register(use_sp, width, (n & 0x1f) as u8),
        arr: None,
        lane: None,
        shift: None,
        extend: None,
        pred: None,
    }
}

/// Build a register operand carrying a shift decoration. A `LSL #0` is folded
/// away (the operand renders as a bare register) to match UAL/Binary Ninja,
/// which elide `, lsl #0` but keep `, ror #0` / `, lsr #0` / `, asr #0`.
#[inline]
fn reg_shifted(width: RegWidth, n: u32, st: ShiftType, amt: u32) -> Operand {
    let shift = if st == ShiftType::Lsl && amt == 0 {
        None
    } else {
        Some((st, amt as u8))
    };
    Operand::Reg {
        reg: gp_register(false, width, (n & 0x1f) as u8),
        arr: None,
        lane: None,
        shift,
        extend: None,
        pred: None,
    }
}

/// The GP register width for an `sf` bit (`0` => `W`/32-bit, `1` => `X`/64-bit).
#[inline]
fn width_of(sf: u32) -> RegWidth {
    if sf & 1 == 1 {
        RegWidth::X64
    } else {
        RegWidth::W32
    }
}

/// Decode a Data Processing -- Register instruction into `out`.
///
/// `ip` is unused by this group (no PC-relative forms) but is accepted for a
/// uniform group-decoder signature. Pure, total and panic-free for every input;
/// unallocated encodings leave `out` as the invalid default.
#[inline]
pub fn decode(word: u32, ip: u64, features: FeatureSet, out: &mut Instruction) {
    let _ = ip;
    let op28_24 = bits(word, 24, 5);
    match op28_24 {
        // Logical (shifted register).
        0b01010 => decode_logical_shifted(word, out),
        // Add/subtract (shifted or extended register).
        0b01011 => {
            if bit(word, 21) == 0 {
                decode_addsub_shifted(word, out);
            } else {
                decode_addsub_extended(word, out);
            }
        }
        // Add/sub-carry, conditional compare/select, 1-source, 2-source.
        0b11010 => {
            let op2 = bits(word, 21, 3); // word<23:21>
            match op2 {
                0b000 => {
                    // The op2==000 slot holds several sub-families, split on
                    // word<15:10> (binja's `op3`): `000000` is add/subtract
                    // (with carry); `xxxxx1` (word<14:10>==00001) is RMIF;
                    // `xxxx10` (word<13:10>==0010) is SETF8/SETF16; and
                    // `001xxx` (word<15:13>==001) is the FEAT_CPA add/subtract
                    // checked-pointer ADDPT/SUBPT (with an LSL #imm3 on `Xm`).
                    let op3 = bits(word, 10, 6);
                    if op3 == 0 {
                        decode_addsub_carry(word, out);
                    } else if (op3 & 0x1f) == 1 {
                        decode_rmif(word, out);
                    } else if (op3 & 0xf) == 2 {
                        decode_setf(word, out);
                    } else if bits(word, 13, 3) == 0b001 {
                        decode_addsub_pt(word, features, out);
                    }
                    // Any other op3 in this slot is UNALLOCATED.
                }
                0b010 => decode_cond_compare(word, out),
                0b100 => decode_cond_select(word, out),
                0b110 => {
                    if bit(word, 30) == 1 {
                        decode_dp_1source(word, features, out);
                    } else {
                        decode_dp_2source(word, features, out);
                    }
                }
                // 001/011/101/111: UNALLOCATED in this group.
                _ => {}
            }
        }
        // Data-processing (3 source).
        0b11011 => decode_dp_3source(word, out),
        // Any other op28_24 within this group is UNALLOCATED.
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Logical (shifted register): AND/BIC/ORR/ORN/EOR/EON/ANDS/BICS.
// ---------------------------------------------------------------------------

/// `AND`/`BIC`/`ORR`/`ORN`/`EOR`/`EON`/`ANDS`/`BICS` (shifted register). `opc`
/// (`word<30:29>`) selects the logical op, `N` (`word<21>`) the bit-inverted
/// variant. For 32-bit forms `imm6<5>` must be `0` (else UNALLOCATED). Emits the
/// `MOV`/`MVN`/`TST` preferred aliases.
#[inline]
fn decode_logical_shifted(word: u32, out: &mut Instruction) {
    let sf = bit(word, 31);
    let opc = bits(word, 29, 2);
    let shift = bits(word, 22, 2);
    let n = bit(word, 21);
    let rm = bits(word, 16, 5);
    let imm6 = bits(word, 10, 6);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    // 32-bit forms: imm6<5> must be 0.
    if sf == 0 && bit(imm6, 5) == 1 {
        return;
    }
    let w = width_of(sf);
    let st = ShiftType::from_bits(shift as u8);

    let code = match (sf, opc, n) {
        (0, 0b00, 0) => Code::AndShifted32,
        (1, 0b00, 0) => Code::AndShifted64,
        (0, 0b00, 1) => Code::BicShifted32,
        (1, 0b00, 1) => Code::BicShifted64,
        (0, 0b01, 0) => Code::OrrShifted32,
        (1, 0b01, 0) => Code::OrrShifted64,
        (0, 0b01, 1) => Code::OrnShifted32,
        (1, 0b01, 1) => Code::OrnShifted64,
        (0, 0b10, 0) => Code::EorShifted32,
        (1, 0b10, 0) => Code::EorShifted64,
        (0, 0b10, 1) => Code::EonShifted32,
        (1, 0b10, 1) => Code::EonShifted64,
        (0, _, 0) => Code::AndsShifted32,
        (1, _, 0) => Code::AndsShifted64,
        (0, _, _) => Code::BicsShifted32,
        (_, _, _) => Code::BicsShifted64,
    };
    out.set(code);

    let is_orr = opc == 0b01 && n == 0;
    let is_orn = opc == 0b01 && n == 1;
    let is_ands = opc == 0b11 && n == 0;

    // ORR Rn==ZR with LSL #0 -> MOV Rd, Rm.
    if is_orr && rn == 31 && st == ShiftType::Lsl && imm6 == 0 {
        out.set_mnemonic(Mnemonic::Mov);
        out.push_operand(reg(false, w, rd));
        out.push_operand(reg(false, w, rm));
        return;
    }

    // ORN Rn==ZR -> MVN Rd, Rm{, shift #amt}.
    if is_orn && rn == 31 {
        out.set_mnemonic(Mnemonic::Mvn);
        out.push_operand(reg(false, w, rd));
        out.push_operand(reg_shifted(w, rm, st, imm6));
        return;
    }

    // ANDS Rd==ZR -> TST Rn, Rm{, shift #amt}.
    if is_ands && rd == 31 {
        out.set_mnemonic(Mnemonic::Tst);
        out.push_operand(reg(false, w, rn));
        out.push_operand(reg_shifted(w, rm, st, imm6));
        return;
    }

    out.push_operand(reg(false, w, rd));
    out.push_operand(reg(false, w, rn));
    out.push_operand(reg_shifted(w, rm, st, imm6));
}

// ---------------------------------------------------------------------------
// Add/subtract (shifted register): ADD/ADDS/SUB/SUBS.
// ---------------------------------------------------------------------------

/// `ADD`/`ADDS`/`SUB`/`SUBS` (shifted register). `shift` (`word<23:22>`) selects
/// LSL/LSR/ASR (`11` is reserved); for 32-bit forms `imm6<5>` must be `0`. Emits
/// the `CMP`/`CMN`/`NEG`/`NEGS` preferred aliases.
#[inline]
fn decode_addsub_shifted(word: u32, out: &mut Instruction) {
    let sf = bit(word, 31);
    let op = bit(word, 30); // 0 = ADD, 1 = SUB
    let s = bit(word, 29); // flag-setting
    let shift = bits(word, 22, 2);
    let rm = bits(word, 16, 5);
    let imm6 = bits(word, 10, 6);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    // shift == 11 is reserved for add/sub.
    if shift == 0b11 {
        return;
    }
    // 32-bit forms: imm6<5> must be 0.
    if sf == 0 && bit(imm6, 5) == 1 {
        return;
    }
    let w = width_of(sf);
    let st = ShiftType::from_bits(shift as u8);

    let code = match (sf, op, s) {
        (0, 0, 0) => Code::AddShifted32,
        (1, 0, 0) => Code::AddShifted64,
        (0, 0, 1) => Code::AddsShifted32,
        (1, 0, 1) => Code::AddsShifted64,
        (0, 1, 0) => Code::SubShifted32,
        (1, 1, 0) => Code::SubShifted64,
        (0, 1, 1) => Code::SubsShifted32,
        _ => Code::SubsShifted64,
    };
    out.set(code);

    let flag_setting = s == 1;

    // SUBS/ADDS Rd==ZR -> CMP/CMN Rn, Rm{, shift #amt}.
    if flag_setting && rd == 31 {
        out.set_mnemonic(if op == 1 { Mnemonic::Cmp } else { Mnemonic::Cmn });
        out.push_operand(reg(false, w, rn));
        out.push_operand(reg_shifted(w, rm, st, imm6));
        return;
    }

    // SUB/SUBS Rn==ZR -> NEG/NEGS Rd, Rm{, shift #amt}.
    if op == 1 && rn == 31 {
        out.set_mnemonic(if flag_setting { Mnemonic::Negs } else { Mnemonic::Neg });
        out.push_operand(reg(false, w, rd));
        out.push_operand(reg_shifted(w, rm, st, imm6));
        return;
    }

    out.push_operand(reg(false, w, rd));
    out.push_operand(reg(false, w, rn));
    out.push_operand(reg_shifted(w, rm, st, imm6));
}

// ---------------------------------------------------------------------------
// Add/subtract (extended register): ADD/ADDS/SUB/SUBS.
// ---------------------------------------------------------------------------

/// `ADD`/`ADDS`/`SUB`/`SUBS` (extended register). `option` (`word<15:13>`) is the
/// extend, `imm3` (`word<12:10>`) the left-shift amount (`> 4` is UNALLOCATED).
/// For non-flag forms Rd/Rn are SP-capable; flag forms make Rn SP-capable and Rd
/// is ZR (with the `CMP`/`CMN` aliases). The `LSL` special spelling is applied
/// when the extend equals the operand-size default and SP is involved.
#[inline]
fn decode_addsub_extended(word: u32, out: &mut Instruction) {
    let sf = bit(word, 31);
    let op = bit(word, 30);
    let s = bit(word, 29);
    let option = bits(word, 13, 3);
    let imm3 = bits(word, 10, 3);
    let rm = bits(word, 16, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    // imm3 > 4 is UNALLOCATED.
    if imm3 > 4 {
        return;
    }
    let w = width_of(sf);
    let ext = ExtendType::from_bits(option as u8);

    let code = match (sf, op, s) {
        (0, 0, 0) => Code::AddExtended32,
        (1, 0, 0) => Code::AddExtended64,
        (0, 0, 1) => Code::AddsExtended32,
        (1, 0, 1) => Code::AddsExtended64,
        (0, 1, 0) => Code::SubExtended32,
        (1, 1, 0) => Code::SubExtended64,
        (0, 1, 1) => Code::SubsExtended32,
        _ => Code::SubsExtended64,
    };
    out.set(code);

    let flag_setting = s == 1;
    // CMP/CMN: SUBS/ADDS with Rd==ZR.
    let is_cmp = flag_setting && rd == 31;

    // The extended-register Rm width is X only for the doubleword extends
    // (UXTX/SXTX, `option<1:0>==11`) of a 64-bit operation; every 32-bit form
    // (and the byte/half/word extends) use the W view of Rm.
    let rm_w = if sf == 1 && (option & 0b11) == 0b11 {
        RegWidth::X64
    } else {
        RegWidth::W32
    };

    // LSL special: when the extend is the size-matching default (UXTX for 64-bit,
    // UXTW for 32-bit) AND SP is involved (Rd or Rn is reg-31, role-dependent),
    // UAL spells it `lsl`. Determine SP involvement per the alias' operand roles:
    // for CMP/CMN only Rn matters (Rd is ZR); otherwise Rd and Rn are SP-capable.
    let default_option = if sf == 1 { 0b011 } else { 0b010 };
    let sp_involved = if is_cmp {
        rn == 31
    } else {
        rd == 31 || rn == 31
    };
    let use_lsl = option == default_option && sp_involved;

    let rm_op = build_extended_rm(rm, rm_w, ext, imm3, use_lsl);

    if is_cmp {
        out.set_mnemonic(if op == 1 { Mnemonic::Cmp } else { Mnemonic::Cmn });
        // Rn is SP-capable.
        out.push_operand(reg(true, w, rn));
        out.push_operand(rm_op);
        return;
    }

    if flag_setting {
        // ADDS/SUBS: Rd is ZR, Rn SP-capable.
        out.push_operand(reg(false, w, rd));
        out.push_operand(reg(true, w, rn));
        out.push_operand(rm_op);
    } else {
        // ADD/SUB: Rd and Rn both SP-capable.
        out.push_operand(reg(true, w, rd));
        out.push_operand(reg(true, w, rn));
        out.push_operand(rm_op);
    }
}

/// Build the extended-register `Rm` operand, applying the `LSL` special spelling.
///
/// When `use_lsl` is set the extend is rendered as `LSL` (a shift), with `LSL #0`
/// elided entirely. Otherwise the extend keyword is shown and the `#amt` is shown
/// only when non-zero (handled by the formatter's `emit_extend`).
#[inline]
fn build_extended_rm(
    rm: u32,
    rm_w: RegWidth,
    ext: ExtendType,
    imm3: u32,
    use_lsl: bool,
) -> Operand {
    let reg = gp_register(false, rm_w, (rm & 0x1f) as u8);
    if use_lsl {
        // Render as a plain LSL shift; LSL #0 folds to a bare register.
        let shift = if imm3 == 0 {
            None
        } else {
            Some((ShiftType::Lsl, imm3 as u8))
        };
        Operand::Reg {
            reg,
            arr: None,
            lane: None,
            shift,
            extend: None,
            pred: None,
        }
    } else {
        // Extend keyword; carry the amount via `shift` so the formatter's
        // `emit_extend` prints `#amt` when non-zero (and elides it at zero).
        Operand::Reg {
            reg,
            arr: None,
            lane: None,
            shift: Some((ShiftType::Lsl, imm3 as u8)),
            extend: Some(ext),
            pred: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Add/subtract (with carry): ADC/ADCS/SBC/SBCS.
// ---------------------------------------------------------------------------

/// `ADC`/`ADCS`/`SBC`/`SBCS`. Fixed bits `word<15:10> == 000000` (else
/// UNALLOCATED). Emits the `NGC`/`NGCS` preferred aliases (SBC/SBCS with Rn==ZR).
#[inline]
fn decode_addsub_carry(word: u32, out: &mut Instruction) {
    let sf = bit(word, 31);
    let op = bit(word, 30); // 0 = ADC, 1 = SBC
    let s = bit(word, 29);
    let opcode2 = bits(word, 10, 6);
    let rm = bits(word, 16, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    // The fixed opcode2 field must be zero.
    if opcode2 != 0 {
        return;
    }
    let w = width_of(sf);

    let code = match (sf, op, s) {
        (0, 0, 0) => Code::Adc32,
        (1, 0, 0) => Code::Adc64,
        (0, 0, 1) => Code::Adcs32,
        (1, 0, 1) => Code::Adcs64,
        (0, 1, 0) => Code::Sbc32,
        (1, 1, 0) => Code::Sbc64,
        (0, 1, 1) => Code::Sbcs32,
        _ => Code::Sbcs64,
    };
    out.set(code);

    // SBC/SBCS Rn==ZR -> NGC/NGCS Rd, Rm.
    if op == 1 && rn == 31 {
        out.set_mnemonic(if s == 1 { Mnemonic::Ngcs } else { Mnemonic::Ngc });
        out.push_operand(reg(false, w, rd));
        out.push_operand(reg(false, w, rm));
        return;
    }

    out.push_operand(reg(false, w, rd));
    out.push_operand(reg(false, w, rn));
    out.push_operand(reg(false, w, rm));
}

/// `ADDPT`/`SUBPT` (FEAT_CPA, scalar): `<Xd|SP>, <Xn|SP>, <Xm>{, LSL #amt}`.
/// Encoding: `sf=1 op(30) S(29)=0 11010 000 Rm 001 imm3 Rn Rd`. `op` selects
/// ADD(0)/SUB(1); `imm3=word<12:10>` is the left-shift applied to `Xm` (rendered
/// `, lsl #amt`, eliding `#0`). 32-bit (`sf==0`) and `S==1` are UNALLOCATED.
#[inline]
fn decode_addsub_pt(word: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Cpa) {
        return;
    }
    let sf = bit(word, 31);
    let op = bit(word, 30);
    let s = bit(word, 29);
    if sf == 0 || s == 1 {
        return;
    }
    let rm = bits(word, 16, 5);
    let amt = bits(word, 10, 3);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);
    out.set(if op == 0 { Code::Addpt } else { Code::Subpt });
    out.push_operand(reg(true, RegWidth::X64, rd));
    out.push_operand(reg(true, RegWidth::X64, rn));
    out.push_operand(reg_shifted(RegWidth::X64, rm, ShiftType::Lsl, amt));
}

// ---------------------------------------------------------------------------
// Flag manipulation (FEAT_FlagM / FEAT_FlagM2): RMIF / SETF8 / SETF16.
// ---------------------------------------------------------------------------

/// `RMIF <Xn>, #<shift>, #<mask>` — rotate `Xn` right by `shift` and insert the
/// masked low nibble into NZCV. Encoding: `sf=1, op=0, S=1`, `imm6=word<20:15>`
/// is the rotate amount, `Rn=word<9:5>`, `mask=word<3:0>`. The fixed bits
/// (`sf==1`, `op==0`, `S==1`, `o2(word<4>)==0`) must hold or the word is
/// UNALLOCATED.
#[inline]
fn decode_rmif(word: u32, out: &mut Instruction) {
    // sf=word<31>==1, op=word<30>==0, S=word<29>==1, o2=word<4>==0.
    if bit(word, 31) != 1 || bit(word, 30) != 0 || bit(word, 29) != 1 || bit(word, 4) != 0 {
        return;
    }
    let imm6 = bits(word, 15, 6);
    let rn = bits(word, 5, 5);
    let mask = bits(word, 0, 4);
    out.set(Code::Rmif);
    out.push_operand(reg(false, RegWidth::X64, rn));
    out.push_operand(Operand::ImmUnsigned(imm6 as u64));
    out.push_operand(Operand::ImmUnsigned(mask as u64));
}

/// `SETF8`/`SETF16 <Wn>` — set NZV flags from the low 8/16 bits of `Wn`.
/// Encoding: `sf=0, op=0, S=1`, `opcode2=word<20:15>==000000`, `sz=word<14>`
/// (0→SETF8, 1→SETF16), `Rn=word<9:5>`, `o3=word<4>==0`, `mask=word<3:0>==1101`.
/// The fixed bits must hold or the word is UNALLOCATED.
#[inline]
fn decode_setf(word: u32, out: &mut Instruction) {
    // sf=word<31>==0, op=word<30>==0, S=word<29>==1; opcode2 (word<20:15>)==0;
    // o3 (word<4>)==0; mask (word<3:0>)==1101.
    if bit(word, 31) != 0 || bit(word, 30) != 0 || bit(word, 29) != 1 {
        return;
    }
    if bits(word, 15, 6) != 0 || bit(word, 4) != 0 || bits(word, 0, 4) != 0b1101 {
        return;
    }
    let sz = bit(word, 14);
    let rn = bits(word, 5, 5);
    out.set(if sz == 0 { Code::Setf8 } else { Code::Setf16 });
    out.push_operand(reg(false, RegWidth::W32, rn));
}

// ---------------------------------------------------------------------------
// Conditional compare (register and immediate): CCMN/CCMP.
// ---------------------------------------------------------------------------

/// `CCMN`/`CCMP` (register and immediate). `op` (`word<30>`) selects CCMP; bit 11
/// selects the immediate form (5-bit `imm5` in place of `Rm`). Fixed bits
/// `word<10> == 0` and `word<4> == 0` (else UNALLOCATED).
#[inline]
fn decode_cond_compare(word: u32, out: &mut Instruction) {
    let sf = bit(word, 31);
    let op = bit(word, 30); // 0 = CCMN, 1 = CCMP
    let s = bit(word, 29); // must be 1
    let imm_form = bit(word, 11); // 1 = immediate, 0 = register
    let o2 = bit(word, 10); // must be 0
    let o3 = bit(word, 4); // must be 0
    let rm = bits(word, 16, 5);
    let cond = bits(word, 12, 4);
    let rn = bits(word, 5, 5);
    let nzcv = bits(word, 0, 4);

    if s != 1 || o2 != 0 || o3 != 0 {
        return;
    }
    let w = width_of(sf);

    let code = match (op, imm_form, sf) {
        (0, 0, 0) => Code::CcmnReg32,
        (0, 0, 1) => Code::CcmnReg64,
        (0, 1, 0) => Code::CcmnImm32,
        (0, 1, 1) => Code::CcmnImm64,
        (1, 0, 0) => Code::CcmpReg32,
        (1, 0, 1) => Code::CcmpReg64,
        (1, 1, 0) => Code::CcmpImm32,
        _ => Code::CcmpImm64,
    };
    out.set(code);

    out.push_operand(reg(false, w, rn));
    if imm_form == 1 {
        // The immediate form reuses the `Rm` field as a 5-bit unsigned `imm5`.
        out.push_operand(Operand::ImmUnsigned(rm as u64));
    } else {
        out.push_operand(reg(false, w, rm));
    }
    out.push_operand(Operand::ImmUnsigned(nzcv as u64));
    out.push_operand(Operand::Cond(Condition::from_u4(cond as u8)));
}

// ---------------------------------------------------------------------------
// Conditional select: CSEL/CSINC/CSINV/CSNEG.
// ---------------------------------------------------------------------------

/// `CSEL`/`CSINC`/`CSINV`/`CSNEG`. `op` (`word<30>`) and `o2` (`word<10>`) select
/// the variant; fixed bit `word<11> == 0` (else UNALLOCATED). Emits the
/// `CSET`/`CSETM` (Rm==Rn==ZR, cond not AL/NV) and `CINC`/`CINV`/`CNEG`
/// (Rm==Rn, cond not AL/NV) preferred aliases — all inverting the condition.
#[inline]
fn decode_cond_select(word: u32, out: &mut Instruction) {
    let sf = bit(word, 31);
    let op = bit(word, 30);
    let s = bit(word, 29); // must be 0
    let o1 = bit(word, 11); // must be 0
    let o2 = bit(word, 10);
    let rm = bits(word, 16, 5);
    let cond = bits(word, 12, 4);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    if s != 0 || o1 != 0 {
        return;
    }
    let w = width_of(sf);

    let code = match (sf, op, o2) {
        (0, 0, 0) => Code::Csel32,
        (1, 0, 0) => Code::Csel64,
        (0, 0, 1) => Code::Csinc32,
        (1, 0, 1) => Code::Csinc64,
        (0, 1, 0) => Code::Csinv32,
        (1, 1, 0) => Code::Csinv64,
        (0, 1, 1) => Code::Csneg32,
        _ => Code::Csneg64,
    };
    out.set(code);

    let c = Condition::from_u4(cond as u8);
    // `cond<3:1> != 111` is the ARM ARM precondition for the CSET/CINC family
    // (i.e. the condition is not AL/NV); equivalently `cond != 1110 && != 1111`.
    let cond_ok = (cond & 0b1110) != 0b1110;
    let is_csinc = op == 0 && o2 == 1; // -> CSET / CINC
    let is_csinv = op == 1 && o2 == 0; // -> CSETM / CINV
    let is_csneg = op == 1 && o2 == 1; // -> CNEG

    // CSET/CSETM: CSINC/CSINV with Rm==Rn==ZR.
    if (is_csinc || is_csinv) && rm == 31 && rn == 31 && cond_ok {
        out.set_mnemonic(if is_csinc { Mnemonic::Cset } else { Mnemonic::Csetm });
        out.push_operand(reg(false, w, rd));
        out.push_operand(Operand::Cond(c.invert()));
        return;
    }

    // CINC/CINV/CNEG: CSINC/CSINV/CSNEG with Rm==Rn. The Rm==Rn==ZR sub-case of
    // CSINC/CSINV was already consumed above as CSET/CSETM; CSNEG has no such
    // sub-alias, so its Rm==Rn==ZR form still becomes CNEG.
    if (is_csinc || is_csinv || is_csneg) && rm == rn && cond_ok {
        let m = if is_csinc {
            Mnemonic::Cinc
        } else if is_csinv {
            Mnemonic::Cinv
        } else {
            Mnemonic::Cneg
        };
        out.set_mnemonic(m);
        out.push_operand(reg(false, w, rd));
        out.push_operand(reg(false, w, rn));
        out.push_operand(Operand::Cond(c.invert()));
        return;
    }

    out.push_operand(reg(false, w, rd));
    out.push_operand(reg(false, w, rn));
    out.push_operand(reg(false, w, rm));
    out.push_operand(Operand::Cond(c));
}

// ---------------------------------------------------------------------------
// Data-processing (3 source): MADD/MSUB/SMADDL/.../UMULH.
// ---------------------------------------------------------------------------

/// `MADD`/`MSUB`/`SMADDL`/`SMSUBL`/`SMULH`/`UMADDL`/`UMSUBL`/`UMULH`. `op31`
/// (`word<23:21>`) and `o0` (`word<15>`) select the variant. Emits the
/// `MUL`/`MNEG`/`SMULL`/`SMNEGL`/`UMULL`/`UMNEGL` preferred aliases (Ra==ZR);
/// `SMULH`/`UMULH` carry no `Ra` operand.
#[inline]
fn decode_dp_3source(word: u32, out: &mut Instruction) {
    let sf = bit(word, 31);
    let op54 = bits(word, 29, 2); // must be 00
    let op31 = bits(word, 21, 3);
    let o0 = bit(word, 15);
    let rm = bits(word, 16, 5);
    let ra = bits(word, 10, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    if op54 != 0 {
        return;
    }

    // Resolve the encoding. Widen-forms (SMADDL/.../UMULH) are 64-bit only.
    let (code, kind) = match (sf, op31, o0) {
        (0, 0b000, 0) => (Code::Madd32, Dp3Kind::MaddSub),
        (1, 0b000, 0) => (Code::Madd64, Dp3Kind::MaddSub),
        (0, 0b000, 1) => (Code::Msub32, Dp3Kind::MaddSub),
        (1, 0b000, 1) => (Code::Msub64, Dp3Kind::MaddSub),
        (1, 0b001, 0) => (Code::Smaddl, Dp3Kind::WidenMaddSub),
        (1, 0b001, 1) => (Code::Smsubl, Dp3Kind::WidenMaddSub),
        // MADDPT/MSUBPT (FEAT_CPA): 64-bit only, op31=011. Plain `Xd, Xn, Xm, Xa`
        // (no MUL/MNEG alias).
        (1, 0b011, 0) => (Code::Maddpt, Dp3Kind::PtMaddSub),
        (1, 0b011, 1) => (Code::Msubpt, Dp3Kind::PtMaddSub),
        (1, 0b010, 0) => (Code::Smulh, Dp3Kind::HighMul),
        (1, 0b101, 0) => (Code::Umaddl, Dp3Kind::WidenMaddSub),
        (1, 0b101, 1) => (Code::Umsubl, Dp3Kind::WidenMaddSub),
        (1, 0b110, 0) => (Code::Umulh, Dp3Kind::HighMul),
        // All other combinations (incl. the 32-bit widen/high forms) are
        // UNALLOCATED.
        _ => return,
    };
    out.set(code);

    match kind {
        Dp3Kind::HighMul => {
            // SMULH/UMULH: Xd, Xn, Xm (no Ra).
            out.push_operand(reg(false, RegWidth::X64, rd));
            out.push_operand(reg(false, RegWidth::X64, rn));
            out.push_operand(reg(false, RegWidth::X64, rm));
        }
        Dp3Kind::MaddSub => {
            let w = width_of(sf);
            let is_msub = o0 == 1;
            // MUL/MNEG alias: Ra==ZR.
            if ra == 31 {
                out.set_mnemonic(if is_msub { Mnemonic::Mneg } else { Mnemonic::Mul });
                out.push_operand(reg(false, w, rd));
                out.push_operand(reg(false, w, rn));
                out.push_operand(reg(false, w, rm));
                return;
            }
            out.push_operand(reg(false, w, rd));
            out.push_operand(reg(false, w, rn));
            out.push_operand(reg(false, w, rm));
            out.push_operand(reg(false, w, ra));
        }
        Dp3Kind::WidenMaddSub => {
            // Xd, Wn, Wm, Xa. SMULL/UMULL/SMNEGL/UMNEGL alias: Ra==ZR.
            let is_msub = o0 == 1;
            let is_signed = op31 == 0b001;
            if ra == 31 {
                let m = match (is_signed, is_msub) {
                    (true, false) => Mnemonic::Smull,
                    (true, true) => Mnemonic::Smnegl,
                    (false, false) => Mnemonic::Umull,
                    (false, true) => Mnemonic::Umnegl,
                };
                out.set_mnemonic(m);
                out.push_operand(reg(false, RegWidth::X64, rd));
                out.push_operand(reg(false, RegWidth::W32, rn));
                out.push_operand(reg(false, RegWidth::W32, rm));
                return;
            }
            out.push_operand(reg(false, RegWidth::X64, rd));
            out.push_operand(reg(false, RegWidth::W32, rn));
            out.push_operand(reg(false, RegWidth::W32, rm));
            out.push_operand(reg(false, RegWidth::X64, ra));
        }
        Dp3Kind::PtMaddSub => {
            // MADDPT/MSUBPT: `Xd, Xn, Xm, Xa` (64-bit, no alias).
            out.push_operand(reg(false, RegWidth::X64, rd));
            out.push_operand(reg(false, RegWidth::X64, rn));
            out.push_operand(reg(false, RegWidth::X64, rm));
            out.push_operand(reg(false, RegWidth::X64, ra));
        }
    }
}

/// Operand-shape selector for the 3-source forms.
#[derive(Clone, Copy)]
enum Dp3Kind {
    /// `MADD`/`MSUB`: `Rd, Rn, Rm, Ra` (same width).
    MaddSub,
    /// `SMADDL`/`SMSUBL`/`UMADDL`/`UMSUBL`: `Xd, Wn, Wm, Xa`.
    WidenMaddSub,
    /// `SMULH`/`UMULH`: `Xd, Xn, Xm`.
    HighMul,
    /// `MADDPT`/`MSUBPT` (FEAT_CPA): `Xd, Xn, Xm, Xa` (64-bit, no alias).
    PtMaddSub,
}

// ---------------------------------------------------------------------------
// Data-processing (2 source): UDIV/SDIV/LSLV/.../CRC32*/SUBP/IRG/GMI/PACGA.
// ---------------------------------------------------------------------------

/// `UDIV`/`SDIV`/`LSLV`/`LSRV`/`ASRV`/`RORV` plus `CRC32*`, `SUBP`/`SUBPS`,
/// `IRG`/`GMI` and `PACGA`. `opcode` (`word<15:10>`) selects the operation; `S`
/// (`word<29>`) is `1` only for `SUBPS`. The variable-shift forms alias to
/// `LSL`/`LSR`/`ASR`/`ROR`. CRC32/SUBP/IRG/GMI/PACGA are feature-gated.
#[inline]
fn decode_dp_2source(word: u32, features: FeatureSet, out: &mut Instruction) {
    let sf = bit(word, 31);
    let s = bit(word, 29);
    let opcode = bits(word, 10, 6);
    let rm = bits(word, 16, 5);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);
    let w = width_of(sf);

    // SUBP / SUBPS: opcode == 000000, 64-bit only (FEAT_MTE).
    if opcode == 0b000000 {
        if sf != 1 {
            return;
        }
        if !features.has(Feature::Mte) {
            return;
        }
        // S selects SUBPS (flag-setting). Rn/Rm are SP-capable.
        out.set(if s == 1 { Code::SubpsDp } else { Code::SubpDp });
        out.push_operand(reg(false, RegWidth::X64, rd));
        out.push_operand(reg(true, RegWidth::X64, rn));
        out.push_operand(reg(true, RegWidth::X64, rm));
        return;
    }

    // The remaining forms all require S == 0.
    if s != 0 {
        return;
    }

    match opcode {
        // UDIV / SDIV.
        0b000010 => {
            out.set(if sf == 1 { Code::Udiv64 } else { Code::Udiv32 });
            push_rrr(out, w, rd, rn, rm);
        }
        0b000011 => {
            out.set(if sf == 1 { Code::Sdiv64 } else { Code::Sdiv32 });
            push_rrr(out, w, rd, rn, rm);
        }
        // Variable shifts -> LSL/LSR/ASR/ROR aliases.
        0b001000 => {
            out.set(if sf == 1 { Code::Lslv64 } else { Code::Lslv32 });
            out.set_mnemonic(Mnemonic::Lsl);
            push_rrr(out, w, rd, rn, rm);
        }
        0b001001 => {
            out.set(if sf == 1 { Code::Lsrv64 } else { Code::Lsrv32 });
            out.set_mnemonic(Mnemonic::Lsr);
            push_rrr(out, w, rd, rn, rm);
        }
        0b001010 => {
            out.set(if sf == 1 { Code::Asrv64 } else { Code::Asrv32 });
            out.set_mnemonic(Mnemonic::Asr);
            push_rrr(out, w, rd, rn, rm);
        }
        0b001011 => {
            out.set(if sf == 1 { Code::Rorv64 } else { Code::Rorv32 });
            out.set_mnemonic(Mnemonic::Ror);
            push_rrr(out, w, rd, rn, rm);
        }
        // IRG / GMI (FEAT_MTE, 64-bit).
        0b000100 => {
            if sf != 1 || !features.has(Feature::Mte) {
                return;
            }
            // IRG <Xd|SP>, <Xn|SP>{, <Xm>}. The optional <Xm> is omitted when it
            // is the zero register (UAL/Binary Ninja drop a trailing `, xzr`).
            out.set(Code::IrgDp);
            out.push_operand(reg(true, RegWidth::X64, rd));
            out.push_operand(reg(true, RegWidth::X64, rn));
            if rm != 31 {
                out.push_operand(reg(false, RegWidth::X64, rm));
            }
        }
        0b000101 => {
            if sf != 1 || !features.has(Feature::Mte) {
                return;
            }
            // GMI <Xd>, <Xn|SP>, <Xm>.
            out.set(Code::GmiDp);
            out.push_operand(reg(false, RegWidth::X64, rd));
            out.push_operand(reg(true, RegWidth::X64, rn));
            out.push_operand(reg(false, RegWidth::X64, rm));
        }
        // PACGA <Xd>, <Xn>, <Xm|SP> (FEAT_PAuth, 64-bit).
        0b001100 => {
            if sf != 1 || !features.has(Feature::PAuth) {
                return;
            }
            out.set(Code::Pacga);
            out.push_operand(reg(false, RegWidth::X64, rd));
            out.push_operand(reg(false, RegWidth::X64, rn));
            out.push_operand(reg(true, RegWidth::X64, rm));
        }
        // CRC32 family (FEAT_CRC32; gated under Base here as the corpus enables
        // it). Rd/Rn are W; Rm is W for B/H/W and X for the X-form.
        0b010000..=0b010111 => {
            decode_crc32(sf, opcode, rd, rn, rm, out);
        }
        // SMAX/UMAX/SMIN/UMIN (register form, FEAT_CSSC). opcode<5:2>==0110,
        // opcode<1> selects min (1) vs max (0), opcode<0> selects unsigned (1)
        // vs signed (0). Rd, Rn, Rm are all of the operation width.
        0b011000..=0b011011 => {
            if !features.has(Feature::Cssc) {
                return;
            }
            let code = match (sf, opcode) {
                (0, 0b011000) => Code::SmaxReg32,
                (1, 0b011000) => Code::SmaxReg64,
                (0, 0b011001) => Code::UmaxReg32,
                (1, 0b011001) => Code::UmaxReg64,
                (0, 0b011010) => Code::SminReg32,
                (1, 0b011010) => Code::SminReg64,
                (0, 0b011011) => Code::UminReg32,
                _ => Code::UminReg64,
            };
            out.set(code);
            push_rrr(out, w, rd, rn, rm);
        }
        _ => {}
    }
}

/// Decode the `CRC32*`/`CRC32C*` 2-source forms. The size is `opcode<1:0>`
/// (`00`=B, `01`=H, `10`=W, `11`=X); the X size requires `sf==1`, all others
/// `sf==0`. `Rd`/`Rn` are always `W`; `Rm` is `X` only for the X size.
#[inline]
fn decode_crc32(sf: u32, opcode: u32, rd: u32, rn: u32, rm: u32, out: &mut Instruction) {
    let c = bit(opcode, 2); // 0 = CRC32, 1 = CRC32C
    let sz = opcode & 0b11; // 00/01/10/11 -> B/H/W/X
    // The X size is 64-bit (sf must be 1); B/H/W are 32-bit (sf must be 0).
    if sz == 0b11 {
        if sf != 1 {
            return;
        }
    } else if sf != 0 {
        return;
    }

    let code = match (c, sz) {
        (0, 0b00) => Code::Crc32b,
        (0, 0b01) => Code::Crc32h,
        (0, 0b10) => Code::Crc32w,
        (0, 0b11) => Code::Crc32x,
        (_, 0b00) => Code::Crc32cb,
        (_, 0b01) => Code::Crc32ch,
        (_, 0b10) => Code::Crc32cw,
        (_, _) => Code::Crc32cx,
    };
    out.set(code);

    // Rd, Rn are W; Rm width depends on size (X for the X-form).
    let rm_w = if sz == 0b11 {
        RegWidth::X64
    } else {
        RegWidth::W32
    };
    out.push_operand(reg(false, RegWidth::W32, rd));
    out.push_operand(reg(false, RegWidth::W32, rn));
    out.push_operand(reg(false, rm_w, rm));
}

/// Push three same-width GP registers `Rd, Rn, Rm`.
#[inline]
fn push_rrr(out: &mut Instruction, w: RegWidth, rd: u32, rn: u32, rm: u32) {
    out.push_operand(reg(false, w, rd));
    out.push_operand(reg(false, w, rn));
    out.push_operand(reg(false, w, rm));
}

// ---------------------------------------------------------------------------
// Data-processing (1 source): RBIT/REV16/REV/REV32/CLZ/CLS + PAuth forms.
// ---------------------------------------------------------------------------

/// `RBIT`/`REV16`/`REV`/`REV32`/`CLZ`/`CLS` plus the pointer-authentication
/// 1-source forms (`PACIA`/`AUTIA`/.../`XPACI`/`XPACD`, FEAT_PAuth). `opcode2`
/// (`word<20:16>`) is `00000` for the basic group and `00001` for PAuth; the
/// operation is `opcode` (`word<15:10>`).
#[inline]
fn decode_dp_1source(word: u32, features: FeatureSet, out: &mut Instruction) {
    let sf = bit(word, 31);
    let s = bit(word, 29); // must be 0
    let opcode2 = bits(word, 16, 5);
    let opcode = bits(word, 10, 6);
    let rn = bits(word, 5, 5);
    let rd = bits(word, 0, 5);

    if s != 0 {
        return;
    }

    match opcode2 {
        0b00000 => decode_dp_1source_basic(sf, opcode, rn, rd, features, out),
        0b00001 => decode_dp_1source_pauth(sf, opcode, rn, rd, features, out),
        _ => {}
    }
}

/// The non-PAuth 1-source forms (`opcode2 == 00000`): `RBIT`/`REV16`/`REV`/
/// `REV32`/`CLZ`/`CLS` plus the FEAT_CSSC `ABS`/`CNT`/`CTZ` forms (gated on
/// [`Feature::Cssc`]). All take a single `Rd, Rn` of the operation width.
#[inline]
fn decode_dp_1source_basic(
    sf: u32,
    opcode: u32,
    rn: u32,
    rd: u32,
    features: FeatureSet,
    out: &mut Instruction,
) {
    let w = width_of(sf);
    let code = match (sf, opcode) {
        (0, 0b000000) => Code::Rbit32,
        (1, 0b000000) => Code::Rbit64,
        (0, 0b000001) => Code::Rev1632,
        (1, 0b000001) => Code::Rev1664,
        // REV (32-bit) uses opcode 000010; REV32 (64-bit) uses opcode 000010.
        (0, 0b000010) => Code::Rev32Bit,
        (1, 0b000010) => Code::Rev3264,
        // REV (64-bit) uses opcode 000011 (32-bit opcode 000011 is UNALLOCATED).
        (1, 0b000011) => Code::Rev64Bit,
        (0, 0b000100) => Code::Clz32,
        (1, 0b000100) => Code::Clz64,
        (0, 0b000101) => Code::Cls32,
        (1, 0b000101) => Code::Cls64,
        // FEAT_CSSC: CTZ (000110), CNT (000111), ABS (001000).
        (_, 0b000110) | (_, 0b000111) | (_, 0b001000) => {
            if !features.has(Feature::Cssc) {
                return;
            }
            match (sf, opcode) {
                (0, 0b000110) => Code::Ctz32,
                (1, 0b000110) => Code::Ctz64,
                (0, 0b000111) => Code::Cnt32,
                (1, 0b000111) => Code::Cnt64,
                (0, 0b001000) => Code::Abs32,
                _ => Code::Abs64,
            }
        }
        _ => return,
    };
    out.set(code);
    out.push_operand(reg(false, w, rd));
    out.push_operand(reg(false, w, rn));
}

/// The pointer-authentication 1-source forms (`opcode2 == 00001`, FEAT_PAuth,
/// 64-bit only). The `Z` variants (`opcode<0> == 1`, encoded with `Rn==11111`)
/// carry only the destination register.
#[inline]
fn decode_dp_1source_pauth(
    sf: u32,
    opcode: u32,
    rn: u32,
    rd: u32,
    features: FeatureSet,
    out: &mut Instruction,
) {
    // PAuth DP forms are 64-bit only and gated on FEAT_PAuth.
    if sf != 1 || !features.has(Feature::PAuth) {
        return;
    }

    // The 6-bit `opcode` (word<15:10>) fully selects the operation. The `Z`
    // variants (`001xxx` and the `XPAC*` forms `0100xx`) are encoded with
    // `Rn == 11111` and carry only the destination register.
    let (code, z) = match opcode {
        // PAC*/AUT* with an explicit source register (`000xxx`).
        0b000000 => (Code::PaciaDp, false),
        0b000001 => (Code::PacibDp, false),
        0b000010 => (Code::PacdaDp, false),
        0b000011 => (Code::PacdbDp, false),
        0b000100 => (Code::AutiaDp, false),
        0b000101 => (Code::AutibDp, false),
        0b000110 => (Code::AutdaDp, false),
        0b000111 => (Code::AutdbDp, false),
        // PAC*Z/AUT*Z zero-source forms (`001xxx`, Rn==11111).
        0b001000 => (Code::PacizaDp, true),
        0b001001 => (Code::PacizbDp, true),
        0b001010 => (Code::PacdzaDp, true),
        0b001011 => (Code::PacdzbDp, true),
        0b001100 => (Code::AutizaDp, true),
        0b001101 => (Code::AutizbDp, true),
        0b001110 => (Code::AutdzaDp, true),
        0b001111 => (Code::AutdzbDp, true),
        // XPACI/XPACD (`010000`/`010001`, Rn==11111).
        0b010000 => (Code::XpaciDp, true),
        0b010001 => (Code::XpacdDp, true),
        _ => return,
    };

    // The Z / XPAC forms require Rn to be the zero register.
    if z && rn != 31 {
        return;
    }

    out.set(code);
    out.push_operand(reg(false, RegWidth::X64, rd));
    if !z {
        // Rn is SP-capable for the PAC*/AUT* (non-Z) data-processing forms.
        out.push_operand(reg(true, RegWidth::X64, rn));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{BufSink, FmtFormatter, Formatter};

    /// Decode `word` at ip 0 with all features, render with the default
    /// formatter, and assert the text equals `expected`. Allocation-free so it
    /// builds on the default no-`alloc` test tier.
    #[track_caller]
    fn render(word: u32, expected: &str) {
        let mut insn = Instruction::default();
        decode(word, 0, FeatureSet::ALL, &mut insn);
        let mut buf = [0u8; 128];
        let mut sink = BufSink::new(&mut buf);
        FmtFormatter::new().format(&insn, &mut sink);
        assert!(!sink.overflowed(), "BufSink overflowed rendering {expected:?}");
        assert_eq!(sink.as_str(), expected, "word={word:#010x}");
    }

    #[test]
    fn logical_shifted_and_aliases() {
        render(0x8AC7E6A3, "and     x3, x21, x7, ror #0x39");
        render(0x2A1503E7, "mov     w7, w21");
        render(0xAA731FFA, "mvn     x26, x19, lsr #0x7");
        render(0xEA863DFF, "tst     x15, x6, asr #0xf");
        // LSL #0 elided on a plain shifted form.
        render(0xCA08037C, "eor     x28, x27, x8");
    }

    #[test]
    fn flag_manipulation_rmif_setf() {
        // RMIF <Xn>, #shift, #mask.
        render(0xBA1785C9, "rmif    x14, #0x2f, #0x9");
        render(0xBA000589, "rmif    x12, #0x0, #0x9");
        // SETF8 / SETF16 <Wn>.
        render(0x3A000A4D, "setf8   w18");
        render(0x3A00494D, "setf16  w10");
    }

    #[test]
    fn addsub_shifted_and_aliases() {
        render(0x8B4BB429, "add     x9, x1, x11, lsr #0x2d");
        render(0xEB8030BF, "cmp     x5, x0, asr #0xc");
        render(0xAB88807F, "cmn     x3, x8, asr #0x20");
        render(0xCB9017FA, "neg     x26, x16, asr #0x5");
        render(0xEB102BFE, "negs    x30, x16, lsl #0xa");
    }

    #[test]
    fn addsub_extended() {
        render(0x8B3B6186, "add     x6, x12, x27, uxtx");
        render(0x8B304F9F, "add     sp, x28, w16, uxtw #0x3");
        render(0xEB2A61FF, "cmp     x15, x10, uxtx");
        // LSL special: uxtx + SP involved + non-zero amount renders as lsl.
        render(0x8B206FFF, "add     sp, sp, x0, lsl #0x3");
        // LSL special with zero amount elides the modifier entirely.
        render(0x8B2063FF, "add     sp, sp, x0");
        // 32-bit extended: Rm stays a W view even for uxtx/sxtx options.
        render(0x0B33604F, "add     w15, w2, w19, uxtx");
        render(0x4B2DEB78, "sub     w24, w27, w13, sxtx #0x2");
    }

    #[test]
    fn addsub_carry_and_aliases() {
        render(0x9A0E0358, "adc     x24, x26, x14");
        render(0xDA1B03E2, "ngc     x2, x27");
        render(0xFA0703F1, "ngcs    x17, x7");
    }

    #[test]
    fn cond_compare() {
        render(0xBA45B044, "ccmn    x2, x5, #0x4, lt");
        render(0xBA537B2F, "ccmn    x25, #0x13, #0xf, vc");
        render(0xFA439341, "ccmp    x26, x3, #0x1, ls");
        render(0x7A4B5A46, "ccmp    w18, #0xb, #0x6, pl");
    }

    #[test]
    fn cond_select_and_aliases() {
        render(0x9A95C0C0, "csel    x0, x6, x21, gt");
        render(0x9A9F97E4, "cset    x4, hi");
        render(0x5A9F53EE, "csetm   w14, mi");
        render(0x9A8BC574, "cinc    x20, x11, le");
        render(0xDA9B5367, "cinv    x7, x27, mi");
        render(0xDA9E97D1, "cneg    x17, x30, hi");
    }

    #[test]
    fn dp_3source_and_aliases() {
        render(0x9B0D31FC, "madd    x28, x15, x13, x12");
        render(0x9B057C24, "mul     x4, x1, x5");
        render(0x1B0CFC7C, "mneg    w28, w3, w12");
        render(0x9B2B0AF3, "smaddl  x19, w23, w11, x2");
        render(0x9B337F45, "smull   x5, w26, w19");
        render(0x9B2CFC4D, "smnegl  x13, w2, w12");
        render(0x9B47643B, "smulh   x27, x1, x7");
        render(0x9BB07F03, "umull   x3, w24, w16");
        render(0x9BDD5542, "umulh   x2, x10, x29");
    }

    #[test]
    fn dp_2source() {
        render(0x9ADF09D9, "udiv    x25, x14, xzr");
        render(0x9ACB2231, "lsl     x17, x17, x11");
        render(0x1AD22518, "lsr     w24, w8, w18");
        render(0x9ADB2B14, "asr     x20, x24, x27");
        render(0x1AD82F76, "ror     w22, w27, w24");
        render(0x1AC74347, "crc32b  w7, w26, w7");
        render(0x9AC24E71, "crc32x  w17, w19, x2");
        render(0x9ADB004D, "subp    x13, x2, x27");
        render(0xBAD601FA, "subps   x26, x15, x22");
        // SUBPS with Rd==ZR stays `subps` (Binary Ninja does not use the CMPP alias).
        render(0xBAC202FF, "subps   xzr, x23, x2");
        render(0x9AC112F1, "irg     x17, x23, x1");
        // IRG with Rm==ZR drops the optional third operand.
        render(0x9ADF130C, "irg     x12, x24");
        render(0x9ADB14C8, "gmi     x8, x6, x27");
        render(0x9AD03137, "pacga   x23, x9, x16");
    }

    #[test]
    fn cssc_1source_and_2source() {
        // 1-source: ABS / CNT / CTZ (FEAT_CSSC).
        render(0xDAC02020, "abs     x0, x1");
        render(0x5AC02083, "abs     w3, w4");
        render(0xDAC01CC5, "cnt     x5, x6");
        render(0x5AC01D07, "cnt     w7, w8");
        render(0xDAC01949, "ctz     x9, x10");
        render(0x5AC0198B, "ctz     w11, w12");
        // 2-source register min/max (FEAT_CSSC).
        render(0x9AC26020, "smax    x0, x1, x2");
        render(0x1ACF61CD, "smax    w13, w14, w15");
        render(0x9AD26A30, "smin    x16, x17, x18");
        render(0x9AD56693, "umax    x19, x20, x21");
        render(0x9AD86EF6, "umin    x22, x23, x24");
    }

    #[test]
    fn dp_1source() {
        render(0xDAC1030C, "pacia   x12, x24");
        render(0xDAC1128D, "autia   x13, x20");
        render(0xDAC123ED, "paciza  x13");
        render(0xDAC133F8, "autiza  x24");
        render(0xDAC143F2, "xpaci   x18");
    }

    #[test]
    fn never_panics_sample() {
        // A range of words in the dp_reg space must never panic.
        for w in (0x0A00_0000u32..0x0A00_0000u32.wrapping_add(4096)).step_by(7) {
            let mut insn = Instruction::default();
            decode(w | 0x1000_0000, 0, FeatureSet::ALL, &mut insn);
        }
    }
}
