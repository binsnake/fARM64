//! Inverse of [`crate::decode::sve::sve_perm`] — permute / predicate / compare.

use super::{arr_size, esize, fld, g, imm, lane, p, pred_qual, read_pattern_opt, sfp, z};
use crate::encode::EncodeError;
use crate::enums::VectorArrangement as VA;
use crate::instruction::Instruction;
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::{Operand, PredQual};
use crate::register::RegClass;

use Code::*;

/// `true` for every permute / predicate / compare SVE [`Code`].
pub(super) fn is_perm(code: Code) -> bool {
    matches!(
        code,
        SveLuti2 | SveLuti4 | SveLuti4Two
            | SveZipUzpTrnZzz | SveZipUzpTrnQ | SveZipUzpTrnPpp | SveTbl | SveTbl2 | SveTbx | SveTbxq | SveRevZz
            | SveRevP | SveUnpk | SvePunpk | SveExtDes | SveExtCon | SveCompact | SveSpliceDes
            | SveSpliceCon | SveClastZ | SveClastV | SveClastR | SveLastV | SveLastR | SveRevbhw
            | SveSelPred | SvePredLogical | SveBrkpPred | SveBrkPred | SveBrkn | SveRdffr
            | SveRdffrPred | SveWrffr | SveSetffr | SvePfalse | SvePtest | SvePfirst | SvePnext
            | SvePtrue | SvePsel | SveLastp | SveFirstp | SveWhile | SveWhileRw | SveCterm
            | SveWhilePair | SveWhilePn
    )
}

/// Encode a permute / predicate / compare SVE instruction.
pub(super) fn enc(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
    let w = match code {
        // ---- vector ZIP/UZP/TRN (element) ----
        SveZipUzpTrnZzz => {
            let op = zip_op(insn.mnemonic())?;
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            base05(size) | fld(1, 21) | fld(zm, 16) | fld(0b011, 13) | fld(op, 10) | fld(zn, 5) | zd
        }
        // ---- 128-bit Q permute ----
        SveZipUzpTrnQ => {
            let (fam, h) = zip_q_fam(insn.mnemonic())?;
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            base05(0b10) | fld(1, 21) | fld(zm, 16) | fld(0b000, 13) | fld(fam, 11) | fld(h, 10)
                | fld(zn, 5)
                | zd
        }
        // ---- predicate ZIP/UZP/TRN ----
        SveZipUzpTrnPpp => {
            let op = zip_op(insn.mnemonic())?;
            let size = esize(insn, 0)?;
            let pd = p(insn, 0)?;
            let pn = p(insn, 1)?;
            let pm = p(insn, 2)?;
            base05(size) | fld(1, 21) | fld(pm, 16) | fld(0b010, 13) | fld(op, 10) | fld(pn, 5) | pd
        }
        // ---- TBL / TBL2 / TBX ----
        SveTbl => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let zn = list_first(insn, 1)?;
            let zm = z(insn, 2)?;
            base05(size) | fld(1, 21) | fld(zm, 16) | fld(0b001100, 10) | fld(zn, 5) | zd
        }
        SveTbl2 => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let zn = list_first(insn, 1)?;
            let zm = z(insn, 2)?;
            base05(size) | fld(1, 21) | fld(zm, 16) | fld(0b001010, 10) | fld(zn, 5) | zd
        }
        SveTbx => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            base05(size) | fld(1, 21) | fld(zm, 16) | fld(0b001011, 10) | fld(zn, 5) | zd
        }
        // ---- TBXQ (SVE2.1 128-bit-segment table lookup with base) ----
        SveTbxq => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            base05(size) | fld(1, 21) | fld(zm, 16) | fld(0b001101, 10) | fld(zn, 5) | zd
        }
        // ---- LUTI2 / LUTI4 (FEAT_LUT lookup table) ----
        SveLuti2 => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let zn = list_first(insn, 1)?;
            let zm = z(insn, 2)?;
            let index = lane(insn, 2)?;
            let common = base45() | fld(zm, 16) | fld(zn, 5) | zd;
            match size {
                // .B: 2-bit index at <23:22>; <12>=1, <11>=0.
                0 => {
                    if index > 3 {
                        return Err(EncodeError::InvalidOperand);
                    }
                    common | fld((index >> 1) & 1, 23) | fld(index & 1, 22) | fld(1, 12)
                }
                // .H: 3-bit index = <23>:<22>:<12>; <11>=1.
                1 => {
                    if index > 7 {
                        return Err(EncodeError::InvalidOperand);
                    }
                    common
                        | fld((index >> 2) & 1, 23)
                        | fld((index >> 1) & 1, 22)
                        | fld(index & 1, 12)
                        | fld(1, 11)
                }
                _ => return Err(EncodeError::InvalidOperand),
            }
        }
        SveLuti4 => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let zn = list_first(insn, 1)?;
            let zm = z(insn, 2)?;
            let index = lane(insn, 2)?;
            let common = base45() | fld(1, 10) | fld(zm, 16) | fld(zn, 5) | zd;
            match size {
                // .B: 1-bit index at <23>; <22>=1, <12>=0, <11>=0.
                0 => {
                    if index > 1 {
                        return Err(EncodeError::InvalidOperand);
                    }
                    common | fld(index & 1, 23) | fld(1, 22)
                }
                // .H: 2-bit index at <23:22>; <12>=1, <11>=1.
                1 => {
                    if index > 3 {
                        return Err(EncodeError::InvalidOperand);
                    }
                    common | fld((index >> 1) & 1, 23) | fld(index & 1, 22) | fld(1, 12) | fld(1, 11)
                }
                _ => return Err(EncodeError::InvalidOperand),
            }
        }
        SveLuti4Two => {
            // Two table registers; `.H` only. 2-bit index at <23:22>; <12>=1,
            // <11>=0, <10>=1.
            if esize(insn, 0)? != 1 {
                return Err(EncodeError::InvalidOperand);
            }
            let zd = z(insn, 0)?;
            let zn = list_first(insn, 1)?;
            let zm = z(insn, 2)?;
            let index = lane(insn, 2)?;
            if index > 3 {
                return Err(EncodeError::InvalidOperand);
            }
            base45()
                | fld(1, 10)
                | fld(1, 12)
                | fld((index >> 1) & 1, 23)
                | fld(index & 1, 22)
                | fld(zm, 16)
                | fld(zn, 5)
                | zd
        }
        // ---- REV (vector) ----
        SveRevZz => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            base05(size) | fld(1, 21) | fld(0b11000, 16) | fld(0b001110, 10) | fld(zn, 5) | zd
        }
        // ---- SUNPK/UUNPK ----
        SveUnpk => {
            let (u, h) = match insn.mnemonic() {
                Mnemonic::Sunpklo => (0, 0),
                Mnemonic::Sunpkhi => (0, 1),
                Mnemonic::Uunpklo => (1, 0),
                _ => (1, 1),
            };
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            base05(size) | fld(1, 21) | fld(0b100, 18) | fld(u, 17) | fld(h, 16) | fld(0b001110, 10)
                | fld(zn, 5)
                | zd
        }
        // ---- predicate REV / PUNPK ----
        SveRevP => {
            let size = esize(insn, 0)?;
            let pd = p(insn, 0)?;
            let pn = p(insn, 1)?;
            base05(size) | fld(1, 21) | fld(0b10100, 16) | fld(0b010, 13) | fld(pn, 5) | pd
        }
        SvePunpk => {
            let hi = matches!(insn.mnemonic(), Mnemonic::Punpkhi);
            let pd = p(insn, 0)?;
            let pn = p(insn, 1)?;
            base05(0) | fld(1, 21) | fld(0b1000, 17) | fld(u32::from(hi), 16) | fld(0b010, 13)
                | fld(pn, 5)
                | pd
        }
        // ---- EXT ----
        SveExtDes => {
            let zdn = z(insn, 0)?;
            let zm = z(insn, 2)?;
            let v = imm(insn, 3)? as u32;
            let imm8h = (v >> 3) & 0x1f;
            let imm8l = v & 7;
            base05(0) | fld(1, 21) | fld(imm8h, 16) | fld(0b000, 13) | fld(imm8l, 10) | fld(zm, 5)
                | zdn
        }
        SveExtCon => {
            let zd = z(insn, 0)?;
            let zn = list_first(insn, 1)?;
            // operand 2 is the spurious z0.b; operand 3 is the imm.
            let v = imm(insn, 3)? as u32;
            let imm8h = (v >> 3) & 0x1f;
            let imm8l = v & 7;
            base05(0) | fld(1, 22) | fld(1, 21) | fld(imm8h, 16) | fld(0b000, 13) | fld(imm8l, 10)
                | fld(zn, 5)
                | zd
        }
        // ---- COMPACT / SPLICE / CLAST / LAST / REVB/H/W / REVD ----
        SveCompact => {
            let a = arr_of(insn, 0)?;
            let zd = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            // COMPACT: <23>=1, <22>=size (.s->0, .d->1), <20:16>=00001, <15:13>=100.
            let b22 = if a == VA::Sd { 1 } else { 0 };
            base05(0) | fld(1, 23) | fld(b22, 22) | fld(1, 21) | fld(0b00001, 16) | fld(0b100, 13)
                | fld(pg, 10)
                | fld(zn, 5)
                | zd
        }
        SveSpliceDes => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zm = z(insn, 3)?;
            base05(size) | fld(1, 21) | fld(0b01100, 16) | fld(0b100, 13) | fld(pg, 10) | fld(zm, 5)
                | zd
        }
        SveSpliceCon => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = list_first(insn, 2)?;
            base05(size) | fld(1, 21) | fld(0b01101, 16) | fld(0b100, 13) | fld(pg, 10) | fld(zn, 5)
                | zd
        }
        SveClastZ => {
            let b = clast_b(insn.mnemonic())?;
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zm = z(insn, 3)?;
            base05(size) | fld(1, 21) | fld(0b01000, 16) | fld(b, 16) | fld(0b100, 13) | fld(pg, 10)
                | fld(zm, 5)
                | zd
        }
        SveClastV => {
            let b = clast_b(insn.mnemonic())?;
            let size = clast_r_size(insn)?;
            let vd = sfp(insn, 0)?;
            let pg = p(insn, 1)?;
            let zm = z(insn, 3)?;
            base05(size) | fld(1, 21) | fld(0b01010, 16) | fld(b, 16) | fld(0b100, 13) | fld(pg, 10)
                | fld(zm, 5)
                | vd
        }
        SveClastR => {
            let b = clast_b(insn.mnemonic())?;
            let size = clast_r_size(insn)?;
            let rd = g(insn, 0)?;
            let pg = p(insn, 1)?;
            let zm = z(insn, 3)?;
            base05(size) | fld(1, 21) | fld(0b10000, 16) | fld(b, 16) | fld(0b101, 13) | fld(pg, 10)
                | fld(zm, 5)
                | rd
        }
        SveLastV => {
            let b = clast_b(insn.mnemonic())?;
            let size = clast_r_size(insn)?;
            let vd = sfp(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            base05(size) | fld(1, 21) | fld(0b00010, 16) | fld(b, 16) | fld(0b100, 13) | fld(pg, 10)
                | fld(zn, 5)
                | vd
        }
        SveLastR => {
            let b = clast_b(insn.mnemonic())?;
            let size = clast_r_size(insn)?;
            let rd = g(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            base05(size) | fld(1, 21) | fld(0b00000, 16) | fld(b, 16) | fld(0b101, 13) | fld(pg, 10)
                | fld(zn, 5)
                | rd
        }
        SveRevbhw => {
            let opc = match insn.mnemonic() {
                Mnemonic::Revb => 0b00100,
                Mnemonic::Revh => 0b00101,
                _ => 0b00110,
            };
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            // `<15:13>` is 100 for merging (`/m`), 101 for FEAT_SVE2p1 zeroing (`/z`).
            let sel = if matches!(pred_qual(insn, 1), Some(PredQual::Zeroing)) { 0b101 } else { 0b100 };
            base05(size) | fld(1, 21) | fld(opc, 16) | fld(sel, 13) | fld(pg, 10) | fld(zn, 5) | zd
        }
        RevdZPZ | SveRevdZpzZero => {
            let zd = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            // `<13>` M-bit: 0 merging (`/m`), 1 FEAT_SVE2p1 zeroing (`/z`).
            let sel = if matches!(code, SveRevdZpzZero) { 0b101 } else { 0b100 };
            base05(0) | fld(1, 21) | fld(0b01110, 16) | fld(sel, 13) | fld(pg, 10) | fld(zn, 5) | zd
        }
        // ---- predicate logical / SEL / break / generation / FFR ----
        SveSelPred => enc_sel_pred(insn)?,
        SvePredLogical => enc_pred_logical(insn)?,
        SveBrkpPred => enc_brkp(insn)?,
        SveBrkPred => enc_brk(insn)?,
        SveBrkn => enc_brkn(insn)?,
        SveRdffr => {
            let pd = p(insn, 0)?;
            fld(0b00100101, 24) | fld(0b00011001111100000000, 4) | pd
        }
        SveRdffrPred => {
            let s = if matches!(insn.mnemonic(), Mnemonic::Rdffrs) { 1 } else { 0 };
            let pd = p(insn, 0)?;
            let pg = p(insn, 1)?;
            fld(0b00100101, 24) | fld(s, 22) | fld(0b011000, 16) | fld(0b1111000, 9) | fld(pg, 5) | pd
        }
        SveWrffr => {
            let pn = p(insn, 0)?;
            fld(0b00100101, 24) | fld(0b001010001001, 12) | fld(pn, 5)
        }
        SveSetffr => fld(0b00100101, 24) | 0x002C_9000,
        SvePfalse => {
            let pd = p(insn, 0)?;
            fld(0b00100101, 24) | fld(0b00011000111001000000, 4) | pd
        }
        SvePtest => {
            let pg = p(insn, 0)?;
            let pn = p(insn, 1)?;
            fld(0b00100101, 24) | fld(0b01010000, 16) | fld(0b11, 14) | fld(pg, 10) | fld(pn, 5)
        }
        SvePfirst => {
            let pdn = p(insn, 0)?;
            let pg = p(insn, 1)?;
            fld(0b00100101, 24) | fld(0b01011000, 16) | fld(0b1100000, 9) | fld(pg, 5) | pdn
        }
        SvePnext => {
            let size = esize(insn, 0)?;
            let pdn = p(insn, 0)?;
            let pg = p(insn, 1)?;
            fld(0b00100101, 24) | fld(size, 22) | fld(0b011001, 16) | fld(0b11000, 11) | fld(1, 10)
                | fld(pg, 5)
                | pdn
        }
        SvePtrue => {
            let s = if matches!(insn.mnemonic(), Mnemonic::Ptrues) { 1 } else { 0 };
            let size = esize(insn, 0)?;
            let pd = p(insn, 0)?;
            let pattern = read_pattern_opt(insn, 1);
            fld(0b00100101, 24) | fld(size, 22) | fld(0b011, 19) | fld(s, 16) | fld(0b11100, 11)
                | fld(pattern, 5)
                | pd
        }
        // ---- LASTP / FIRSTP (extract predicate-as-counter) ----
        SveLastp | SveFirstp => {
            let size = esize(insn, 2)?; // element from Pn.T (operand 2)
            let rd = g(insn, 0)?;
            let pg = p(insn, 1)?;
            let pn = p(insn, 2)?;
            let op = if matches!(code, SveLastp) { 0b00010 } else { 0b00001 };
            fld(0b00100101, 24) | fld(size, 22) | fld(1, 21) | fld(op, 16) | fld(0b10, 14)
                | fld(pg, 10)
                | fld(pn, 5)
                | rd
        }
        // ---- predicate-indexed DUP ----
        SvePsel => enc_psel(insn)?,
        // ---- WHILE / CTERM ----
        SveWhile => enc_while(insn)?,
        // ---- SVE2.1 WHILE predicate-pair / predicate-as-counter ----
        SveWhilePair => enc_while_pair(insn)?,
        SveWhilePn => enc_while_pn(insn)?,
        SveWhileRw => {
            let size = esize(insn, 0)?;
            let pd = p(insn, 0)?;
            let rn = g(insn, 1)?;
            let rm = g(insn, 2)?;
            let rw = if matches!(insn.mnemonic(), Mnemonic::Whilerw) { 1 } else { 0 };
            fld(0b00100101, 24) | fld(size, 22) | fld(1, 21) | fld(rm, 16) | fld(0b001100, 10)
                | fld(rn, 5)
                | fld(rw, 4)
                | pd
        }
        SveCterm => {
            let sz = match insn.op(0) {
                Operand::Reg { reg, .. } if reg.width_bits() == 64 => 1,
                _ => 0,
            };
            let rn = g(insn, 0)?;
            let rm = g(insn, 1)?;
            let op = if matches!(insn.mnemonic(), Mnemonic::Ctermne) { 1 } else { 0 };
            fld(0b00100101, 24) | fld(1, 23) | fld(sz, 22) | fld(1, 21) | fld(rm, 16)
                | fld(0b001000, 10)
                | fld(rn, 5)
                | fld(op, 4)
        }
        _ => return Ok(None),
    };
    Ok(Some(w))
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// Base word for top byte 0x05 with `size` in `<23:22>`.
#[inline]
fn base05(size: u32) -> u32 {
    fld(0b00000101, 24) | fld(size, 22)
}

/// Skeleton for the FEAT_LUT `LUTI2`/`LUTI4` reads (top byte 0x45, `<21>=1`,
/// `<15:13>=0b101`). `<23:22>`/`<12>`/`<11>`/`<10>` are filled in per form.
#[inline]
fn base45() -> u32 {
    fld(0b01000101, 24) | fld(1, 21) | fld(0b101, 13)
}

/// The arrangement of operand `n`.
#[inline]
fn arr_of(insn: &Instruction, n: usize) -> Result<VA, EncodeError> {
    match insn.op(n) {
        Operand::Reg { arr: Some(a), .. } => Ok(a),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// First register of a [`Operand::MultiReg`] list at `n`.
fn list_first(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    match insn.op(n) {
        Operand::MultiReg { regs, .. } => Ok(regs[0].number() as u32),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The `<12:10>` op for a ZIP/UZP/TRN mnemonic.
fn zip_op(m: Mnemonic) -> Result<u32, EncodeError> {
    Ok(match m {
        Mnemonic::Zip1 => 0b000,
        Mnemonic::Zip2 => 0b001,
        Mnemonic::Uzp1 => 0b010,
        Mnemonic::Uzp2 => 0b011,
        Mnemonic::Trn1 => 0b100,
        Mnemonic::Trn2 => 0b101,
        _ => return Err(EncodeError::InvalidOperand),
    })
}

/// The `(fam, h)` for a 128-bit Q ZIP/UZP/TRN mnemonic.
fn zip_q_fam(m: Mnemonic) -> Result<(u32, u32), EncodeError> {
    Ok(match m {
        Mnemonic::Zip1 => (0b00, 0),
        Mnemonic::Zip2 => (0b00, 1),
        Mnemonic::Uzp1 => (0b01, 0),
        Mnemonic::Uzp2 => (0b01, 1),
        Mnemonic::Trn1 => (0b11, 0),
        Mnemonic::Trn2 => (0b11, 1),
        _ => return Err(EncodeError::InvalidOperand),
    })
}

/// The `B` bit (CLASTA/LASTA=0 / CLASTB/LASTB=1).
fn clast_b(m: Mnemonic) -> Result<u32, EncodeError> {
    Ok(match m {
        Mnemonic::Clasta | Mnemonic::Lasta => 0,
        Mnemonic::Clastb | Mnemonic::Lastb => 1,
        _ => return Err(EncodeError::InvalidOperand),
    })
}

/// The `size` for CLAST/LAST to a GP register, read from the Z source operand.
fn clast_r_size(insn: &Instruction) -> Result<u32, EncodeError> {
    for n in 0..insn.op_count() {
        if let Operand::Reg {
            reg,
            arr: Some(a),
            ..
        } = insn.op(n)
        {
            if reg.class() == RegClass::Sve {
                return arr_size(a);
            }
        }
    }
    Err(EncodeError::InvalidOperand)
}

/// SEL (predicate) and its MOV alias.
fn enc_sel_pred(insn: &Instruction) -> Result<u32, EncodeError> {
    let pd = p(insn, 0)?;
    let base = fld(0b00100101, 24) | fld(0b01, 14);
    if matches!(insn.mnemonic(), Mnemonic::Mov) {
        let pg = p(insn, 1)?;
        let pn = p(insn, 2)?;
        Ok(base | fld(pd, 16) | fld(pg, 10) | fld(1, 9) | fld(pn, 5) | fld(1, 4) | pd)
    } else {
        let pg = p(insn, 1)?;
        let pn = p(insn, 2)?;
        let pm = p(insn, 3)?;
        Ok(base | fld(pm, 16) | fld(pg, 10) | fld(1, 9) | fld(pn, 5) | fld(1, 4) | pd)
    }
}

/// Predicate logical group (AND/ORR/EOR/... + MOV/MOVS/NOT/NOTS aliases).
fn enc_pred_logical(insn: &Instruction) -> Result<u32, EncodeError> {
    let m = insn.mnemonic();
    let pd = p(insn, 0)?;
    let (key, pg, pn, pm) = match m {
        Mnemonic::Mov if insn.op_count() == 2 => {
            let pn = p(insn, 1)?;
            (0b1000u32, pn, pn, pn)
        }
        Mnemonic::Movs if insn.op_count() == 2 => {
            let pn = p(insn, 1)?;
            (0b1100, pn, pn, pn)
        }
        Mnemonic::Mov => {
            let pg = p(insn, 1)?;
            let pn = p(insn, 2)?;
            (0b0000, pg, pn, pn)
        }
        Mnemonic::Movs => {
            let pg = p(insn, 1)?;
            let pn = p(insn, 2)?;
            (0b0100, pg, pn, pn)
        }
        Mnemonic::Not => {
            let pg = p(insn, 1)?;
            let pn = p(insn, 2)?;
            (0b0010, pg, pn, pg)
        }
        Mnemonic::Nots => {
            let pg = p(insn, 1)?;
            let pn = p(insn, 2)?;
            (0b0110, pg, pn, pg)
        }
        _ => {
            let pg = p(insn, 1)?;
            let pn = p(insn, 2)?;
            let pm = p(insn, 3)?;
            (pred_logical_key(m)?, pg, pn, pm)
        }
    };
    let op = (key >> 3) & 1;
    let s = (key >> 2) & 1;
    let o2 = (key >> 1) & 1;
    let o3 = key & 1;
    Ok(fld(0b00100101, 24) | fld(op, 23) | fld(s, 22) | fld(pm, 16) | fld(0b01, 14) | fld(pg, 10)
        | fld(o2, 9)
        | fld(pn, 5)
        | fld(o3, 4)
        | pd)
}

/// `(op,S,o2,o3)` key for a predicate-logical mnemonic.
fn pred_logical_key(m: Mnemonic) -> Result<u32, EncodeError> {
    Ok(match m {
        Mnemonic::And => 0b0000,
        Mnemonic::Bic => 0b0001,
        Mnemonic::Eor => 0b0010,
        Mnemonic::Ands => 0b0100,
        Mnemonic::Bics => 0b0101,
        Mnemonic::Eors => 0b0110,
        Mnemonic::Orr => 0b1000,
        Mnemonic::Orn => 0b1001,
        Mnemonic::Nor => 0b1010,
        Mnemonic::Nand => 0b1011,
        Mnemonic::Orrs => 0b1100,
        Mnemonic::Orns => 0b1101,
        Mnemonic::Nors => 0b1110,
        Mnemonic::Nands => 0b1111,
        _ => return Err(EncodeError::InvalidOperand),
    })
}

/// BRKP{A,B}{S}.
fn enc_brkp(insn: &Instruction) -> Result<u32, EncodeError> {
    let (b, s) = match insn.mnemonic() {
        Mnemonic::Brkpa => (0, 0),
        Mnemonic::Brkpas => (0, 1),
        Mnemonic::Brkpb => (1, 0),
        _ => (1, 1),
    };
    let pd = p(insn, 0)?;
    let pg = p(insn, 1)?;
    let pn = p(insn, 2)?;
    let pm = p(insn, 3)?;
    Ok(fld(0b00100101, 24) | fld(s, 22) | fld(pm, 16) | fld(0b11, 14) | fld(pg, 10) | fld(pn, 5)
        | fld(b, 4)
        | pd)
}

/// BRKA/BRKAS/BRKB/BRKBS.
fn enc_brk(insn: &Instruction) -> Result<u32, EncodeError> {
    let (bb, s) = match insn.mnemonic() {
        Mnemonic::Brka => (0, 0),
        Mnemonic::Brkas => (0, 1),
        Mnemonic::Brkb => (1, 0),
        _ => (1, 1),
    };
    let pd = p(insn, 0)?;
    let pg = p(insn, 1)?;
    let pn = p(insn, 2)?;
    let merging = matches!(pred_qual(insn, 1), Some(PredQual::Merging));
    let mbit = if s == 0 && merging { 1 } else { 0 };
    Ok(fld(0b00100101, 24) | fld(bb, 23) | fld(s, 22) | fld(0b010000, 16) | fld(0b01, 14) | fld(pg, 10)
        | fld(pn, 5)
        | fld(mbit, 4)
        | pd)
}

/// BRKN/BRKNS.
fn enc_brkn(insn: &Instruction) -> Result<u32, EncodeError> {
    let s = if matches!(insn.mnemonic(), Mnemonic::Brkns) { 1 } else { 0 };
    let pd = p(insn, 0)?;
    let pg = p(insn, 1)?;
    let pn = p(insn, 2)?;
    Ok(fld(0b00100101, 24) | fld(s, 22) | fld(0b011000, 16) | fld(0b01, 14) | fld(pg, 10) | fld(pn, 5)
        | pd)
}

/// Predicate-indexed DUP.
fn enc_psel(insn: &Instruction) -> Result<u32, EncodeError> {
    // `PSEL <Pd>, <Pn>, <Pm>.<T>[<Wv>{, #imm}]`. Pd=<3:0>, Pn=<13:10>, Pm=<8:5>,
    // Wv=12+<17:16>, element/index in the `tszh:tszl` field `<23:22>:<20:18>`.
    let pd = p(insn, 0)?;
    let pn = p(insn, 1)?;
    let (pm, wv, arr, imm) = match insn.op(2) {
        Operand::IndexedElement {
            reg, index, imm, arr: Some(a), ..
        } => (reg.number() as u32, index.number() as u32, a, imm as u32),
        _ => return Err(EncodeError::InvalidOperand),
    };
    // esize from the arrangement; the `tsz` field = (index << (e+1)) | (1 << e).
    let e: u32 = match arr {
        VA::Sb => 0,
        VA::Sh => 1,
        VA::Ss => 2,
        VA::Sd => 3,
        _ => return Err(EncodeError::InvalidOperand),
    };
    let tsz = (imm << (e + 1)) | (1 << e); // 5-bit `<23:22>:<20:18>`
    if tsz > 0x1f {
        return Err(EncodeError::InvalidImmediate);
    }
    let tszh = (tsz >> 3) & 3; // <23:22>
    let tszl = tsz & 7; // <20:18>
    let wvf = wv.wrapping_sub(12) & 3; // <17:16>
    Ok(fld(0b00100101, 24) | fld(tszh, 22) | fld(1, 21) | fld(tszl, 18) | fld(wvf, 16)
        | fld(0b01, 14)
        | fld(pn, 10)
        | fld(pm, 5)
        | pd)
}

/// WHILE<cc>.
fn enc_while(insn: &Instruction) -> Result<u32, EncodeError> {
    let size = esize(insn, 0)?;
    let pd = p(insn, 0)?;
    let rn = g(insn, 1)?;
    let rm = g(insn, 2)?;
    let sf = match insn.op(1) {
        Operand::Reg { reg, .. } if reg.width_bits() == 64 => 1,
        _ => 0,
    };
    let (u, lt, eq) = match insn.mnemonic() {
        Mnemonic::Whilelt => (0, 1, 0),
        Mnemonic::Whilele => (0, 1, 1),
        Mnemonic::Whilelo => (1, 1, 0),
        Mnemonic::Whilels => (1, 1, 1),
        Mnemonic::Whilege => (0, 0, 0),
        Mnemonic::Whilegt => (0, 0, 1),
        Mnemonic::Whilehi => (1, 0, 1),
        Mnemonic::Whilehs => (1, 0, 0),
        _ => return Err(EncodeError::InvalidOperand),
    };
    Ok(fld(0b00100101, 24) | fld(size, 22) | fld(1, 21) | fld(rm, 16) | fld(sf, 12) | fld(u, 11)
        | fld(lt, 10)
        | fld(rn, 5)
        | fld(eq, 4)
        | pd)
}

/// `(U, lt, eq)` condition triple for a `WHILE<cc>` mnemonic.
fn while_cond(m: Mnemonic) -> Result<(u32, u32, u32), EncodeError> {
    Ok(match m {
        Mnemonic::Whilelt => (0, 1, 0),
        Mnemonic::Whilele => (0, 1, 1),
        Mnemonic::Whilelo => (1, 1, 0),
        Mnemonic::Whilels => (1, 1, 1),
        Mnemonic::Whilege => (0, 0, 0),
        Mnemonic::Whilegt => (0, 0, 1),
        Mnemonic::Whilehi => (1, 0, 1),
        Mnemonic::Whilehs => (1, 0, 0),
        _ => return Err(EncodeError::InvalidOperand),
    })
}

/// SVE2.1 `WHILE<cc> {Pd.T, Pd+1.T}, Xn, Xm` (predicate pair). Layout:
/// `00100101 size 1 Rm 010 1 U lt Rn 1 Pd<3:1> eq`, with `<8>=1` fixed.
fn enc_while_pair(insn: &Instruction) -> Result<u32, EncodeError> {
    // Result pair `{P(2k).T, P(2k+1).T}` -> k in <3:1>; size from the arrangement.
    let (first, a) = match insn.op(0) {
        Operand::MultiReg { regs, arr: Some(a), count: 2, .. } => (regs[0].number() as u32, a),
        _ => return Err(EncodeError::InvalidOperand),
    };
    if first & 1 != 0 {
        return Err(EncodeError::InvalidOperand);
    }
    let size = arr_size(a)?;
    let k = first >> 1; // 0..=7
    let rn = g(insn, 1)?;
    let rm = g(insn, 2)?;
    let (u, lt, eq) = while_cond(insn.mnemonic())?;
    Ok(fld(0b00100101, 24)
        | fld(size, 22)
        | fld(1, 21)
        | fld(rm, 16)
        | fld(0b010, 13)
        | fld(1, 12)
        | fld(u, 11)
        | fld(lt, 10)
        | fld(rn, 5)
        | fld(1, 4) // fixed marker
        | fld(k, 1)
        | eq)
}

/// SVE2.1 `WHILE<cc> PNd.T, Xn, Xm, VLx{2,4}` (predicate as counter). Layout:
/// `00100101 size 1 Rm 01 vl 0 U lt Rn 1 eq PN<2:0>`, with `<8>=1` fixed.
fn enc_while_pn(insn: &Instruction) -> Result<u32, EncodeError> {
    let (pn, a) = match insn.op(0) {
        Operand::PredCounter { reg, arr: Some(a), .. } => (reg.number() as u32, a),
        _ => return Err(EncodeError::InvalidOperand),
    };
    if !(8..=15).contains(&pn) {
        return Err(EncodeError::InvalidOperand);
    }
    let size = arr_size(a)?;
    let rn = g(insn, 1)?;
    let rm = g(insn, 2)?;
    let vl = match insn.op(3) {
        Operand::VlMul(2) => 0,
        Operand::VlMul(4) => 1,
        _ => return Err(EncodeError::InvalidOperand),
    };
    let (u, lt, eq) = while_cond(insn.mnemonic())?;
    Ok(fld(0b00100101, 24)
        | fld(size, 22)
        | fld(1, 21)
        | fld(rm, 16)
        | fld(0b01, 14)
        | fld(vl, 13)
        | fld(u, 11)
        | fld(lt, 10)
        | fld(rn, 5)
        | fld(1, 4) // fixed marker
        | fld(eq, 3)
        | (pn - 8))
}
