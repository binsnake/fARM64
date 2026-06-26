//! Inverse of [`crate::decode::sve::sve_int`] — SVE/SVE2 integer family.

use super::{
    arr_size, enc_left_shift, enc_right_shift, esize, esize_of, fld, g, imm, lane, p, pred_qual,
    read_pattern_mul, reg, sfp, simm, z,
};
use crate::encode::EncodeError;
use crate::enums::VectorArrangement as VA;
use crate::instruction::Instruction;
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::{Operand, PredQual};
use crate::register::RegClass;

use Code::*;

/// `true` for every integer-family SVE [`Code`].
pub(super) fn is_int(code: Code) -> bool {
    matches!(
        code,
        // unpredicated arith / logical / mul
        SveAddZzz | SveSubZzz | SveSqaddZzz | SveUqaddZzz | SveSqsubZzz | SveUqsubZzz
            | SveAndZzz | SveOrrZzz | SveEorZzz | SveBicZzz | SveMovZzz
            | SveMulZzz | SvePmulZzz | SveSmulhZzz | SveUmulhZzz
            | SveXar | SveEor3 | SveBcax | SveBsl | SveBsl1n | SveBsl2n | SveNbsl
        // logical / dup immediate
            | SveOrrZi | SveEorZi | SveAndZi | SveDupmZi
            | SveAddZi | SveSubZi | SveSubrZi | SveSqaddZi | SveUqaddZi | SveSqsubZi | SveUqsubZi
            | SveSmaxZi | SveUmaxZi | SveSminZi | SveUminZi | SveMulZi | SveDupImm
            | SveCpyImmMerge | SveCpyImmZero
        // dup/cpy/insr/sel
            | SveDupScalar | SveInsrScalar | SveInsrVec | SveDupIdx | SveCpyScalar | SveCpyVec
            | SveSelZpzz | SveRbitZpz
        // index / addvl / rdvl (+ SME streaming addsvl / addspl / rdsvl)
            | SveIndexImmImm | SveIndexRi | SveIndexIr | SveIndexRr | SveAddvl | SveAddpl | SveRdvl
            | SveAddsvl | SveAddspl | SveRdsvl
        // shift unpred
            | SveAsrWide | SveLsrWide | SveLslWide | SveAsrZi | SveLsrZi | SveLslZi
        // adr / movprfx
            | SveAdrSxtw | SveAdrUxtw | SveAdrSameScaled | SveMovprfxZz | SveMovprfxZpz
        // inc/dec
            | SveIncDecVector | SveIncDecScalar | SveSqIncDecScalarSx | SveCntElem
            | SveIncDecPVector | SveIncDecPScalar | SveSqIncDecPScalarSx | SveCntp
        // predicated binary / unary / reductions / mla
            | SveAddZpzz | SveSubZpzz | SveSubrZpzz | SveSmaxZpzz | SveUmaxZpzz | SveSminZpzz
            | SveUminZpzz | SveSabdZpzz | SveUabdZpzz | SveMulZpzz | SveSmulhZpzz | SveUmulhZpzz
            | SveSdivZpzz | SveUdivZpzz | SveSdivrZpzz | SveUdivrZpzz | SveOrrZpzz | SveEorZpzz
            | SveAndZpzz | SveBicZpzz
            | SveSaddv | SveUaddv | SveSmaxv | SveUminv | SveSminv | SveUmaxv | SveOrv | SveEorv
            | SveAndv
            | SveMlaZpzzz | SveMlsZpzzz | SveMadZpzzz | SveMsbZpzzz
            | SveAsrZpzi | SveLsrZpzi | SveLslZpzi | SveAsrdZpzi
            | SveAsrZpzz | SveLsrZpzz | SveLslZpzz | SveAsrrZpzz | SveLsrrZpzz | SveLslrZpzz
            | SveAsrWidePred | SveLsrWidePred | SveLslWidePred
            | SveSxtbZpz | SveUxtbZpz | SveSxthZpz | SveUxthZpz | SveSxtwZpz | SveUxtwZpz
            | SveAbsZpz | SveNegZpz | SveClsZpz | SveClzZpz | SveCntZpz | SveCnotZpz | SveNotZpz
        // compares
            | SveCmpZi | SveCmpZz | SveCmpZw
        // sve2 dot / mul-add / widening (0x44 / 0x45)
            | SveSdot | SveUdot | SveSdotIdx | SveUdotIdx | SveDotMixed
            | SveSdotH | SveUdotH | SveSdotHIdx | SveUdotHIdx
            | SveZipqUzpq | SveTblq
        // sve2.1 misc + quadword reductions to vector
            | SveExpand | SveDupq | SveExtq | SvePmov
            | SveAddqv | SveSmaxqv | SveUmaxqv | SveSminqv | SveUminqv
            | SveOrqv | SveEorqv | SveAndqv
            | SveAddptPred | SveSubptPred | SveAddptUnpred | SveSubptUnpred
            | SveCdot | SveCdotIdx | SveCmla | SveCmlaIdx | SveSqrdcmlah | SveSqrdcmlahIdx
            | SveMlaIdx | SveMlsIdx | SveMulIdx
            | SveSqrdmlah | SveSqrdmlsh | SveSqrdmlahIdx | SveSqrdmlshIdx
            | SveSqdmulhIdx | SveSqrdmulhIdx
            | SveMlaLong | SveMlaLongIdx | SveMulLong | SveMulLongIdx | SvePmulLong
            | SveSqdmlalLong | SveSqdmlalLongBt | SveSqdmlalLongIdx | SveSqdmulLong | SveSqdmulLongIdx
            | SveSclamp | SveUclamp
            | SveHalvingZpzz | SveSatRoundZpzz | SvePairZpzz | SveAdalp | SveSatUnaryZpz | SveRecipEst
            | SveMatmulInt | SveBitPerm | SveEorInterleave | SveHistcnt | SveHistseg
            | SveAesMc | SveAesZz | SveSm4e | SveSm4ekey | SveRax1 | SveMatch
            | SveShiftAccum | SveShiftLongImm | SveShiftInsert | SveShiftNarrow | SveExtractNarrow
            | SveAddLong | SveAbdLong | SveAddWide | SveAddHighNarrow | SveAddLongBt | SveAbaLong
            | SveAddCarryLong | SveCadd | SveSqcadd | SveAbaSame
            | SveSabal | SveUabal
        // i3: SVE2.3 quadword pair add / 2-way dot, SVE2.2 sqabs/sqneg zeroing,
        // FEAT_CPA madpt/mlapt/subp
            | SveAddqp | SveAddsubp | SveSdotHb | SveUdotHb | SveSqabsZ | SveSqnegZ
            | SveMadpt | SveMlapt | SveSubpPred
    )
}

/// Encode an integer-family SVE instruction. Returns `Ok(None)` when `code`
/// does not belong to this family.
pub(super) fn enc(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
    let w = match code {
        // ---- unpredicated arithmetic (0x04, <21>=1, <15:13>=000) ----
        SveAddZzz | SveSubZzz | SveSqaddZzz | SveUqaddZzz | SveSqsubZzz | SveUqsubZzz => {
            let op = match code {
                SveAddZzz => 0b000,
                SveSubZzz => 0b001,
                SveSqaddZzz => 0b100,
                SveUqaddZzz => 0b101,
                SveSqsubZzz => 0b110,
                _ => 0b111,
            };
            arith_zzz(insn, op)?
        }
        // ---- unpredicated logical (0x04, <15:13>=001, <12:10>=100) ----
        SveAndZzz | SveOrrZzz | SveEorZzz | SveBicZzz => {
            let op = match code {
                SveAndZzz => 0b00,
                SveOrrZzz => 0b01,
                SveEorZzz => 0b10,
                _ => 0b11,
            };
            // .d arrangement; Zd, Zn, Zm.
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            base04(1, 0b001) | fld(op, 22) | fld(0b100, 10) | fld(zm, 16) | fld(zn, 5) | zd
        }
        SveMovZzz => {
            // ORR Zd.D, Zn.D, Zn.D alias.
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            base04(1, 0b001) | fld(0b01, 22) | fld(0b100, 10) | fld(zn, 16) | fld(zn, 5) | zd
        }
        // ---- unpredicated multiply (0x04, <15:13>=011) ----
        SveMulZzz | SvePmulZzz | SveSmulhZzz | SveUmulhZzz => {
            let (op, mnem) = match code {
                SvePmulZzz => (0b001u32, None),
                SveSmulhZzz => (0b010, None),
                SveUmulhZzz => (0b011, None),
                _ => match insn.mnemonic() {
                    Mnemonic::Sqdmulh => (0b100, Some(())),
                    Mnemonic::Sqrdmulh => (0b101, Some(())),
                    _ => (0b000, None),
                },
            };
            let _ = mnem;
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            base04(1, 0b011) | fld(size, 22) | fld(op, 10) | fld(zm, 16) | fld(zn, 5) | zd
        }
        // ---- SVE2 bitwise ternary (0x04, <15:11>=00111) ----
        SveEor3 | SveBcax | SveBsl | SveBsl1n | SveBsl2n | SveNbsl => {
            let (opc, o2) = match code {
                SveEor3 => (0b00, 0),
                SveBcax => (0b01, 0),
                SveBsl => (0b00, 1),
                SveBsl1n => (0b01, 1),
                SveBsl2n => (0b10, 1),
                _ => (0b11, 1),
            };
            let zdn = z(insn, 0)?;
            let zm = z(insn, 2)?;
            let zk = z(insn, 3)?;
            base04(1, 0b001) | fld(opc, 22) | fld(0b00111, 11) | fld(o2, 10) | fld(zm, 16) | fld(zk, 5)
                | zdn
        }
        // ---- SVE2 XAR (0x04, <15:10>=001101) ----
        SveXar => {
            let a = arr_of(insn, 0)?;
            let amount = imm(insn, 3)? as u32;
            let (tsz, imm3) = enc_right_shift(a, amount)?;
            let zdn = z(insn, 0)?;
            let zm = z(insn, 2)?;
            // base group 0x04, <15:10>=001101; tszh=<23:22>,tszl=<20:19>,imm3=<18:16>.
            fld(0b00000100, 24)
                | fld(tsz >> 2, 22)
                | fld(1, 21)
                | fld(tsz & 3, 19)
                | fld(imm3, 16)
                | fld(0b001101, 10)
                | fld(zm, 5)
                | zdn
        }
        // ---- logical / dup immediate (0x05, <21:19>=000) ----
        SveOrrZi | SveEorZi | SveAndZi => {
            let opc = match code {
                SveOrrZi => 0b00,
                SveEorZi => 0b01,
                _ => 0b10,
            };
            let a = arr_of(insn, 0)?;
            let zdn = z(insn, 0)?;
            let val = replicate_element(imm(insn, 2)?, a)?;
            let imm13 = enc_sve_bitmask(val)?;
            fld(0b00000101, 24) | fld(opc, 22) | fld(imm13, 5) | zdn
        }
        SveDupmZi => {
            let a = arr_of(insn, 0)?;
            let zd = z(insn, 0)?;
            let val = replicate_element(imm(insn, 1)?, a)?;
            let imm13 = enc_sve_bitmask(val)?;
            fld(0b00000101, 24) | fld(0b11, 22) | fld(imm13, 5) | zd
        }
        // ---- CPY immediate (0x05, <21:20>=01, <15>=0) ----
        SveCpyImmMerge | SveCpyImmZero => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let merging = matches!(code, SveCpyImmMerge);
            let (imm8, sh) = read_dup_imm(insn, 2)?;
            // `LSL #8` (`sh == 1`) is reserved for the `.b` element size — the
            // decoder leaves it UNDEFINED, so the encoder must refuse it too.
            if size == 0b00 && sh == 1 {
                return Err(EncodeError::InvalidImmediate);
            }
            fld(0b00000101, 24)
                | fld(size, 22)
                | fld(0b01, 20)
                | fld(pg, 16)
                | fld(if merging { 1 } else { 0 }, 14)
                | fld(sh, 13)
                | fld(imm8, 5)
                | zd
        }
        // ---- DUP scalar / INSR / DUP indexed (0x05, <15:13>=001) ----
        SveDupScalar => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let rn = g(insn, 1)?;
            base05_001(size, 0b00000) | fld(rn, 5) | zd
        }
        SveInsrScalar => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let rm = g(insn, 1)?;
            base05_001(size, 0b00100) | fld(rm, 5) | zd
        }
        SveInsrVec => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let vm = sfp(insn, 1)?;
            base05_001(size, 0b10100) | fld(vm, 5) | zd
        }
        SveDupIdx => enc_dup_idx(insn)?,
        // ---- CPY scalar / vector (0x05, <15:13>=101/100) ----
        SveCpyScalar => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let rn = g(insn, 2)?;
            fld(0b00000101, 24) | fld(size, 22) | fld(1, 21) | fld(0b01000, 16) | fld(0b101, 13)
                | fld(pg, 10)
                | fld(rn, 5)
                | zd
        }
        SveCpyVec => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let vn = sfp(insn, 2)?;
            fld(0b00000101, 24) | fld(size, 22) | fld(1, 21) | fld(0b00000, 16) | fld(0b100, 13)
                | fld(pg, 10)
                | fld(vn, 5)
                | zd
        }
        SveRbitZpz => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            // `<15:13>` is 100 for the merging (`/m`) form, 101 for the
            // FEAT_SVE2p1 zeroing (`/z`) form.
            let sel = if matches!(pred_qual(insn, 1), Some(PredQual::Zeroing)) { 0b101 } else { 0b100 };
            fld(0b00000101, 24) | fld(size, 22) | fld(1, 21) | fld(0b00111, 16) | fld(sel, 13)
                | fld(pg, 10)
                | fld(zn, 5)
                | zd
        }
        // ---- SEL (0x05, <15:14>=11, <21>=1) ----
        SveSelZpzz => enc_sel(insn)?,
        // ---- INDEX / ADDVL / ADDPL / RDVL (0x04, <15:13>=010) ----
        SveIndexImmImm => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let i1 = (simm(insn, 1)? as u32) & 0x1f;
            let i2 = (simm(insn, 2)? as u32) & 0x1f;
            base04(1, 0b010) | fld(size, 22) | fld(0b000, 10) | fld(i2, 16) | fld(i1, 5) | zd
        }
        SveIndexRi => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let rn = g(insn, 1)?;
            let i2 = (simm(insn, 2)? as u32) & 0x1f;
            base04(1, 0b010) | fld(size, 22) | fld(0b001, 10) | fld(i2, 16) | fld(rn, 5) | zd
        }
        SveIndexIr => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let i1 = (simm(insn, 1)? as u32) & 0x1f;
            let rm = g(insn, 2)?;
            base04(1, 0b010) | fld(size, 22) | fld(0b010, 10) | fld(rm, 16) | fld(i1, 5) | zd
        }
        SveIndexRr => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let rn = g(insn, 1)?;
            let rm = g(insn, 2)?;
            base04(1, 0b010) | fld(size, 22) | fld(0b011, 10) | fld(rm, 16) | fld(rn, 5) | zd
        }
        SveAddvl | SveAddpl => {
            let op22 = if matches!(code, SveAddvl) { 0b00 } else { 0b01 };
            let rd = reg(insn, 0)?;
            let rn = reg(insn, 1)?;
            let imm6 = (simm(insn, 2)? as u32) & 0x3f;
            base04(1, 0b010) | fld(op22, 22) | fld(0b100, 10) | fld(rn, 16) | fld(imm6, 5) | rd
        }
        SveRdvl => {
            let rd = reg(insn, 0)?;
            let imm6 = (simm(insn, 1)? as u32) & 0x3f;
            base04(1, 0b010) | fld(0b10, 22) | fld(0b100, 10) | fld(0b11111, 16) | fld(imm6, 5) | rd
        }
        // SME streaming analogues: word<11>=1 (the `0b110` in the bits<12:10> slot).
        SveAddsvl | SveAddspl => {
            let op22 = if matches!(code, SveAddsvl) { 0b00 } else { 0b01 };
            let rd = reg(insn, 0)?;
            let rn = reg(insn, 1)?;
            let imm6 = (simm(insn, 2)? as u32) & 0x3f;
            base04(1, 0b010) | fld(op22, 22) | fld(0b110, 10) | fld(rn, 16) | fld(imm6, 5) | rd
        }
        SveRdsvl => {
            let rd = reg(insn, 0)?;
            let imm6 = (simm(insn, 1)? as u32) & 0x3f;
            base04(1, 0b010) | fld(0b10, 22) | fld(0b110, 10) | fld(0b11111, 16) | fld(imm6, 5) | rd
        }
        // ---- shift unpred (0x04, <15:13>=100) ----
        SveAsrWide | SveLsrWide | SveLslWide => {
            let op = match code {
                SveAsrWide => 0b000,
                SveLsrWide => 0b001,
                _ => 0b011,
            };
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            base04(1, 0b100) | fld(size, 22) | fld(op, 10) | fld(zm, 16) | fld(zn, 5) | zd
        }
        SveAsrZi | SveLsrZi | SveLslZi => {
            let op = match code {
                SveAsrZi => 0b100,
                SveLsrZi => 0b101,
                _ => 0b111,
            };
            let a = arr_of(insn, 0)?;
            let amount = imm(insn, 2)? as u32;
            let (tsz, imm3) = if matches!(code, SveLslZi) {
                enc_left_shift(a, amount)?
            } else {
                enc_right_shift(a, amount)?
            };
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            fld(0b00000100, 24)
                | fld(tsz >> 2, 22)
                | fld(1, 21)
                | fld(tsz & 3, 19)
                | fld(imm3, 16)
                | fld(0b100, 13)
                | fld(op, 10)
                | fld(zn, 5)
                | zd
        }
        // ---- ADR / MOVPRFX (0x04, <15:13>=101) ----
        SveAdrSxtw | SveAdrUxtw | SveAdrSameScaled => enc_adr_vec(insn, code)?,
        SveMovprfxZz => {
            let zd = reg(insn, 0)?;
            let zn = reg(insn, 1)?;
            base04(1, 0b101) | fld(0b00, 22) | fld(0b111, 10) | fld(zn, 5) | zd
        }
        SveMovprfxZpz => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let merging = matches!(pred_qual(insn, 1), Some(PredQual::Merging));
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            // 0x04 reductions slot: <15:13>=001, <20:16>=1000M.
            base04(0, 0b001) | fld(size, 22) | fld(0b10000 | u32::from(merging), 16) | fld(pg, 10)
                | fld(zn, 5)
                | zd
        }
        // ---- INC/DEC vector (0x04, <15:13>=110) ----
        SveIncDecVector => enc_incdec_vec(insn)?,
        // ---- CNT / INC/DEC scalar (0x04, <15:13>=111) ----
        SveCntElem | SveIncDecScalar | SveSqIncDecScalarSx => enc_cnt_incdec_scalar(insn, code)?,
        // ---- INC/DEC by predicate (0x25, <15:13>=100) ----
        SveIncDecPVector | SveIncDecPScalar | SveSqIncDecPScalarSx => enc_incdec_pred(insn, code)?,
        SveCntp => {
            let size = esize(insn, 2)?;
            let rd = g(insn, 0)?;
            let pg = p(insn, 1)?;
            let pn = p(insn, 2)?;
            fld(0b00100101, 24) | fld(size, 22) | fld(0b10, 20) | fld(0b10, 14) | fld(pg, 10)
                | fld(pn, 5)
                | rd
        }
        // ---- predicated binary (0x04, <15:13>=000) ----
        SveAddZpzz | SveSubZpzz | SveSubrZpzz | SveSmaxZpzz | SveUmaxZpzz | SveSminZpzz
        | SveUminZpzz | SveSabdZpzz | SveUabdZpzz | SveMulZpzz | SveSmulhZpzz | SveUmulhZpzz
        | SveSdivZpzz | SveUdivZpzz | SveSdivrZpzz | SveUdivrZpzz | SveOrrZpzz | SveEorZpzz
        | SveAndZpzz | SveBicZpzz => {
            let opc = pred_binary_opc(code);
            let size = esize(insn, 0)?;
            let (zdn, pg, zm) = read_pred_binary(insn)?;
            base04(0, 0b000) | fld(size, 22) | fld(opc, 16) | fld(pg, 10) | fld(zm, 5) | zdn
        }
        // ---- ADDPT/SUBPT predicated (FEAT_CPA, 0x04, <15:13>=000, .d only) ----
        SveAddptPred | SveSubptPred => {
            let opc = if matches!(code, SveAddptPred) { 0b00100 } else { 0b00101 };
            let (zdn, pg, zm) = read_pred_binary(insn)?;
            base04(0, 0b000) | fld(0b11, 22) | fld(opc, 16) | fld(pg, 10) | fld(zm, 5) | zdn
        }
        // ---- ADDPT/SUBPT unpredicated (FEAT_CPA, 0x04, <15:13>=000, .d only) ----
        SveAddptUnpred | SveSubptUnpred => {
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            let op = if matches!(code, SveAddptUnpred) { 0b010 } else { 0b011 };
            base04(1, 0b000) | fld(0b11, 22) | fld(op, 10) | fld(zm, 16) | fld(zn, 5) | zd
        }
        // ---- reductions (0x04, <15:13>=001) ----
        SveSaddv | SveUaddv => {
            let opc = if matches!(code, SveSaddv) { 0b00000 } else { 0b00001 };
            let size = esize(insn, 2)?;
            let dd = sfp(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            base04(0, 0b001) | fld(size, 22) | fld(opc, 16) | fld(pg, 10) | fld(zn, 5) | dd
        }
        SveSmaxv | SveUmaxv | SveSminv | SveUminv | SveOrv | SveEorv | SveAndv => {
            let opc = match code {
                SveSmaxv => 0b01000,
                SveUmaxv => 0b01001,
                SveSminv => 0b01010,
                SveUminv => 0b01011,
                SveOrv => 0b11000,
                SveEorv => 0b11001,
                _ => 0b11010,
            };
            let size = esize(insn, 2)?;
            let dd = sfp(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            base04(0, 0b001) | fld(size, 22) | fld(opc, 16) | fld(pg, 10) | fld(zn, 5) | dd
        }
        // ---- SVE2.1 quadword reductions to a NEON `V` register (0x04) ----
        SveAddqv | SveSmaxqv | SveUmaxqv | SveSminqv | SveUminqv | SveOrqv | SveEorqv | SveAndqv => {
            let opc = match code {
                SveAddqv => 0b00101,
                SveSmaxqv => 0b01100,
                SveUmaxqv => 0b01101,
                SveSminqv => 0b01110,
                SveUminqv => 0b01111,
                SveOrqv => 0b11100,
                SveEorqv => 0b11101,
                _ => 0b11110,
            };
            let size = esize(insn, 2)?; // element size from the source `Zn`
            let vd = reg(insn, 0)?; // destination `Vd`
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            base04(0, 0b001) | fld(size, 22) | fld(opc, 16) | fld(pg, 10) | fld(zn, 5) | vd
        }
        // ---- SVE2.1 misc (0x05): EXPAND / DUPQ / EXTQ / PMOV ----
        SveExpand => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            fld(0b00000101, 24) | fld(size, 22) | fld(1, 21) | fld(0b10001, 16) | fld(0b100, 13)
                | fld(pg, 10)
                | fld(zn, 5)
                | zd
        }
        SveDupq => {
            let zd = z(insn, 0)?;
            let size = esize(insn, 0)?; // 0=.b .. 3=.d
            let zn = z(insn, 1)?;
            let idx = lane(insn, 1)?;
            let tsz = (idx << (size + 1)) | (1u32 << size);
            if tsz > 0x1f {
                return Err(EncodeError::InvalidImmediate);
            }
            fld(0b00000101, 24) | fld(1, 21) | fld(tsz, 16) | fld(0b001, 13) | fld(0b001, 10)
                | fld(zn, 5)
                | zd
        }
        SveExtq => {
            let zd = z(insn, 0)?;
            let zm = z(insn, 2)?;
            let imm4 = (imm(insn, 3)? as u32) & 0xf;
            fld(0b00000101, 24) | fld(0b01, 22) | fld(1, 21) | fld(imm4, 16) | fld(0b001, 13)
                | fld(0b001, 10)
                | fld(zm, 5)
                | zd
        }
        SvePmov => {
            // bits<4:0> = destination (Pd or Zd); bits<9:5> = source (Zn or Pn).
            let d4_0 = reg(insn, 0)?;
            let s9_5 = reg(insn, 1)?;
            // The predicate carries the arrangement; the vector carries the lane.
            let (dbit, a, lane) = match insn.op(0) {
                Operand::Reg { reg, arr: Some(a), .. } if reg.class() == RegClass::Predicate => {
                    (0u32, a, pmov_lane(insn, 1)) // P <- Z
                }
                _ => {
                    let a = match insn.op(1) {
                        Operand::Reg { arr: Some(a), .. } => a,
                        _ => return Err(EncodeError::InvalidOperand),
                    };
                    (1u32, a, pmov_lane(insn, 0)) // Z <- P
                }
            };
            let pos = arr_size(a)?;
            let t = (1u32 << pos) | (lane & ((1u32 << pos) - 1));
            fld(0b00000101, 24)
                | fld((t >> 3) & 1, 23)
                | fld((t >> 2) & 1, 22)
                | fld(1, 21)
                | fld(1, 19)
                | fld((t >> 1) & 1, 18)
                | fld(t & 1, 17)
                | fld(dbit, 16)
                | fld(0b001, 13)
                | fld(0b110, 10)
                | fld(s9_5, 5)
                | d4_0
        }
        // ---- MLA / MLS / MAD / MSB (0x04) ----
        SveMlaZpzzz | SveMlsZpzzz => {
            let sel = if matches!(code, SveMlsZpzzz) { 0b011 } else { 0b010 };
            let size = esize(insn, 0)?;
            let zda = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            let zm = z(insn, 3)?;
            base04(0, sel) | fld(size, 22) | fld(zm, 16) | fld(pg, 10) | fld(zn, 5) | zda
        }
        SveMadZpzzz | SveMsbZpzzz => {
            let sel = if matches!(code, SveMsbZpzzz) { 0b111 } else { 0b110 };
            let size = esize(insn, 0)?;
            let zdn = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zm = z(insn, 2)?;
            let za = z(insn, 3)?;
            base04(0, sel) | fld(size, 22) | fld(zm, 16) | fld(pg, 10) | fld(za, 5) | zdn
        }
        // ---- predicated shift immediate (0x04, <15:13>=100) ----
        SveAsrZpzi | SveLsrZpzi | SveLslZpzi | SveAsrdZpzi => enc_shift_pred_imm(insn, code)?,
        // ---- predicated shift by vector (0x04, <15:13>=100) ----
        SveAsrZpzz | SveLsrZpzz | SveLslZpzz | SveAsrrZpzz | SveLsrrZpzz | SveLslrZpzz => {
            let opc = match code {
                SveAsrZpzz => 0b10000,
                SveLsrZpzz => 0b10001,
                SveLslZpzz => 0b10011,
                SveAsrrZpzz => 0b10100,
                SveLsrrZpzz => 0b10101,
                _ => 0b10111,
            };
            let size = esize(insn, 0)?;
            let (zdn, pg, zm) = read_pred_binary(insn)?;
            base04(0, 0b100) | fld(size, 22) | fld(opc, 16) | fld(pg, 10) | fld(zm, 5) | zdn
        }
        // ---- predicated wide shift (0x04, <15:13>=100) ----
        SveAsrWidePred | SveLsrWidePred | SveLslWidePred => {
            let opc = match code {
                SveAsrWidePred => 0b11000,
                SveLsrWidePred => 0b11001,
                _ => 0b11011,
            };
            let size = esize(insn, 0)?;
            let zdn = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zm = z(insn, 3)?;
            base04(0, 0b100) | fld(size, 22) | fld(opc, 16) | fld(pg, 10) | fld(zm, 5) | zdn
        }
        // ---- unary predicated (0x04, <15:13>=101) ----
        SveSxtbZpz | SveUxtbZpz | SveSxthZpz | SveUxthZpz | SveSxtwZpz | SveUxtwZpz | SveAbsZpz
        | SveNegZpz | SveClsZpz | SveClzZpz | SveCntZpz | SveCnotZpz | SveNotZpz => {
            let opc = match code {
                SveSxtbZpz => 0b10000,
                SveUxtbZpz => 0b10001,
                SveSxthZpz => 0b10010,
                SveUxthZpz => 0b10011,
                SveSxtwZpz => 0b10100,
                SveUxtwZpz => 0b10101,
                SveAbsZpz => 0b10110,
                SveNegZpz => 0b10111,
                SveClsZpz => 0b11000,
                SveClzZpz => 0b11001,
                SveCntZpz => 0b11010,
                SveCnotZpz => 0b11011,
                _ => 0b11110,
            };
            // `<20>` selects merging (`/m`) vs the FEAT_SVE2p1 zeroing (`/z`) form.
            let opc = if matches!(pred_qual(insn, 1), Some(PredQual::Zeroing)) {
                opc & 0b0_1111
            } else {
                opc
            };
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            base04(0, 0b101) | fld(size, 22) | fld(opc, 16) | fld(pg, 10) | fld(zn, 5) | zd
        }
        // ---- integer immediate arithmetic / minmax / mul / dup (0x25) ----
        SveAddZi | SveSubZi | SveSubrZi | SveSqaddZi | SveUqaddZi | SveSqsubZi | SveUqsubZi => {
            let op16 = match code {
                SveAddZi => 0b000,
                SveSubZi => 0b001,
                SveSubrZi => 0b011,
                SveSqaddZi => 0b100,
                SveUqaddZi => 0b101,
                SveSqsubZi => 0b110,
                _ => 0b111,
            };
            let size = esize(insn, 0)?;
            let zdn = z(insn, 0)?;
            let (imm8, sh) = read_imm8_shift(insn, 2)?;
            base25_imm(size) | fld(0b00, 19) | fld(op16, 16) | fld(sh, 13) | fld(imm8, 5) | zdn
        }
        SveSmaxZi | SveUmaxZi | SveSminZi | SveUminZi => {
            let (op17, u) = match code {
                SveSmaxZi => (0b00, 0),
                SveUmaxZi => (0b00, 1),
                SveSminZi => (0b01, 0),
                _ => (0b01, 1),
            };
            let size = esize(insn, 0)?;
            let zdn = z(insn, 0)?;
            let imm8 = (imm(insn, 2)? as u32) & 0xff;
            base25_imm(size) | fld(0b01, 19) | fld(op17, 17) | fld(u, 16) | fld(imm8, 5) | zdn
        }
        SveMulZi => {
            let size = esize(insn, 0)?;
            let zdn = z(insn, 0)?;
            let imm8 = (simm(insn, 2)? as u32) & 0xff;
            base25_imm(size) | fld(0b10, 19) | fld(0b000, 16) | fld(imm8, 5) | zdn
        }
        SveDupImm => {
            let size = esize(insn, 0)?;
            let zdn = z(insn, 0)?;
            let (imm8, sh) = read_dup_imm(insn, 1)?;
            base25_imm(size) | fld(0b11, 19) | fld(0, 16) | fld(sh, 13) | fld(imm8, 5) | zdn
        }
        // ---- compares ----
        SveCmpZi => enc_cmp_imm(insn)?,
        SveCmpZz | SveCmpZw => enc_cmp_vec(insn, code)?,
        // ---- SVE2 (0x44 / 0x45) ----
        _ => return enc_sve2(insn, code),
    };
    Ok(Some(w))
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// The arrangement of operand `n`, or an error.
#[inline]
fn arr_of(insn: &Instruction, n: usize) -> Result<VA, EncodeError> {
    match insn.op(n) {
        Operand::Reg { arr: Some(a), .. } => Ok(a),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Base word for the 0x04 top byte with `<21>=b21` and `<15:13>=sel`.
#[inline]
fn base04(b21: u32, sel: u32) -> u32 {
    fld(0b00000100, 24) | fld(b21, 21) | fld(sel, 13)
}

/// The lane index of operand `n` (0 when it carries none), for PMOV's vector.
#[inline]
fn pmov_lane(insn: &Instruction, n: usize) -> u32 {
    match insn.op(n) {
        Operand::Reg { lane: Some(l), .. } => l as u32,
        _ => 0,
    }
}

/// Base word for the 0x05 `<15:13>=001` family with `<20:16>=opc2016`. These
/// DUP-scalar / INSR forms have a fixed `<21>=1`.
#[inline]
fn base05_001(size: u32, opc2016: u32) -> u32 {
    fld(0b00000101, 24)
        | fld(size, 22)
        | fld(1, 21)
        | fld(opc2016, 16)
        | fld(0b001, 13)
        | fld(0b001110, 10)
}

/// Base word for the 0x25 integer-immediate `<15:13>=110`, `<21>=1` block.
#[inline]
fn base25_imm(size: u32) -> u32 {
    fld(0b00100101, 24) | fld(size, 22) | fld(1, 21) | fld(0b110, 13)
}

/// Pack an unpredicated arithmetic ZZZ word (0x04, <15:13>=000).
fn arith_zzz(insn: &Instruction, op: u32) -> Result<u32, EncodeError> {
    let size = esize(insn, 0)?;
    let zd = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    Ok(base04(1, 0b000) | fld(size, 22) | fld(op, 10) | fld(zm, 16) | fld(zn, 5) | zd)
}

/// The `<20:16>` opcode for a predicated binary code.
fn pred_binary_opc(code: Code) -> u32 {
    match code {
        SveAddZpzz => 0b00000,
        SveSubZpzz => 0b00001,
        SveSubrZpzz => 0b00011,
        SveSmaxZpzz => 0b01000,
        SveUmaxZpzz => 0b01001,
        SveSminZpzz => 0b01010,
        SveUminZpzz => 0b01011,
        SveSabdZpzz => 0b01100,
        SveUabdZpzz => 0b01101,
        SveMulZpzz => 0b10000,
        SveSmulhZpzz => 0b10010,
        SveUmulhZpzz => 0b10011,
        SveSdivZpzz => 0b10100,
        SveUdivZpzz => 0b10101,
        SveSdivrZpzz => 0b10110,
        SveUdivrZpzz => 0b10111,
        SveOrrZpzz => 0b11000,
        SveEorZpzz => 0b11001,
        SveAndZpzz => 0b11010,
        _ => 0b11011, // BIC
    }
}

/// Read `(zdn, pg, zm)` of a predicated destructive binary: operands are
/// `[Zdn, Pg/M, Zdn, Zm]`.
fn read_pred_binary(insn: &Instruction) -> Result<(u32, u32, u32), EncodeError> {
    let zdn = z(insn, 0)?;
    let pg = p(insn, 1)?;
    let zm = z(insn, 3)?;
    Ok((zdn, pg, zm))
}

/// Read an `(imm8, sh)` from an add/sub immediate operand `n`
/// (`push_imm8_shift` inverse): a plain value, a `<<8`-shifted value, or the
/// `#0x0, lsl #8` form.
fn read_imm8_shift(insn: &Instruction, n: usize) -> Result<(u32, u32), EncodeError> {
    match insn.op(n) {
        Operand::ImmShiftedMove { imm: 0, lsl: 8 } => Ok((0, 1)),
        Operand::ImmUnsigned(v) | Operand::ImmLogical(v) => {
            if v <= 0xff {
                Ok((v as u32, 0))
            } else if v & 0xff == 0 && (v >> 8) <= 0xff {
                Ok(((v >> 8) as u32, 1))
            } else {
                Err(EncodeError::InvalidImmediate)
            }
        }
        Operand::ImmSigned(v) | Operand::ImmSignedDec(v) => {
            let v = v as u64;
            if v <= 0xff {
                Ok((v as u32, 0))
            } else {
                Err(EncodeError::InvalidImmediate)
            }
        }
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Read a DUP/CPY broadcast immediate (`push_dup_imm` inverse): a signed value
/// optionally `<<8`. Returns `(imm8, sh)`.
fn read_dup_imm(insn: &Instruction, n: usize) -> Result<(u32, u32), EncodeError> {
    let v = simm(insn, n)?;
    // Try unshifted: -128..=127.
    if (-128..=127).contains(&v) {
        return Ok(((v as u32) & 0xff, 0));
    }
    // Shifted: v == (imm8 as i8 as i64) << 8.
    if v & 0xff == 0 {
        let hi = v >> 8;
        if (-128..=127).contains(&hi) {
            return Ok(((hi as u32) & 0xff, 1));
        }
    }
    Err(EncodeError::InvalidImmediate)
}

/// Inverse of `sve_bitmask`: pack a 64-bit broadcast `value` into the 13-bit
/// `imm_n:immr:imms` field for the SVE logical-immediate forms.
fn enc_sve_bitmask(value: u64) -> Result<u32, EncodeError> {
    let (n, immr, imms) =
        crate::encode::bits::encode_bit_masks(value, 64).ok_or(EncodeError::InvalidImmediate)?;
    Ok((n << 12) | (immr << 6) | imms)
}

/// Replicate the low element `val` (of arrangement `a`'s width) across a full
/// 64-bit container, matching the decoder's `decode_bit_masks(.., 64)` value.
fn replicate_element(val: u64, a: VA) -> Result<u64, EncodeError> {
    let (_, esz) = esize_of(a)?;
    let masked = if esz >= 64 { val } else { val & ((1u64 << esz) - 1) };
    let mut acc = 0u64;
    let mut shift = 0u32;
    while shift < 64 {
        acc |= masked << shift;
        shift += esz;
    }
    Ok(acc)
}

/// DUP indexed (0x05, <15:13>=001, <12:10>=000): `MOV Zd.T, Zn.T[idx]` or the
/// `MOV Zd.T, V<n>` scalar-broadcast (idx == 0) form.
fn enc_dup_idx(insn: &Instruction) -> Result<u32, EncodeError> {
    let a = arr_of(insn, 0)?;
    let zd = z(insn, 0)?;
    let (esz_idx, _) = esize_q(a)?;
    let (zn, idx) = match insn.op(1) {
        Operand::Reg {
            lane: Some(l), reg, ..
        } => (reg.number() as u32, l as u32),
        // scalar broadcast (idx == 0): B/H/S/D/Q register.
        Operand::Reg { reg, .. } => (reg.number() as u32, 0u32),
        _ => return Err(EncodeError::InvalidOperand),
    };
    // tsz = imm2:tsz (7 bits) = idx<<(esz_idx+1) | (1<<esz_idx).
    let tsz = (idx << (esz_idx + 1)) | (1u32 << esz_idx);
    if tsz > 0x7f {
        return Err(EncodeError::InvalidImmediate);
    }
    let imm2 = (tsz >> 5) & 3;
    let tszlo = tsz & 0x1f;
    // Fixed: <21>=1, <15:13>=001, <12:10>=000.
    Ok(fld(0b00000101, 24) | fld(imm2, 22) | fld(1, 21) | fld(tszlo, 16) | fld(0b001, 13) | fld(zn, 5)
        | zd)
}

/// Element-size index for an arrangement, allowing `.q` (=4).
fn esize_q(a: VA) -> Result<(u32, u32), EncodeError> {
    match a {
        VA::Sb => Ok((0, 8)),
        VA::Sh => Ok((1, 16)),
        VA::Ss => Ok((2, 32)),
        VA::Sd => Ok((3, 64)),
        VA::Sq => Ok((4, 128)),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// SEL (0x05, <15:14>=11, <21>=1): either `SEL Zd, Pg, Zn, Zm` or the MOV
/// alias `MOV Zd.T, Pg/M, Zn.T`.
fn enc_sel(insn: &Instruction) -> Result<u32, EncodeError> {
    let size = esize(insn, 0)?;
    let zd = z(insn, 0)?;
    let base = fld(0b00000101, 24) | fld(size, 22) | fld(1, 21) | fld(0b11, 14);
    if matches!(insn.mnemonic(), Mnemonic::Mov) {
        // MOV: Zm == Zd; operands [Zd, Pg/M, Zn].
        let pg = p(insn, 1)?;
        let zn = z(insn, 2)?;
        Ok(base | fld(zd, 16) | fld(pg, 10) | fld(zn, 5) | zd)
    } else {
        let pg = p(insn, 1)?;
        let zn = z(insn, 2)?;
        let zm = z(insn, 3)?;
        Ok(base | fld(zm, 16) | fld(pg, 10) | fld(zn, 5) | zd)
    }
}

/// SVE ADR vector form (0x04, <15:12>=1010).
fn enc_adr_vec(insn: &Instruction, code: Code) -> Result<u32, EncodeError> {
    let zd = z(insn, 0)?;
    let (zn, zm, amount) = match insn.op(1) {
        Operand::SveMem {
            base, offset, amount, ..
        } => (base.number() as u32, offset.number() as u32, amount as u32),
        _ => return Err(EncodeError::InvalidOperand),
    };
    let opc = match code {
        SveAdrSxtw => 0b00,
        SveAdrUxtw => 0b01,
        SveAdrSameScaled => {
            // arrangement: .s -> <22>=0 (opc=10), .d -> <22>=1 (opc=11).
            let a = match insn.op(1) {
                Operand::SveMem { arr: Some(a), .. } => a,
                _ => return Err(EncodeError::InvalidOperand),
            };
            if a == VA::Sd {
                0b11
            } else {
                0b10
            }
        }
        _ => return Err(EncodeError::InvalidOperand),
    };
    Ok(base04(1, 0b101) | fld(opc, 22) | fld(0b1010, 12) | fld(zm, 16) | fld(amount, 10) | fld(zn, 5)
        | zd)
}

/// INC/DEC vector (0x04, <15:13>=110).
fn enc_incdec_vec(insn: &Instruction) -> Result<u32, EncodeError> {
    let a = arr_of(insn, 0)?;
    let size = arr_size(a)?;
    if size == 0 {
        return Err(EncodeError::InvalidOperand);
    }
    let zdn = z(insn, 0)?;
    let (pattern, imm4) = read_pattern_mul(insn, 1)?;
    let (nonsat, dec, unsigned) = match insn.mnemonic() {
        Mnemonic::Inch | Mnemonic::Incw | Mnemonic::Incd => (true, 0, 0),
        Mnemonic::Dech | Mnemonic::Decw | Mnemonic::Decd => (true, 1, 0),
        Mnemonic::Sqinch | Mnemonic::Sqincw | Mnemonic::Sqincd => (false, 0, 0),
        Mnemonic::Uqinch | Mnemonic::Uqincw | Mnemonic::Uqincd => (false, 0, 1),
        Mnemonic::Sqdech | Mnemonic::Sqdecw | Mnemonic::Sqdecd => (false, 1, 0),
        Mnemonic::Uqdech | Mnemonic::Uqdecw | Mnemonic::Uqdecd => (false, 1, 1),
        _ => return Err(EncodeError::InvalidOperand),
    };
    let mut w = base04(1, 0b110) | fld(size, 22) | fld(imm4, 16) | fld(pattern, 5) | zdn;
    if nonsat {
        w |= fld(1, 20) | fld(dec, 10);
    } else {
        w |= fld(dec, 11) | fld(unsigned, 10);
    }
    Ok(w)
}

/// CNT / INC-DEC scalar by element count (0x04, <15:13>=111).
fn enc_cnt_incdec_scalar(insn: &Instruction, _code: Code) -> Result<u32, EncodeError> {
    let m = insn.mnemonic();
    // CNTB/H/W/D.
    if let Some(size) = cnt_size(m) {
        let rd = g(insn, 0)?;
        let (pattern, imm4) = read_pattern_mul(insn, 1)?;
        return Ok(base04(1, 0b111) | fld(size, 22) | fld(imm4, 16) | fld(0b000, 10) | fld(pattern, 5)
            | rd);
    }
    // INC/DEC scalar (unsaturated, X form).
    if let Some((size, dec)) = incdec_scalar_size(m) {
        let rd = g(insn, 0)?;
        let (pattern, imm4) = read_pattern_mul(insn, 1)?;
        let op = if dec { 0b001 } else { 0b000 };
        return Ok(base04(1, 0b111) | fld(size, 22) | fld(1, 20) | fld(imm4, 16) | fld(op, 10)
            | fld(pattern, 5)
            | rd);
    }
    // Saturating scalar.
    let (size, unsigned, dec) = sat_scalar(m)?;
    let rd = g(insn, 0)?;
    // sf bit: signed -> _x (sf=1) has only Xdn; _sx (sf=0) has Xdn, Wdn.
    //         unsigned -> _x (sf=1) Xdn; _uw (sf=0) Wdn.
    let (sf, tail_start) = if unsigned == 0 {
        // operand 1 is Wdn for _sx form, otherwise pattern/mul.
        match insn.op(1) {
            Operand::Reg { reg, .. } if reg.class() == crate::register::RegClass::Gp => (0u32, 2usize),
            _ => (1u32, 1usize),
        }
    } else {
        // unsigned: width of operand 0 selects sf (X -> 1, W -> 0).
        let sf = match insn.op(0) {
            Operand::Reg { reg, .. } => {
                if reg.width_bits() == 64 {
                    1
                } else {
                    0
                }
            }
            _ => return Err(EncodeError::InvalidOperand),
        };
        (sf, 1usize)
    };
    let (pattern, imm4) = read_pattern_mul(insn, tail_start)?;
    // <12:10>: D<11>, U<10>; for the saturating block op<12>=1.
    let op = fld(1, 12) | fld(if dec { 1 } else { 0 }, 11) | fld(unsigned, 10);
    Ok(base04(1, 0b111) | fld(size, 22) | fld(sf, 20) | fld(imm4, 16) | op | fld(pattern, 5) | rd)
}

/// CNTB/H/W/D size from the mnemonic.
fn cnt_size(m: Mnemonic) -> Option<u32> {
    Some(match m {
        Mnemonic::Cntb => 0,
        Mnemonic::Cnth => 1,
        Mnemonic::Cntw => 2,
        Mnemonic::Cntd => 3,
        _ => return None,
    })
}

/// Non-saturating INC/DEC scalar `(size, is_dec)` from the mnemonic.
fn incdec_scalar_size(m: Mnemonic) -> Option<(u32, bool)> {
    Some(match m {
        Mnemonic::Incb => (0, false),
        Mnemonic::Inch => (1, false),
        Mnemonic::Incw => (2, false),
        Mnemonic::Incd => (3, false),
        Mnemonic::Decb => (0, true),
        Mnemonic::Dech => (1, true),
        Mnemonic::Decw => (2, true),
        Mnemonic::Decd => (3, true),
        _ => return None,
    })
}

/// Saturating scalar `(size, unsigned, dec)` from the mnemonic.
fn sat_scalar(m: Mnemonic) -> Result<(u32, u32, bool), EncodeError> {
    Ok(match m {
        Mnemonic::Sqincb => (0, 0, false),
        Mnemonic::Sqinch => (1, 0, false),
        Mnemonic::Sqincw => (2, 0, false),
        Mnemonic::Sqincd => (3, 0, false),
        Mnemonic::Uqincb => (0, 1, false),
        Mnemonic::Uqinch => (1, 1, false),
        Mnemonic::Uqincw => (2, 1, false),
        Mnemonic::Uqincd => (3, 1, false),
        Mnemonic::Sqdecb => (0, 0, true),
        Mnemonic::Sqdech => (1, 0, true),
        Mnemonic::Sqdecw => (2, 0, true),
        Mnemonic::Sqdecd => (3, 0, true),
        Mnemonic::Uqdecb => (0, 1, true),
        Mnemonic::Uqdech => (1, 1, true),
        Mnemonic::Uqdecw => (2, 1, true),
        Mnemonic::Uqdecd => (3, 1, true),
        _ => return Err(EncodeError::InvalidOperand),
    })
}

/// INC/DEC by predicate count (0x25, <15:13>=100).
fn enc_incdec_pred(insn: &Instruction, code: Code) -> Result<u32, EncodeError> {
    let m = insn.mnemonic();
    let opc = match m {
        Mnemonic::Sqincp => 0b000,
        Mnemonic::Uqincp => 0b001,
        Mnemonic::Sqdecp => 0b010,
        Mnemonic::Uqdecp => 0b011,
        Mnemonic::Incp => 0b100,
        Mnemonic::Decp => 0b101,
        _ => return Err(EncodeError::InvalidOperand),
    };
    // Fixed pattern: <21>=1, <20:19>=01, <15:12>=1000.
    let base = fld(0b00100101, 24) | fld(1, 21) | fld(1, 19) | fld(0b1000, 12);
    match code {
        SveIncDecPVector => {
            let size = esize(insn, 0)?;
            let zdn = z(insn, 0)?;
            let pm = p(insn, 1)?;
            // vector form: <11>=0.
            Ok(base | fld(size, 22) | fld(opc, 16) | fld(pm, 5) | zdn)
        }
        SveIncDecPScalar => {
            // plain INCP/DECP scalar: Xdn, Pg.T. <11>=1.
            let size = esize(insn, 1)?;
            let rdn = g(insn, 0)?;
            let pm = p(insn, 1)?;
            Ok(base | fld(size, 22) | fld(opc, 16) | fld(1, 11) | fld(pm, 5) | rdn)
        }
        SveSqIncDecPScalarSx => {
            let unsigned = matches!(m, Mnemonic::Uqincp | Mnemonic::Uqdecp);
            let size = esize(insn, 1)?;
            let rdn = g(insn, 0)?;
            let pm = p(insn, 1)?;
            let sf = if !unsigned {
                // _x: only [Xdn, Pg.T] (sf=1); _sx: [Xdn, Pg.T, Wdn] (sf=0).
                if matches!(insn.op(2), Operand::Reg { .. }) {
                    0
                } else {
                    1
                }
            } else {
                // _x: Xdn (sf=1); _uw: Wdn (sf=0).
                match insn.op(0) {
                    Operand::Reg { reg, .. } if reg.width_bits() == 64 => 1,
                    _ => 0,
                }
            };
            Ok(base | fld(size, 22) | fld(opc, 16) | fld(1, 11) | fld(sf, 10) | fld(pm, 5) | rdn)
        }
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Predicated shift immediate (0x04, <15:13>=100): ASR/LSR/LSL/ASRD + SVE2.
fn enc_shift_pred_imm(insn: &Instruction, code: Code) -> Result<u32, EncodeError> {
    let m = insn.mnemonic();
    let a = arr_of(insn, 0)?;
    let amount = imm(insn, 3)? as u32;
    let left = matches!(m, Mnemonic::Lsl | Mnemonic::Sqshl | Mnemonic::Uqshl | Mnemonic::Sqshlu);
    let (tsz, imm3) = if left {
        enc_left_shift(a, amount)?
    } else {
        enc_right_shift(a, amount)?
    };
    let opc = match m {
        Mnemonic::Asr => 0b00000,
        Mnemonic::Lsr => 0b00001,
        Mnemonic::Lsl => 0b00011,
        Mnemonic::Asrd => 0b00100,
        Mnemonic::Sqshl => 0b00110,
        Mnemonic::Uqshl => 0b00111,
        Mnemonic::Srshr => 0b01100,
        Mnemonic::Urshr => 0b01101,
        Mnemonic::Sqshlu => 0b01111,
        _ => match code {
            SveAsrZpzi => 0b00000,
            SveLsrZpzi => 0b00001,
            SveLslZpzi => 0b00011,
            _ => 0b00100,
        },
    };
    let zdn = z(insn, 0)?;
    let pg = p(insn, 1)?;
    // tszh=<23:22>, tszl=<9:8>, imm3=<7:5>.
    Ok(base04(0, 0b100) | fld(tsz >> 2, 22) | fld(opc, 16) | fld(pg, 10) | fld(tsz & 3, 8)
        | fld(imm3, 5)
        | zdn)
}

/// Compare with immediate (0x24 unsigned / 0x25 signed).
fn enc_cmp_imm(insn: &Instruction) -> Result<u32, EncodeError> {
    let m = insn.mnemonic();
    let size = esize(insn, 0)?;
    let pd = p(insn, 0)?;
    let pg = p(insn, 1)?;
    let zn = z(insn, 2)?;
    // Unsigned-immediate compares (0x24, <21>=1): imm7 unsigned.
    if matches!(m, Mnemonic::Cmphs | Mnemonic::Cmphi | Mnemonic::Cmplo | Mnemonic::Cmpls) {
        let imm7 = (imm(insn, 3)? as u32) & 0x7f;
        let (b13, ne) = match m {
            Mnemonic::Cmphs => (0, 0),
            Mnemonic::Cmphi => (0, 1),
            Mnemonic::Cmplo => (1, 0),
            _ => (1, 1),
        };
        return Ok(fld(0b00100100, 24) | fld(size, 22) | fld(1, 21) | fld(imm7, 14) | fld(b13, 13)
            | fld(pg, 10)
            | fld(zn, 5)
            | fld(ne, 4)
            | pd);
    }
    // Signed-immediate compares (0x25, <21>=0): imm5 signed.
    let imm5 = (simm(insn, 3)? as u32) & 0x1f;
    let (op, lt, ne) = match m {
        Mnemonic::Cmpge => (0, 0, 0),
        Mnemonic::Cmpgt => (0, 0, 1),
        Mnemonic::Cmplt => (0, 1, 0),
        Mnemonic::Cmple => (0, 1, 1),
        Mnemonic::Cmpeq => (1, 0, 0),
        Mnemonic::Cmpne => (1, 0, 1),
        _ => return Err(EncodeError::InvalidOperand),
    };
    // <15:13>: op<15>, then <14>=0, <13>=lt. sel = (op<<2)|(0<<1)|lt -> bits 15:13.
    let sel = fld(op, 15) | fld(lt, 13);
    Ok(fld(0b00100101, 24) | fld(size, 22) | fld(imm5, 16) | sel | fld(pg, 10) | fld(zn, 5)
        | fld(ne, 4)
        | pd)
}

/// Compare vector / wide (0x24, <21>=0).
fn enc_cmp_vec(insn: &Instruction, code: Code) -> Result<u32, EncodeError> {
    let m = insn.mnemonic();
    let wide = matches!(code, SveCmpZw);
    let size = esize(insn, 0)?;
    let pd = p(insn, 0)?;
    let pg = p(insn, 1)?;
    let zn = z(insn, 2)?;
    let zm = z(insn, 3)?;
    let (op, b14, b13, ne) = match (m, wide) {
        (Mnemonic::Cmphs, false) => (0, 0, 0, 0),
        (Mnemonic::Cmphi, false) => (0, 0, 0, 1),
        (Mnemonic::Cmpeq, true) => (0, 0, 1, 0),
        (Mnemonic::Cmpne, true) => (0, 0, 1, 1),
        (Mnemonic::Cmpge, true) => (0, 1, 0, 0),
        (Mnemonic::Cmpgt, true) => (0, 1, 0, 1),
        (Mnemonic::Cmplt, true) => (0, 1, 1, 0),
        (Mnemonic::Cmple, true) => (0, 1, 1, 1),
        (Mnemonic::Cmpge, false) => (1, 0, 0, 0),
        (Mnemonic::Cmpgt, false) => (1, 0, 0, 1),
        (Mnemonic::Cmpeq, false) => (1, 0, 1, 0),
        (Mnemonic::Cmpne, false) => (1, 0, 1, 1),
        (Mnemonic::Cmphs, true) => (1, 1, 0, 0),
        (Mnemonic::Cmphi, true) => (1, 1, 0, 1),
        (Mnemonic::Cmplo, true) => (1, 1, 1, 0),
        (Mnemonic::Cmpls, true) => (1, 1, 1, 1),
        _ => return Err(EncodeError::InvalidOperand),
    };
    Ok(fld(0b00100100, 24) | fld(size, 22) | fld(zm, 16) | fld(op, 15) | fld(b14, 14) | fld(b13, 13)
        | fld(pg, 10)
        | fld(zn, 5)
        | fld(ne, 4)
        | pd)
}

// SVE2 (0x44 / 0x45) families are inverted in `sve2.rs`-style code below.
include!("int_sve2.rs");
