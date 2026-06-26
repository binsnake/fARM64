//! Inverse of [`crate::decode::sve::sve_fp`] — SVE/SVE2 floating-point family.

use super::{esize, fld, lane, p, pred_qual, reg, sfp, z};
use crate::encode::bits::encode_vfp_imm;
use crate::encode::EncodeError;
use crate::enums::VectorArrangement as VA;
use crate::instruction::Instruction;
use crate::mnemonic::Code;
use crate::operand::{Operand, PredQual};

use Code::*;

/// `true` for every floating-point SVE [`Code`].
pub(super) fn is_fp(code: Code) -> bool {
    matches!(
        code,
        // unpredicated 3/2-source
        SveFaddZzz | SveFsubZzz | SveFmulZzz | SveFtsmulZzz | SveFrecpsZzz | SveFrsqrtsZzz
        // unary / reductions / recip-est
            | SveFaddv | SveFmaxnmv | SveFminnmv | SveFmaxv | SveFminv | SveFadda
            | SveFrecpe | SveFrsqrte
        // compare with zero
            | SveFcmgeZ0 | SveFcmgtZ0 | SveFcmltZ0 | SveFcmleZ0 | SveFcmeqZ0 | SveFcmneZ0
        // predicated binary + immediate + FTMAD
            | SveFaddZpzz | SveFsubZpzz | SveFmulZpzz | SveFsubrZpzz | SveFmaxnmZpzz | SveFminnmZpzz
            | SveFmaxZpzz | SveFminZpzz | SveFabdZpzz | SveFscaleZpzz | SveFmulxZpzz | SveFdivrZpzz
            | SveFdivZpzz
            | SveFaddZpzi | SveFsubZpzi | SveFmulZpzi | SveFsubrZpzi | SveFmaxnmZpzi | SveFminnmZpzi
            | SveFmaxZpzi | SveFminZpzi | SveFtmad
        // predicated unary (rint/recpx/sqrt) + converts + flogb
            | SveFrintnZpz | SveFrintpZpz | SveFrintmZpz | SveFrintzZpz | SveFrintaZpz | SveFrintxZpz
            | SveFrintiZpz | SveFrecpxZpz | SveFsqrtZpz | SveFlogbZpz
            | SveFcvt | SveBfcvt | SveFcvtx | SveFcvtzs | SveFcvtzu | SveScvtf | SveUcvtf
        // vector compare
            | SveFcmgeZz | SveFcmgtZz | SveFcmeqZz | SveFcmneZz | SveFcmuoZz | SveFacgeZz | SveFacgtZz
        // FMA 4-operand
            | SveFmlaZpzzz | SveFmlsZpzzz | SveFnmlaZpzzz | SveFnmlsZpzzz | SveFmadZpzzz | SveFmsbZpzzz
            | SveFnmadZpzzz | SveFnmsbZpzzz
        // 0x64
            | SveFcmlaZpzzz | SveFcadd | SveFaddpZpzz | SveFmaxnmpZpzz | SveFminnmpZpzz | SveFmaxpZpzz
            | SveFminpZpzz | SveFcvtnt | SveFcvtlt | SveFcvtxnt | SveBfcvtnt
            | SveFmlaIdx | SveFmlsIdx | SveFmulIdx | SveFcmlaIdx | SveFmmla | SveBfmmla
            | SveBfdot | SveBfdotIdx | SveBfmlalb | SveBfmlalt | SveBfmlalbIdx | SveBfmlaltIdx
            | SveFmlalb | SveFmlalt | SveFmlslb | SveFmlslt
            | SveFmlalbIdx | SveFmlaltIdx | SveFmlslbIdx | SveFmlsltIdx
        // 0x04 / 0x05 / 0x25 leaves
            | SveFabsZpz | SveFnegZpz | SveFexpa | SveFtssel | SveFcpy | SveFdup
        // sve2.1 quadword FP reductions
            | SveFaddqv | SveFmaxnmqv | SveFminnmqv | SveFmaxqv | SveFminqv
    )
}

/// Encode a floating-point SVE instruction.
pub(super) fn enc(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
    let w = match code {
        // ---- 0x65 unpredicated 3-register ----
        SveFaddZzz | SveFsubZzz | SveFmulZzz | SveFtsmulZzz | SveFrecpsZzz | SveFrsqrtsZzz => {
            let opc = match code {
                SveFaddZzz => 0b000,
                SveFsubZzz => 0b001,
                SveFmulZzz => 0b010,
                SveFtsmulZzz => 0b011,
                SveFrecpsZzz => 0b110,
                _ => 0b111,
            };
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            base65(0) | fld(size, 22) | fld(zm, 16) | fld(opc, 10) | fld(zn, 5) | zd
        }
        // ---- 0x65 reductions / recip-est / fadda ----
        SveFaddv | SveFmaxnmv | SveFminnmv | SveFmaxv | SveFminv => {
            let opc = match code {
                SveFaddv => 0b000,
                SveFmaxnmv => 0b100,
                SveFminnmv => 0b101,
                SveFmaxv => 0b110,
                _ => 0b111,
            };
            let size = esize(insn, 2)?;
            let vd = sfp(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            base65(0) | fld(size, 22) | fld(opc, 16) | fld(0b001, 13) | fld(pg, 10) | fld(zn, 5) | vd
        }
        SveFadda => {
            let size = esize(insn, 3)?;
            let vd = sfp(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 3)?;
            base65(0) | fld(size, 22) | fld(0b11000, 16) | fld(0b001, 13) | fld(pg, 10) | fld(zn, 5)
                | vd
        }
        SveFrecpe | SveFrsqrte => {
            let opc = if matches!(code, SveFrecpe) { 0b01110 } else { 0b01111 };
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            base65(0) | fld(size, 22) | fld(opc, 16) | fld(0b001100, 10) | fld(zn, 5) | zd
        }
        // ---- 0x65 compare with zero ----
        SveFcmgeZ0 | SveFcmgtZ0 | SveFcmltZ0 | SveFcmleZ0 | SveFcmeqZ0 | SveFcmneZ0 => {
            let (op, b4) = match code {
                SveFcmgeZ0 => (0b000, 0),
                SveFcmgtZ0 => (0b000, 1),
                SveFcmltZ0 => (0b001, 0),
                SveFcmleZ0 => (0b001, 1),
                SveFcmeqZ0 => (0b010, 0),
                _ => (0b011, 0),
            };
            let size = esize(insn, 0)?;
            let pd = p(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            base65(0) | fld(size, 22) | fld(1, 20) | fld(op, 16) | fld(0b001, 13) | fld(pg, 10)
                | fld(zn, 5)
                | fld(b4, 4)
                | pd
        }
        // ---- 0x65 predicated binary ----
        SveFaddZpzz | SveFsubZpzz | SveFmulZpzz | SveFsubrZpzz | SveFmaxnmZpzz | SveFminnmZpzz
        | SveFmaxZpzz | SveFminZpzz | SveFabdZpzz | SveFscaleZpzz | SveFmulxZpzz | SveFdivrZpzz
        | SveFdivZpzz => {
            let opc = match code {
                SveFaddZpzz => 0b00000,
                SveFsubZpzz => 0b00001,
                SveFmulZpzz => 0b00010,
                SveFsubrZpzz => 0b00011,
                SveFmaxnmZpzz => 0b00100,
                SveFminnmZpzz => 0b00101,
                SveFmaxZpzz => 0b00110,
                SveFminZpzz => 0b00111,
                SveFabdZpzz => 0b01000,
                SveFscaleZpzz => 0b01001,
                SveFmulxZpzz => 0b01010,
                SveFdivrZpzz => 0b01100,
                _ => 0b01101,
            };
            let size = esize(insn, 0)?;
            let zdn = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zm = z(insn, 3)?;
            base65(0) | fld(size, 22) | fld(opc, 16) | fld(0b100, 13) | fld(pg, 10) | fld(zm, 5) | zdn
        }
        // ---- 0x65 predicated immediate ----
        SveFaddZpzi | SveFsubZpzi | SveFmulZpzi | SveFsubrZpzi | SveFmaxnmZpzi | SveFminnmZpzi
        | SveFmaxZpzi | SveFminZpzi => {
            let (opc3, i1) = fp_imm_field(insn, code)?;
            let size = esize(insn, 0)?;
            let zdn = z(insn, 0)?;
            let pg = p(insn, 1)?;
            base65(0) | fld(size, 22) | fld(0b11, 19) | fld(opc3, 16) | fld(0b100, 13) | fld(pg, 10)
                | fld(i1, 5)
                | zdn
        }
        SveFtmad => {
            let size = esize(insn, 0)?;
            let zdn = z(insn, 0)?;
            let zm = z(insn, 2)?;
            let imm3 = super::imm(insn, 3)? as u32 & 7;
            base65(0) | fld(size, 22) | fld(0b10, 19) | fld(imm3, 16) | fld(0b100000, 10) | fld(zm, 5)
                | zdn
        }
        // ---- 0x65 predicated unary (rint / recpx / sqrt) ----
        SveFrintnZpz | SveFrintpZpz | SveFrintmZpz | SveFrintzZpz | SveFrintaZpz | SveFrintxZpz
        | SveFrintiZpz | SveFrecpxZpz | SveFsqrtZpz => {
            let opc = match code {
                SveFrintnZpz => 0b00000,
                SveFrintpZpz => 0b00001,
                SveFrintmZpz => 0b00010,
                SveFrintzZpz => 0b00011,
                SveFrintaZpz => 0b00100,
                SveFrintxZpz => 0b00110,
                SveFrintiZpz => 0b00111,
                SveFrecpxZpz => 0b01100,
                _ => 0b01101,
            };
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            base65(0) | fld(size, 22) | fld(opc, 16) | fld(0b101, 13) | fld(pg, 10) | fld(zn, 5) | zd
        }
        // ---- 0x65 FLOGB ----
        SveFlogbZpz => {
            let a = arr_of(insn, 0)?;
            let sz = super::arr_size(a)?;
            let zd = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            base65(0) | fld(0b00, 22) | fld(0b011, 19) | fld(sz, 17) | fld(0b101, 13) | fld(pg, 10)
                | fld(zn, 5)
                | zd
        }
        // ---- 0x65 converts ----
        SveFcvt | SveBfcvt | SveFcvtx | SveFcvtzs | SveFcvtzu | SveScvtf | SveUcvtf => {
            enc_65_convert(insn, code)?
        }
        // ---- 0x65 vector compare ----
        SveFcmgeZz | SveFcmgtZz | SveFcmeqZz | SveFcmneZz | SveFcmuoZz | SveFacgeZz | SveFacgtZz => {
            let (sel, b4) = match code {
                SveFcmgeZz => (0b010, 0),
                SveFcmgtZz => (0b010, 1),
                SveFcmeqZz => (0b011, 0),
                SveFcmneZz => (0b011, 1),
                SveFcmuoZz => (0b110, 0),
                SveFacgeZz => (0b110, 1),
                _ => (0b111, 1),
            };
            let size = esize(insn, 0)?;
            let pd = p(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            let zm = z(insn, 3)?;
            base65(0) | fld(size, 22) | fld(zm, 16) | fld(sel, 13) | fld(pg, 10) | fld(zn, 5)
                | fld(b4, 4)
                | pd
        }
        // ---- 0x65 FMA 4-operand ----
        SveFmlaZpzzz | SveFmlsZpzzz | SveFnmlaZpzzz | SveFnmlsZpzzz => {
            let op = match code {
                SveFmlaZpzzz => 0b000,
                SveFmlsZpzzz => 0b001,
                SveFnmlaZpzzz => 0b010,
                _ => 0b011,
            };
            let size = esize(insn, 0)?;
            let zda = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            let zm = z(insn, 3)?;
            base65(1) | fld(size, 22) | fld(zm, 16) | fld(op, 13) | fld(pg, 10) | fld(zn, 5) | zda
        }
        SveFmadZpzzz | SveFmsbZpzzz | SveFnmadZpzzz | SveFnmsbZpzzz => {
            let op = match code {
                SveFmadZpzzz => 0b100,
                SveFmsbZpzzz => 0b101,
                SveFnmadZpzzz => 0b110,
                _ => 0b111,
            };
            let size = esize(insn, 0)?;
            let zdn = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zm = z(insn, 2)?;
            let za = z(insn, 3)?;
            base65(1) | fld(size, 22) | fld(za, 16) | fld(op, 13) | fld(pg, 10) | fld(zm, 5) | zdn
        }
        // ---- 0x64 FCMLA vector ----
        SveFcmlaZpzzz => {
            let size = esize(insn, 0)?;
            let zda = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            let zm = z(insn, 3)?;
            let rot = rot4(insn, 4)?;
            base64(0) | fld(size, 22) | fld(zm, 16) | fld(rot, 13) | fld(pg, 10) | fld(zn, 5) | zda
        }
        // ---- 0x64 FCADD ----
        SveFcadd => {
            let size = esize(insn, 0)?;
            let zdn = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zm = z(insn, 3)?;
            let rot = match super::imm(insn, 4)? {
                90 => 0,
                270 => 1,
                _ => return Err(EncodeError::InvalidImmediate),
            };
            base64(0) | fld(size, 22) | fld(rot, 16) | fld(0b100, 13) | fld(pg, 10) | fld(zm, 5) | zdn
        }
        // ---- 0x64 pairwise ----
        SveFaddpZpzz | SveFmaxnmpZpzz | SveFminnmpZpzz | SveFmaxpZpzz | SveFminpZpzz => {
            let opc = match code {
                SveFaddpZpzz => 0b000,
                SveFmaxnmpZpzz => 0b100,
                SveFminnmpZpzz => 0b101,
                SveFmaxpZpzz => 0b110,
                _ => 0b111,
            };
            let size = esize(insn, 0)?;
            let zdn = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zm = z(insn, 3)?;
            base64(0) | fld(size, 22) | fld(0b10, 19) | fld(opc, 16) | fld(0b100, 13) | fld(pg, 10)
                | fld(zm, 5)
                | zdn
        }
        // ---- 0x64 narrow/long converts ----
        SveFcvtnt | SveFcvtlt | SveFcvtxnt | SveBfcvtnt => enc_64_narrow(insn, code)?,
        // ---- 0x64 indexed multiply-add / multiply ----
        SveFmlaIdx | SveFmlsIdx => {
            let is_fmls = matches!(code, SveFmlsIdx);
            enc_64_fmla_idx(insn, is_fmls)?
        }
        SveFmulIdx => enc_64_fmul_idx(insn)?,
        SveFcmlaIdx => enc_64_fcmla_idx(insn)?,
        SveFmmla => {
            let a = arr_of(insn, 0)?;
            let opc = if a == VA::Sd { 0b11 } else { 0b10 };
            let zda = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            base64(1) | fld(opc, 22) | fld(zm, 16) | fld(0b111001, 10) | fld(zn, 5) | zda
        }
        SveBfmmla => {
            let zda = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            base64(1) | fld(0b01, 22) | fld(zm, 16) | fld(0b111001, 10) | fld(zn, 5) | zda
        }
        // ---- 0x64 BFDOT ----
        SveBfdot => {
            let zda = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            base64(1) | fld(0b01, 22) | fld(zm, 16) | fld(0b100000, 10) | fld(zn, 5) | zda
        }
        SveBfdotIdx => {
            let zda = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            let idx = lane(insn, 2)?;
            base64(1) | fld(0b01, 22) | fld(idx & 3, 19) | fld(zm & 7, 16) | fld(0b0100, 12)
                | fld(zn, 5)
                | zda
        }
        // ---- 0x64 bf16 / half multiply-add-long ----
        SveBfmlalb | SveBfmlalt | SveFmlalb | SveFmlalt | SveFmlslb | SveFmlslt => {
            enc_64_mlal_vec(insn, code)?
        }
        SveBfmlalbIdx | SveBfmlaltIdx | SveFmlalbIdx | SveFmlaltIdx | SveFmlslbIdx | SveFmlsltIdx => {
            enc_64_mlal_idx(insn, code)?
        }
        // ---- 0x04 FABS / FNEG ----
        SveFabsZpz | SveFnegZpz => {
            let opc = if matches!(code, SveFabsZpz) { 0b11100 } else { 0b11101 };
            // `<20>` selects merging (`/m`) vs FEAT_SVE2p1 zeroing (`/z`).
            let opc = if matches!(pred_qual(insn, 1), Some(PredQual::Zeroing)) {
                opc & 0b0_1111
            } else {
                opc
            };
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            fld(0b00000100, 24) | fld(size, 22) | fld(opc, 16) | fld(0b101, 13) | fld(pg, 10)
                | fld(zn, 5)
                | zd
        }
        // ---- 0x64 SVE2.1 quadword FP reductions to a NEON `V` register ----
        SveFaddqv | SveFmaxnmqv | SveFminnmqv | SveFmaxqv | SveFminqv => {
            let opc = match code {
                SveFaddqv => 0b10000,
                SveFmaxnmqv => 0b10100,
                SveFminnmqv => 0b10101,
                SveFmaxqv => 0b10110,
                _ => 0b10111,
            };
            let size = esize(insn, 2)?; // element size from the source `Zn`
            let vd = reg(insn, 0)?; // destination `Vd`
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            fld(0b01100100, 24) | fld(size, 22) | fld(opc, 16) | fld(0b101, 13) | fld(pg, 10)
                | fld(zn, 5)
                | vd
        }
        // ---- 0x04 FEXPA / FTSSEL ----
        SveFexpa => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            fld(0b00000100, 24) | fld(size, 22) | fld(1, 21) | fld(0b00000, 16) | fld(0b101110, 10)
                | fld(zn, 5)
                | zd
        }
        SveFtssel => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            fld(0b00000100, 24) | fld(size, 22) | fld(1, 21) | fld(zm, 16) | fld(0b101100, 10)
                | fld(zn, 5)
                | zd
        }
        // ---- 0x05 FCPY (FMOV) ----
        SveFcpy => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let imm8 = fp_imm8(insn, 2, size)?;
            fld(0b00000101, 24) | fld(size, 22) | fld(0b01, 20) | fld(pg, 16) | fld(0b110, 13)
                | fld(imm8, 5)
                | zd
        }
        // ---- 0x25 FDUP (FMOV) ----
        SveFdup => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let imm8 = fp_imm8(insn, 1, size)?;
            fld(0b00100101, 24) | fld(size, 22) | fld(0b111001, 16) | fld(0b110, 13) | fld(imm8, 5)
                | zd
        }
        _ => return Ok(None),
    };
    Ok(Some(w))
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// Base word for top byte 0x65 with `<21>=b21`.
#[inline]
fn base65(b21: u32) -> u32 {
    fld(0b01100101, 24) | fld(b21, 21)
}

/// Base word for top byte 0x64 with `<21>=b21`.
#[inline]
fn base64(b21: u32) -> u32 {
    fld(0b01100100, 24) | fld(b21, 21)
}

/// The arrangement of operand `n`.
#[inline]
fn arr_of(insn: &Instruction, n: usize) -> Result<VA, EncodeError> {
    match insn.op(n) {
        Operand::Reg { arr: Some(a), .. } => Ok(a),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Recover the 2-bit `rot` field from a rotation immediate (0/90/180/270).
fn rot4(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    match super::imm(insn, n)? {
        0 => Ok(0),
        90 => Ok(1),
        180 => Ok(2),
        270 => Ok(3),
        _ => Err(EncodeError::InvalidImmediate),
    }
}

/// `(opc3, i1)` for the FP predicated-immediate forms. The decoder maps each
/// op to one of {0.0,0.5,1.0,2.0} via the `i1` bit; invert by matching the
/// stored f32 against the two legal constants for the op.
fn fp_imm_field(insn: &Instruction, code: Code) -> Result<(u32, u32), EncodeError> {
    // Operand layout: [Zdn, Pg/M, Zdn, #imm] — the FP constant is operand 3.
    let v = match insn.op(3) {
        Operand::FpImm(f) => f,
        _ => return Err(EncodeError::InvalidOperand),
    };
    let bits = v.to_bits();
    // The two candidates per op: (opc3, val_for_i1_0, val_for_i1_1).
    let (opc3, c0, c1) = match code {
        SveFaddZpzi => (0b000, HALF, ONE),
        SveFsubZpzi => (0b001, HALF, ONE),
        SveFmulZpzi => (0b010, HALF, TWO),
        SveFsubrZpzi => (0b011, HALF, ONE),
        SveFmaxnmZpzi => (0b100, ZERO, ONE),
        SveFminnmZpzi => (0b101, ZERO, ONE),
        SveFmaxZpzi => (0b110, ZERO, ONE),
        _ => (0b111, ZERO, ONE),
    };
    let i1 = if bits == c0 {
        0
    } else if bits == c1 {
        1
    } else {
        return Err(EncodeError::InvalidImmediate);
    };
    Ok((opc3, i1))
}

const ZERO: u32 = 0x0000_0000;
const HALF: u32 = 0x3f00_0000;
const ONE: u32 = 0x3f80_0000;
const TWO: u32 = 0x4000_0000;

/// Recover the 8-bit `VFPExpandImm` encoding from an FP-immediate operand `n`
/// at element width selected by `size` (1=h/16, 2=s/32, else d/64).
fn fp_imm8(insn: &Instruction, n: usize, size: u32) -> Result<u32, EncodeError> {
    let v = match insn.op(n) {
        Operand::FpImm(f) => f,
        _ => return Err(EncodeError::InvalidOperand),
    };
    let nbits = match size {
        1 => 16,
        2 => 32,
        _ => 64,
    };
    encode_vfp_imm(v, nbits).ok_or(EncodeError::InvalidImmediate)
}

/// FP convert sub-block (0x65). Selects the `<23:16>` opcode from (code, arrs).
fn enc_65_convert(insn: &Instruction, code: Code) -> Result<u32, EncodeError> {
    let da = arr_of(insn, 0)?;
    let zd = z(insn, 0)?;
    let pg = p(insn, 1)?;
    let zn = z(insn, 2)?;
    let sa = arr_of(insn, 2)?;
    let sel = conv_sel_65(code, da, sa)?;
    Ok(base65(0) | fld(sel, 16) | fld(0b101, 13) | fld(pg, 10) | fld(zn, 5) | zd)
}

/// The `<23:16>` selector for an 0x65 convert from (code, dst, src).
fn conv_sel_65(code: Code, da: VA, sa: VA) -> Result<u32, EncodeError> {
    use VA::{Sd, Sh, Ss};
    let sel = match (code, da, sa) {
        (SveFcvt, Ss, Sh) => 0b10_001_001,
        (SveFcvt, Sh, Ss) => 0b10_001_000,
        (SveFcvt, Sd, Sh) => 0b11_001_001,
        (SveFcvt, Sh, Sd) => 0b11_001_000,
        (SveFcvt, Sd, Ss) => 0b11_001_011,
        (SveFcvt, Ss, Sd) => 0b11_001_010,
        (SveBfcvt, Sh, Ss) => 0b10_001_010,
        (SveFcvtx, Ss, Sd) => 0b00_001_010,
        (SveFcvtzs, Sh, Sh) => 0b01_011_010,
        (SveFcvtzs, Ss, Sh) => 0b01_011_100,
        (SveFcvtzs, Sd, Sh) => 0b01_011_110,
        (SveFcvtzs, Ss, Ss) => 0b10_011_100,
        (SveFcvtzs, Sd, Ss) => 0b11_011_100,
        (SveFcvtzs, Ss, Sd) => 0b11_011_000,
        (SveFcvtzs, Sd, Sd) => 0b11_011_110,
        (SveFcvtzu, Sh, Sh) => 0b01_011_011,
        (SveFcvtzu, Ss, Sh) => 0b01_011_101,
        (SveFcvtzu, Sd, Sh) => 0b01_011_111,
        (SveFcvtzu, Ss, Ss) => 0b10_011_101,
        (SveFcvtzu, Sd, Ss) => 0b11_011_101,
        (SveFcvtzu, Ss, Sd) => 0b11_011_001,
        (SveFcvtzu, Sd, Sd) => 0b11_011_111,
        (SveScvtf, Sh, Sh) => 0b01_010_010,
        (SveScvtf, Sh, Ss) => 0b01_010_100,
        (SveScvtf, Sh, Sd) => 0b01_010_110,
        (SveScvtf, Ss, Ss) => 0b10_010_100,
        (SveScvtf, Sd, Ss) => 0b11_010_000,
        (SveScvtf, Ss, Sd) => 0b11_010_100,
        (SveScvtf, Sd, Sd) => 0b11_010_110,
        (SveUcvtf, Sh, Sh) => 0b01_010_011,
        (SveUcvtf, Sh, Ss) => 0b01_010_101,
        (SveUcvtf, Sh, Sd) => 0b01_010_111,
        (SveUcvtf, Ss, Ss) => 0b10_010_101,
        (SveUcvtf, Sd, Ss) => 0b11_010_001,
        (SveUcvtf, Ss, Sd) => 0b11_010_101,
        (SveUcvtf, Sd, Sd) => 0b11_010_111,
        _ => return Err(EncodeError::InvalidOperand),
    };
    Ok(sel)
}

/// 0x64 narrow / long converts.
#[allow(clippy::unusual_byte_groupings)]
fn enc_64_narrow(insn: &Instruction, code: Code) -> Result<u32, EncodeError> {
    let da = arr_of(insn, 0)?;
    let sa = arr_of(insn, 2)?;
    let zd = z(insn, 0)?;
    let pg = p(insn, 1)?;
    let zn = z(insn, 2)?;
    use VA::{Sd, Sh, Ss};
    let sel = match (code, da, sa) {
        (SveFcvtnt, Sh, Ss) => 0b10_0010_00,
        (SveFcvtnt, Ss, Sd) => 0b11_0010_10,
        (SveFcvtlt, Ss, Sh) => 0b10_0010_01,
        (SveFcvtlt, Sd, Ss) => 0b11_0010_11,
        (SveFcvtxnt, Ss, Sd) => 0b00_0010_10,
        (SveBfcvtnt, Sh, Ss) => 0b10_0010_10,
        _ => return Err(EncodeError::InvalidOperand),
    };
    Ok(base64(0) | fld(sel, 16) | fld(0b101, 13) | fld(pg, 10) | fld(zn, 5) | zd)
}

/// FMLA/FMLS by indexed element (0x64).
fn enc_64_fmla_idx(insn: &Instruction, is_fmls: bool) -> Result<u32, EncodeError> {
    let zda = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let idx = lane(insn, 2)?;
    let op = if is_fmls { 1 } else { 0 };
    let (sizebits, idxzm) = fp_idx_layout(arr_of(insn, 0)?, idx, zm)?;
    Ok(base64(1) | sizebits | idxzm | fld(0b00000, 11) | fld(op, 10) | fld(zn, 5) | zda)
}

/// FMUL by indexed element (0x64).
fn enc_64_fmul_idx(insn: &Instruction) -> Result<u32, EncodeError> {
    let zd = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let idx = lane(insn, 2)?;
    let (sizebits, idxzm) = fp_idx_layout(arr_of(insn, 0)?, idx, zm)?;
    Ok(base64(1) | sizebits | idxzm | fld(0b001000, 10) | fld(zn, 5) | zd)
}

/// The size + index/Zm bits for an FP by-element form (`.h`/`.s`/`.d`).
fn fp_idx_layout(a: VA, idx: u32, zm: u32) -> Result<(u32, u32), EncodeError> {
    match a {
        // .h: size=0x (we set <23:22>=00), Zm<18:16>, idx=i3h:i3l=<22>:<20:19>.
        VA::Sh => Ok((
            fld((idx >> 2) & 1, 22),
            fld(idx & 3, 19) | fld(zm & 7, 16),
        )),
        VA::Ss => Ok((fld(0b10, 22), fld(idx & 3, 19) | fld(zm & 7, 16))),
        VA::Sd => Ok((fld(0b11, 22), fld(idx & 1, 20) | fld(zm & 0xf, 16))),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// FCMLA by indexed element (0x64).
fn enc_64_fcmla_idx(insn: &Instruction) -> Result<u32, EncodeError> {
    let zda = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let idx = lane(insn, 2)?;
    let rot = rot4(insn, 3)?;
    let a = arr_of(insn, 0)?;
    // .h (size=10): Zm<18:16>, i2=<20:19>; .s (size=11): Zm<19:16>, i1=<20>.
    let (sizebits, idxzm) = match a {
        VA::Sh => (fld(0b10, 22), fld(idx & 3, 19) | fld(zm & 7, 16)),
        VA::Ss => (fld(0b11, 22), fld(idx & 1, 20) | fld(zm & 0xf, 16)),
        _ => return Err(EncodeError::InvalidOperand),
    };
    Ok(base64(1) | sizebits | idxzm | fld(0b0001, 12) | fld(rot, 10) | fld(zn, 5) | zda)
}

/// 0x64 bf16/half multiply-add-long, vector form.
fn enc_64_mlal_vec(insn: &Instruction, code: Code) -> Result<u32, EncodeError> {
    let (o2, op, t) = mlal_fields(code)?;
    let zda = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    // Fixed <23>=1; o2 at <22>; <15:14>=10, <13>=op, <12:11>=00, T=<10>.
    Ok(base64(1) | fld(1, 23) | fld(o2, 22) | fld(zm, 16) | fld(0b10, 14) | fld(op, 13) | fld(t, 10)
        | fld(zn, 5)
        | zda)
}

/// 0x64 bf16/half multiply-add-long, indexed form.
fn enc_64_mlal_idx(insn: &Instruction, code: Code) -> Result<u32, EncodeError> {
    let (o2, op, t) = mlal_fields(code)?;
    let zda = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let idx = lane(insn, 2)?;
    let i3h = (idx >> 1) & 3;
    let i3l = idx & 1;
    Ok(base64(1) | fld(1, 23) | fld(o2, 22) | fld(i3h, 19) | fld(zm & 7, 16) | fld(0b01, 14)
        | fld(op, 13)
        | fld(i3l, 11)
        | fld(t, 10)
        | fld(zn, 5)
        | zda)
}

/// `(o2, op, T)` for the MLAL-long family.
fn mlal_fields(code: Code) -> Result<(u32, u32, u32), EncodeError> {
    Ok(match code {
        SveBfmlalb => (1, 0, 0),
        SveBfmlalt => (1, 0, 1),
        SveBfmlalbIdx => (1, 0, 0),
        SveBfmlaltIdx => (1, 0, 1),
        SveFmlalb => (0, 0, 0),
        SveFmlalt => (0, 0, 1),
        SveFmlslb => (0, 1, 0),
        SveFmlslt => (0, 1, 1),
        SveFmlalbIdx => (0, 0, 0),
        SveFmlaltIdx => (0, 0, 1),
        SveFmlslbIdx => (0, 1, 0),
        SveFmlsltIdx => (0, 1, 1),
        _ => return Err(EncodeError::InvalidOperand),
    })
}
