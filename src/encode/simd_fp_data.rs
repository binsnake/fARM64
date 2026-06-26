// Included into `simd_fp.rs` — Advanced SIMD data-movement encoders.
//
// Inverse of `crate::decode::simd_fp::simd_data`: copy (DUP/INS/SMOV/UMOV),
// permute (ZIP/UZP/TRN), extract (EXT), table (TBL/TBX), modified-immediate
// (MOVI/MVNI/ORR/BIC/FMOV) and shift-by-immediate (vector + scalar).

mod simd_data {
    use super::*;

    pub(super) fn encode(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
        use Code::*;
        let w = match code {
            DupElement | DupGeneral | InsGeneral | InsElement | Smov | Umov
            | DupElementScalar => enc_copy(insn, code)?,
            Uzp1 | Uzp2 | Trn1 | Trn2 | Zip1 | Zip2 => enc_permute(insn, code)?,
            Ext => enc_ext(insn)?,
            Tbl | Tbx => enc_table(insn, code)?,
            Luti2Vec | Luti4Vec | Luti4TwoVec => enc_luti_neon(insn, code)?,
            MoviVector | MvniVector | MoviScalarD | MoviVec2D | OrrVecImm | BicVecImm
            | FmovVecImmS | FmovVecImmH | FmovVecImmD2 => enc_modified_immediate(insn, code)?,
            SshrVec | UshrVec | SsraVec | UsraVec | SrshrVec | UrshrVec | SrsraVec | UrsraVec
            | SriVec | ShlVec | SliVec | SqshluImmVec | SqshlImmVec | UqshlImmVec | ShrnVec
            | Shrn2Vec | RshrnVec | Rshrn2Vec | SqshrnVec | Sqshrn2Vec | SqrshrnVec
            | Sqrshrn2Vec | SqshrunVec | Sqshrun2Vec | SqrshrunVec | Sqrshrun2Vec | UqshrnVec
            | Uqshrn2Vec | UqrshrnVec | Uqrshrn2Vec | SshllVec | Sshll2Vec | UshllVec
            | Ushll2Vec | SxtlVec | Sxtl2Vec | UxtlVec | Uxtl2Vec => {
                enc_shift_vector(insn, code)?
            }
            SshrScalar | UshrScalar | SsraScalar | UsraScalar | SrshrScalar | UrshrScalar
            | SrsraScalar | UrsraScalar | SriScalar | ShlScalar | SliScalar | SqshluImmScalar
            | SqshlImmScalar | UqshlImmScalar | SqshrnScalar | SqrshrnScalar | SqshrunScalar
            | SqrshrunScalar | UqshrnScalar | UqrshrnScalar | ScvtfFixedScalar
            | UcvtfFixedScalar | FcvtzsFixedScalar | FcvtzuFixedScalar => {
                enc_shift_scalar(insn, code)?
            }
            ScvtfFixedVec | UcvtfFixedVec | FcvtzsFixedVec | FcvtzuFixedVec => {
                enc_shift_vector(insn, code)?
            }
            _ => return Ok(None),
        };
        Ok(Some(w))
    }

    // =======================================================================
    // Copy.
    // =======================================================================

    /// Build the `imm5` element-size selector for `(esize_bits, index)` — inverse
    /// of `imm5_size_index`.
    fn imm5_for(esize: u32, index: u8) -> Result<u32, EncodeError> {
        let idx = index as u32;
        Ok(match esize {
            8 => {
                if idx > 0xf {
                    return Err(EncodeError::InvalidOperand);
                }
                (idx << 1) | 0b1
            }
            16 => {
                if idx > 0x7 {
                    return Err(EncodeError::InvalidOperand);
                }
                (idx << 2) | 0b10
            }
            32 => {
                if idx > 0x3 {
                    return Err(EncodeError::InvalidOperand);
                }
                (idx << 3) | 0b100
            }
            64 => {
                if idx > 0x1 {
                    return Err(EncodeError::InvalidOperand);
                }
                (idx << 4) | 0b1000
            }
            _ => return Err(EncodeError::InvalidOperand),
        })
    }

    /// Element width (bits) of a `.b/.h/.s/.d` indexed-element arrangement.
    fn elem_esize(a: VA) -> Result<u32, EncodeError> {
        Ok(match a {
            VA::V16B | VA::V8B => 8,
            VA::V8H | VA::V4H => 16,
            VA::V4S | VA::V2S => 32,
            VA::V2D | VA::V1D => 64,
            _ => return Err(EncodeError::InvalidOperand),
        })
    }

    /// Element width (bits) implied by a *destination* full arrangement
    /// (`.8b`/`.16b`/...). For DUP(element)/DUP(general).
    fn dst_esize(a: VA) -> Result<(u32, u32), EncodeError> {
        // returns (esize, q)
        Ok(match a {
            VA::V8B => (8, 0),
            VA::V16B => (8, 1),
            VA::V4H => (16, 0),
            VA::V8H => (16, 1),
            VA::V2S => (32, 0),
            VA::V4S => (32, 1),
            VA::V2D => (64, 1),
            _ => return Err(EncodeError::InvalidOperand),
        })
    }

    fn enc_copy(insn: &Instruction, code: Code) -> R {
        use Code::*;
        // Base: 0 Q op 01110000 imm5 0 imm4 1 Rn Rd.
        match code {
            DupElement => {
                // Vd.T, Vn.Ts[index]. imm5 from element size + index.
                let (esize, q) = dst_esize(arr_of(insn, 0)?)?;
                let index = lane_of(insn, 1)?;
                let imm5 = imm5_for(esize, index)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                Ok(copy_word(q, 0, imm5, 0b0000, rn, rd))
            }
            DupElementScalar => {
                // MOV <V><d>, Vn.Ts[index] (scalar copy, asisdone). esize/index
                // come from the source indexed-element operand. op=0, imm4=0000.
                let a = arr_of(insn, 1)?;
                let esize = elem_esize(a)?;
                let index = lane_of(insn, 1)?;
                let imm5 = imm5_for(esize, index)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                Ok(scalar_copy_word(0, imm5, 0b0000, rn, rd))
            }
            DupGeneral => {
                // Vd.T, R<n>. imm5 from element size; index bits are zero.
                let (esize, q) = dst_esize(arr_of(insn, 0)?)?;
                let imm5 = imm5_for(esize, 0)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                Ok(copy_word(q, 0, imm5, 0b0001, rn, rd))
            }
            InsGeneral => {
                // MOV Vd.Ts[index], R<n>. imm4=0011, op=0, Q irrelevant->1.
                let a = arr_of(insn, 0)?;
                let esize = elem_esize(a)?;
                let index = lane_of(insn, 0)?;
                let imm5 = imm5_for(esize, index)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                Ok(copy_word(1, 0, imm5, 0b0011, rn, rd))
            }
            Smov => {
                // SMOV Wd|Xd, Vn.Ts[index]. Q = X-width (1) vs W (0).
                let w = reg_of(insn, 0)?;
                let q = if w.width_bits() == 64 { 1u32 } else { 0 };
                let a = arr_of(insn, 1)?;
                let esize = elem_esize(a)?;
                let index = lane_of(insn, 1)?;
                let imm5 = imm5_for(esize, index)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                Ok(copy_word(q, 0, imm5, 0b0101, rn, rd))
            }
            Umov => {
                // UMOV/MOV Wd|Xd, Vn.Ts[index]. Q = X (1) vs W (0).
                let w = reg_of(insn, 0)?;
                let q = if w.width_bits() == 64 { 1u32 } else { 0 };
                let a = arr_of(insn, 1)?;
                let esize = elem_esize(a)?;
                let index = lane_of(insn, 1)?;
                let imm5 = imm5_for(esize, index)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                Ok(copy_word(q, 0, imm5, 0b0111, rn, rd))
            }
            InsElement => {
                // MOV Vd.Ts[index1], Vn.Ts[index2]. op=1, imm5 from dst, imm4 from
                // src index (shifted by log2(esize/8)).
                let a = arr_of(insn, 0)?;
                let esize = elem_esize(a)?;
                let dst_index = lane_of(insn, 0)?;
                let src_index = lane_of(insn, 1)?;
                let imm5 = imm5_for(esize, dst_index)?;
                let shift = match esize {
                    8 => 0,
                    16 => 1,
                    32 => 2,
                    _ => 3,
                };
                let imm4 = (src_index as u32) << shift;
                if imm4 > 0xf {
                    return Err(EncodeError::InvalidOperand);
                }
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                Ok(copy_word(1, 1, imm5, imm4, rn, rd))
            }
            _ => Err(EncodeError::Unsupported),
        }
    }

    /// Copy word skeleton: `0 Q op 01110000 imm5 0 imm4 1 Rn Rd`.
    #[inline]
    fn copy_word(q: u32, op: u32, imm5: u32, imm4: u32, rn: u32, rd: u32) -> u32 {
        (q << 30)
            | (op << 29)
            | (0b01110 << 24)
            // word<23:21>=000, imm5 at <20:16>.
            | (imm5 << 16)
            // word<15>=0, imm4 at <14:11>, word<10>=1.
            | (imm4 << 11)
            | (1 << 10)
            | (rn << 5)
            | rd
    }

    /// Scalar-copy word skeleton (`asisdone`): `01 op 1 11110 000 imm5 0 imm4 1
    /// Rn Rd`.
    #[inline]
    fn scalar_copy_word(op: u32, imm5: u32, imm4: u32, rn: u32, rd: u32) -> u32 {
        (0b01 << 30)
            | (op << 29)
            | (0b11110 << 24)
            // word<23:21>=000, imm5 at <20:16>.
            | (imm5 << 16)
            // word<15>=0, imm4 at <14:11>, word<10>=1.
            | (imm4 << 11)
            | (1 << 10)
            | (rn << 5)
            | rd
    }

    // =======================================================================
    // Permute.
    // =======================================================================

    fn enc_permute(insn: &Instruction, code: Code) -> R {
        use Code::*;
        let opcode = match code {
            Uzp1 => 0b001u32,
            Trn1 => 0b010,
            Zip1 => 0b011,
            Uzp2 => 0b101,
            Trn2 => 0b110,
            _ => 0b111, // Zip2
        };
        let a = arr_of(insn, 0)?;
        let size = size_of_arr(a)?;
        let q = q_of_arr(a)?;
        let rm = reg_num(insn, 2)?;
        let rn = reg_num(insn, 1)?;
        let rd = reg_num(insn, 0)?;
        // 0 Q 001110 size 0 Rm 0 opcode 10 Rn Rd.
        let word = (q << 30)
            | (0b001110 << 24)
            | (size << 22)
            | (rm << 16)
            | (opcode << 12)
            | (0b10 << 10)
            | (rn << 5)
            | rd;
        Ok(word)
    }

    // =======================================================================
    // Extract.
    // =======================================================================

    fn enc_ext(insn: &Instruction) -> R {
        let a = arr_of(insn, 0)?;
        let q = match a {
            VA::V8B => 0u32,
            VA::V16B => 1,
            _ => return Err(EncodeError::InvalidOperand),
        };
        let imm4 = imm_u(insn, 3)? as u32;
        if imm4 > 0xf || (q == 0 && (imm4 >> 3) & 1 == 1) {
            return Err(EncodeError::InvalidImmediate);
        }
        let rm = reg_num(insn, 2)?;
        let rn = reg_num(insn, 1)?;
        let rd = reg_num(insn, 0)?;
        // 0 Q 101110 00 0 Rm 0 imm4 0 Rn Rd.
        let word = (q << 30)
            | (0b101110 << 24)
            | (rm << 16)
            | (imm4 << 11)
            | (rn << 5)
            | rd;
        Ok(word)
    }

    // =======================================================================
    // Table lookup.
    // =======================================================================

    fn enc_table(insn: &Instruction, code: Code) -> R {
        let a = arr_of(insn, 0)?;
        let q = match a {
            VA::V8B => 0u32,
            VA::V16B => 1,
            _ => return Err(EncodeError::InvalidOperand),
        };
        let op = if code == Code::Tbx { 1u32 } else { 0 };
        // Register list is operand 1; its `count` gives len+1, first reg = Rn.
        let (rn, count) = match insn.op(1) {
            Operand::MultiReg { regs, count, .. } => (regs[0].number() as u32, count),
            _ => return Err(EncodeError::InvalidOperand),
        };
        if count == 0 || count > 4 {
            return Err(EncodeError::InvalidOperand);
        }
        let len = (count - 1) as u32;
        let rm = reg_num(insn, 2)?;
        let rd = reg_num(insn, 0)?;
        // 0 Q 001110 000 Rm 0 len op 00 Rn Rd.
        let word = (q << 30)
            | (0b001110 << 24)
            | (rm << 16)
            | (len << 13)
            | (op << 12)
            | (rn << 5)
            | rd;
        Ok(word)
    }

    /// LUTI2/LUTI4 (FEAT_LUT, NEON) — inverse of `decode_luti_neon`. Layout:
    /// `0 1 0 01110 size 0 Rm 0 <14:12> 00 Rn Rd` (Q fixed to 1). `size` and the
    /// `<14:12>` field are recovered from the destination arrangement (op0), the
    /// `Code` and the table index (the lane on op2).
    fn enc_luti_neon(insn: &Instruction, code: Code) -> R {
        use Code::*;
        let rd = reg_num(insn, 0)?;
        // Table register-list operand (op1): first register number.
        let rn = match insn.op(1) {
            Operand::MultiReg { regs, .. } => regs[0].number() as u32,
            _ => return Err(EncodeError::InvalidOperand),
        };
        // Vector-element selector (op2): Vm register + bracketed table index.
        let rm = reg_num(insn, 2)?;
        let index = lane_of(insn, 2)? as u32;
        let dst = arr_of(insn, 0)?;
        let (size, b14, b13, b12) = match code {
            Luti2Vec => match dst {
                VA::V16B => (0b10u32, (index >> 1) & 1, index & 1, 1u32),
                VA::V8H => (0b11, (index >> 2) & 1, (index >> 1) & 1, index & 1),
                _ => return Err(EncodeError::InvalidOperand),
            },
            // .16b single-register table: 1-bit index at <14>; <13>=1, <12>=0.
            Luti4Vec => (0b01, index & 1, 1, 0),
            // .8h two-register table: 2-bit index at <14:13>; <12>=1.
            Luti4TwoVec => (0b01, (index >> 1) & 1, index & 1, 1),
            _ => return Err(EncodeError::Unsupported),
        };
        let word = (1u32 << 30) // Q == 1 (128-bit only)
            | (0b01110 << 24)
            | (size << 22)
            | (rm << 16)
            | (b14 << 14)
            | (b13 << 13)
            | (b12 << 12)
            | (rn << 5)
            | rd;
        Ok(word)
    }

    // =======================================================================
    // Modified immediate.
    // =======================================================================

    fn enc_modified_immediate(insn: &Instruction, code: Code) -> R {
        use Code::*;
        // Base: 0 Q op 0111100000 abc cmode o2 1 defgh Rn Rd.
        // We recover (op, cmode, o2, imm8, Q) and pack.
        let rd = reg_num(insn, 0)?;
        let (q, op, cmode, o2, imm8) = match code {
            // MOVI byte (cmode 1110, op 0, o2 0) OR MOVI shifted/MSL forms OR
            // MVNI forms — distinguished by the operand kind.
            MoviVector | MvniVector => recover_movi_mvni(insn, code)?,
            // 64-bit MOVI per-byte: scalar Dd or .2d, cmode=1110, op=1.
            MoviScalarD => {
                let val = imm_u(insn, 1)?;
                let imm8 = encode_advsimd_movi64(val).ok_or(EncodeError::InvalidImmediate)?;
                (0u32, 1u32, 0b1110u32, 0u32, imm8)
            }
            MoviVec2D => {
                let val = imm_u(insn, 1)?;
                let imm8 = encode_advsimd_movi64(val).ok_or(EncodeError::InvalidImmediate)?;
                (1u32, 1u32, 0b1110u32, 0u32, imm8)
            }
            // ORR/BIC vector immediate (cmode<0>==1): 16/32-bit family.
            OrrVecImm | BicVecImm => recover_orr_bic(insn, code)?,
            // FMOV vector immediate.
            FmovVecImmS => {
                let a = arr_of(insn, 0)?;
                let q = match a {
                    VA::V2S => 0u32,
                    VA::V4S => 1,
                    _ => return Err(EncodeError::InvalidOperand),
                };
                let f = fpimm_of(insn, 1)?;
                let imm8 = encode_vfp_imm(f, 32).ok_or(EncodeError::InvalidImmediate)?;
                (q, 0, 0b1111, 0, imm8)
            }
            FmovVecImmH => {
                let a = arr_of(insn, 0)?;
                let q = match a {
                    VA::V4H => 0u32,
                    VA::V8H => 1,
                    _ => return Err(EncodeError::InvalidOperand),
                };
                let f = fpimm_of(insn, 1)?;
                let imm8 = encode_vfp_imm(f, 16).ok_or(EncodeError::InvalidImmediate)?;
                (q, 0, 0b1111, 1, imm8)
            }
            FmovVecImmD2 => {
                let f = fpimm_of(insn, 1)?;
                let imm8 = encode_vfp_imm(f, 64).ok_or(EncodeError::InvalidImmediate)?;
                (1u32, 1, 0b1111, 0, imm8)
            }
            _ => return Err(EncodeError::Unsupported),
        };

        if imm8 > 0xff {
            return Err(EncodeError::InvalidImmediate);
        }
        let abc = (imm8 >> 5) & 0b111;
        let defgh = imm8 & 0b11111;
        let word = (q << 30)
            | (op << 29)
            | (0b0111100000 << 19)
            | (abc << 16)
            | (cmode << 12)
            | (o2 << 11)
            | (1 << 10)
            | (defgh << 5)
            | rd;
        Ok(word)
    }

    /// Recover `(q, op, cmode, o2, imm8)` for a MOVI/MVNI vector immediate from
    /// the operand (which is `ImmShiftedMove{imm8,lsl}`, `ImmShiftedMsl{imm8,
    /// msl}`, or a plain `ImmUnsigned` byte for the cmode==1110 byte form).
    fn recover_movi_mvni(insn: &Instruction, code: Code) -> Result<(u32, u32, u32, u32, u32), EncodeError> {
        let op = if code == Code::MvniVector { 1u32 } else { 0 };
        let a = arr_of(insn, 0)?;
        match insn.op(1) {
            // Plain byte immediate: MOVI byte (cmode 1110), only for MOVI (op==0),
            // arrangement .8b/.16b.
            Operand::ImmUnsigned(v) => {
                let q = match a {
                    VA::V8B => 0u32,
                    VA::V16B => 1,
                    _ => return Err(EncodeError::InvalidOperand),
                };
                if op != 0 || v > 0xff {
                    return Err(EncodeError::InvalidImmediate);
                }
                Ok((q, 0, 0b1110, 0, v as u32))
            }
            // Shifted (LSL) immediate: 16-bit (.4h/.8h) or 32-bit (.2s/.4s).
            Operand::ImmShiftedMove { imm, lsl } => {
                let imm8 = imm as u32;
                match a {
                    VA::V4H | VA::V8H => {
                        let q = if a == VA::V8H { 1u32 } else { 0 };
                        // cmode = 10 x0 ; cmode<1>=lsl/8 (0 or 1), cmode<0>=0.
                        let shbit = match lsl {
                            0 => 0u32,
                            8 => 1,
                            _ => return Err(EncodeError::InvalidImmediate),
                        };
                        let cmode = 0b1000 | (shbit << 1);
                        Ok((q, op, cmode, 0, imm8))
                    }
                    VA::V2S | VA::V4S => {
                        let q = if a == VA::V4S { 1u32 } else { 0 };
                        // cmode = 0 xx 0 ; cmode<2:1> = lsl/8 (0/1/2/3).
                        let amt = match lsl {
                            0 => 0u32,
                            8 => 1,
                            16 => 2,
                            24 => 3,
                            _ => return Err(EncodeError::InvalidImmediate),
                        };
                        let cmode = amt << 1;
                        Ok((q, op, cmode, 0, imm8))
                    }
                    _ => Err(EncodeError::InvalidOperand),
                }
            }
            // MSL immediate: 32-bit family (.2s/.4s), cmode = 110 x.
            Operand::ImmShiftedMsl { imm, msl } => {
                let q = match a {
                    VA::V2S => 0u32,
                    VA::V4S => 1,
                    _ => return Err(EncodeError::InvalidOperand),
                };
                let cbit = match msl {
                    8 => 0u32,
                    16 => 1,
                    _ => return Err(EncodeError::InvalidImmediate),
                };
                let cmode = 0b1100 | cbit;
                Ok((q, op, cmode, 0, imm as u32))
            }
            _ => Err(EncodeError::InvalidOperand),
        }
    }

    /// Recover `(q, op, cmode, o2, imm8)` for ORR/BIC vector immediate.
    fn recover_orr_bic(insn: &Instruction, code: Code) -> Result<(u32, u32, u32, u32, u32), EncodeError> {
        let op = if code == Code::BicVecImm { 1u32 } else { 0 };
        let a = arr_of(insn, 0)?;
        let (imm, lsl) = match insn.op(1) {
            Operand::ImmShiftedMove { imm, lsl } => (imm as u32, lsl),
            _ => return Err(EncodeError::InvalidOperand),
        };
        match a {
            VA::V4H | VA::V8H => {
                let q = if a == VA::V8H { 1u32 } else { 0 };
                let shbit = match lsl {
                    0 => 0u32,
                    8 => 1,
                    _ => return Err(EncodeError::InvalidImmediate),
                };
                // cmode = 10 x1 ; cmode<0>=1 (ORR/BIC).
                let cmode = 0b1001 | (shbit << 1);
                Ok((q, op, cmode, 0, imm))
            }
            VA::V2S | VA::V4S => {
                let q = if a == VA::V4S { 1u32 } else { 0 };
                let amt = match lsl {
                    0 => 0u32,
                    8 => 1,
                    16 => 2,
                    24 => 3,
                    _ => return Err(EncodeError::InvalidImmediate),
                };
                // cmode = 0 xx 1 ; cmode<0>=1.
                let cmode = (amt << 1) | 1;
                Ok((q, op, cmode, 0, imm))
            }
            _ => Err(EncodeError::InvalidOperand),
        }
    }

    // =======================================================================
    // Shift by immediate (vector).
    // =======================================================================

    /// Encode `(immh, immb)` for a left-shift form: `immh:immb = esize + shift`.
    fn left_immhb(esize: u32, shift: u32) -> Result<(u32, u32), EncodeError> {
        let val = esize + shift;
        immhb_from_val(esize, val)
    }

    /// Encode `(immh, immb)` for a right-shift form: `immh:immb = 2*esize - shift`.
    fn right_immhb(esize: u32, shift: u32) -> Result<(u32, u32), EncodeError> {
        if shift == 0 || shift > 2 * esize {
            return Err(EncodeError::InvalidImmediate);
        }
        let val = 2 * esize - shift;
        immhb_from_val(esize, val)
    }

    /// Split a 7-bit `immh:immb` value, validating the `immh` high bit lands in
    /// the right element-size band.
    fn immhb_from_val(esize: u32, val: u32) -> Result<(u32, u32), EncodeError> {
        if val == 0 || val > 0x7f {
            return Err(EncodeError::InvalidImmediate);
        }
        let immh = (val >> 3) & 0xf;
        let immb = val & 0x7;
        // Confirm immh's band matches esize (so decode picks the same element).
        let ok = match esize {
            8 => immh == 1,
            16 => (0b0010..=0b0011).contains(&immh),
            32 => (0b0100..=0b0111).contains(&immh),
            64 => (0b1000..=0b1111).contains(&immh),
            _ => false,
        };
        if !ok {
            return Err(EncodeError::InvalidImmediate);
        }
        Ok((immh, immb))
    }

    /// Arrangement -> (esize, q) for a same-size shift.
    fn shift_arr_esize(a: VA) -> Result<(u32, u32), EncodeError> {
        Ok(match a {
            VA::V8B => (8, 0),
            VA::V16B => (8, 1),
            VA::V4H => (16, 0),
            VA::V8H => (16, 1),
            VA::V2S => (32, 0),
            VA::V4S => (32, 1),
            VA::V2D => (64, 1),
            _ => return Err(EncodeError::InvalidOperand),
        })
    }

    fn enc_shift_vector(insn: &Instruction, code: Code) -> R {
        use Code::*;
        // (U, opcode5, dir) where dir: 'r'=right same, 'l'=left same.
        // Narrowing, long, and fixed-cvt are handled by dedicated arms.
        match code {
            // Narrowing shift-right.
            ShrnVec | Shrn2Vec | RshrnVec | Rshrn2Vec | SqshrnVec | Sqshrn2Vec | SqrshrnVec
            | Sqrshrn2Vec | SqshrunVec | Sqshrun2Vec | SqrshrunVec | Sqrshrun2Vec | UqshrnVec
            | Uqshrn2Vec | UqrshrnVec | Uqrshrn2Vec => enc_narrow_vec(insn, code),
            // Long shift-left + SXTL/UXTL.
            SshllVec | Sshll2Vec | UshllVec | Ushll2Vec | SxtlVec | Sxtl2Vec | UxtlVec
            | Uxtl2Vec => enc_shll_vec(insn, code),
            // Fixed-point convert.
            ScvtfFixedVec | UcvtfFixedVec | FcvtzsFixedVec | FcvtzuFixedVec => {
                let (u, opcode) = match code {
                    ScvtfFixedVec => (0u32, 0b11100u32),
                    UcvtfFixedVec => (1, 0b11100),
                    FcvtzsFixedVec => (0, 0b11111),
                    _ => (1, 0b11111),
                };
                let a = arr_of(insn, 0)?;
                let (esize, q) = shift_arr_esize(a)?;
                if esize == 8 {
                    return Err(EncodeError::InvalidOperand);
                }
                let shift = imm_u(insn, 2)? as u32;
                let (immh, immb) = right_immhb(esize, shift)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                Ok(shift_word(q, u, immh, immb, opcode, rn, rd))
            }
            // Right-shift same-size.
            SshrVec | UshrVec | SsraVec | UsraVec | SrshrVec | UrshrVec | SrsraVec | UrsraVec
            | SriVec => {
                let (u, opcode) = match code {
                    SshrVec => (0u32, 0b00000u32),
                    UshrVec => (1, 0b00000),
                    SsraVec => (0, 0b00010),
                    UsraVec => (1, 0b00010),
                    SrshrVec => (0, 0b00100),
                    UrshrVec => (1, 0b00100),
                    SrsraVec => (0, 0b00110),
                    UrsraVec => (1, 0b00110),
                    _ => (1, 0b01000), // SriVec
                };
                let a = arr_of(insn, 0)?;
                let (esize, q) = shift_arr_esize(a)?;
                let shift = imm_u(insn, 2)? as u32;
                let (immh, immb) = right_immhb(esize, shift)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                Ok(shift_word(q, u, immh, immb, opcode, rn, rd))
            }
            // Left-shift same-size.
            ShlVec | SliVec | SqshluImmVec | SqshlImmVec | UqshlImmVec => {
                let (u, opcode) = match code {
                    ShlVec => (0u32, 0b01010u32),
                    SliVec => (1, 0b01010),
                    SqshluImmVec => (1, 0b01100),
                    SqshlImmVec => (0, 0b01110),
                    _ => (1, 0b01110), // UqshlImmVec
                };
                let a = arr_of(insn, 0)?;
                let (esize, q) = shift_arr_esize(a)?;
                let shift = imm_u(insn, 2)? as u32;
                let (immh, immb) = left_immhb(esize, shift)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                Ok(shift_word(q, u, immh, immb, opcode, rn, rd))
            }
            _ => Err(EncodeError::Unsupported),
        }
    }

    /// Vector shift word skeleton: `0 Q U 011110 immh immb opcode 1 Rn Rd`.
    #[inline]
    fn shift_word(q: u32, u: u32, immh: u32, immb: u32, opcode: u32, rn: u32, rd: u32) -> u32 {
        (q << 30)
            | (u << 29)
            | (0b011110 << 23)
            | (immh << 19)
            | (immb << 16)
            | (opcode << 11)
            | (1 << 10)
            | (rn << 5)
            | rd
    }

    /// Narrowing shift-right (vector): dst Tb (narrow), src Ta (wide). size from
    /// dst element; `shift = 2*dst_esize - immh:immb`.
    fn enc_narrow_vec(insn: &Instruction, code: Code) -> R {
        use Code::*;
        let (u, opcode, two) = match code {
            ShrnVec => (0u32, 0b10000u32, false),
            Shrn2Vec => (0, 0b10000, true),
            RshrnVec => (0, 0b10001, false),
            Rshrn2Vec => (0, 0b10001, true),
            SqshrnVec => (0, 0b10010, false),
            Sqshrn2Vec => (0, 0b10010, true),
            SqrshrnVec => (0, 0b10011, false),
            Sqrshrn2Vec => (0, 0b10011, true),
            SqshrunVec => (1, 0b10000, false),
            Sqshrun2Vec => (1, 0b10000, true),
            SqrshrunVec => (1, 0b10001, false),
            Sqrshrun2Vec => (1, 0b10001, true),
            UqshrnVec => (1, 0b10010, false),
            Uqshrn2Vec => (1, 0b10010, true),
            UqrshrnVec => (1, 0b10011, false),
            _ => (1, 0b10011, true), // Uqrshrn2Vec
        };
        let dst = arr_of(insn, 0)?;
        let dst_esize = match dst {
            VA::V8B | VA::V16B => 8u32,
            VA::V4H | VA::V8H => 16,
            VA::V2S | VA::V4S => 32,
            _ => return Err(EncodeError::InvalidOperand),
        };
        let q = if two { 1u32 } else { 0 };
        let shift = imm_u(insn, 2)? as u32;
        // immh:immb = 2*dst_esize - shift, with immh band = dst_esize.
        let (immh, immb) = narrow_immhb(dst_esize, shift)?;
        let rn = reg_num(insn, 1)?;
        let rd = reg_num(insn, 0)?;
        Ok(shift_word(q, u, immh, immb, opcode, rn, rd))
    }

    /// `immh:immb` for a narrowing form (immh<3> must be 0).
    fn narrow_immhb(dst_esize: u32, shift: u32) -> Result<(u32, u32), EncodeError> {
        if shift == 0 || shift > 2 * dst_esize {
            return Err(EncodeError::InvalidImmediate);
        }
        let val = 2 * dst_esize - shift;
        let immh = (val >> 3) & 0xf;
        let immb = val & 0x7;
        let ok = match dst_esize {
            8 => immh == 1,
            16 => (0b0010..=0b0011).contains(&immh),
            32 => (0b0100..=0b0111).contains(&immh),
            _ => false,
        };
        if !ok || (immh >> 3) & 1 == 1 {
            return Err(EncodeError::InvalidImmediate);
        }
        Ok((immh, immb))
    }

    /// SSHLL/USHLL + SXTL/UXTL: dst Ta (long), src Tb (narrow). size from src
    /// element; `immh:immb = src_esize + shift`.
    fn enc_shll_vec(insn: &Instruction, code: Code) -> R {
        use Code::*;
        let (u, two, is_xtl) = match code {
            SshllVec => (0u32, false, false),
            Sshll2Vec => (0, true, false),
            UshllVec => (1, false, false),
            Ushll2Vec => (1, true, false),
            SxtlVec => (0, false, true),
            Sxtl2Vec => (0, true, true),
            UxtlVec => (1, false, true),
            _ => (1, true, true), // Uxtl2Vec
        };
        let src = arr_of(insn, 1)?;
        let src_esize = match src {
            VA::V8B | VA::V16B => 8u32,
            VA::V4H | VA::V8H => 16,
            VA::V2S | VA::V4S => 32,
            _ => return Err(EncodeError::InvalidOperand),
        };
        let q = if two { 1u32 } else { 0 };
        let shift = if is_xtl { 0 } else { imm_u(insn, 2)? as u32 };
        let (immh, immb) = left_immhb(src_esize, shift)?;
        let rn = reg_num(insn, 1)?;
        let rd = reg_num(insn, 0)?;
        Ok(shift_word(q, u, immh, immb, 0b10100, rn, rd))
    }

    // =======================================================================
    // Shift by immediate (scalar).
    // =======================================================================

    fn enc_shift_scalar(insn: &Instruction, code: Code) -> R {
        use Code::*;
        match code {
            // Right-shift same-size (D only).
            SshrScalar | UshrScalar | SsraScalar | UsraScalar | SrshrScalar | UrshrScalar
            | SrsraScalar | UrsraScalar | SriScalar => {
                let (u, opcode) = match code {
                    SshrScalar => (0u32, 0b00000u32),
                    UshrScalar => (1, 0b00000),
                    SsraScalar => (0, 0b00010),
                    UsraScalar => (1, 0b00010),
                    SrshrScalar => (0, 0b00100),
                    UrshrScalar => (1, 0b00100),
                    SrsraScalar => (0, 0b00110),
                    UrsraScalar => (1, 0b00110),
                    _ => (1, 0b01000), // SriScalar
                };
                let eb = scalar_width(insn, 0)?;
                if eb != 64 {
                    return Err(EncodeError::InvalidOperand);
                }
                let shift = imm_u(insn, 2)? as u32;
                let (immh, immb) = right_immhb(64, shift)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                Ok(scalar_shift_word(u, immh, immb, opcode, rn, rd))
            }
            // Left-shift D-only (SHL/SLI).
            ShlScalar | SliScalar => {
                let u = if code == SliScalar { 1u32 } else { 0 };
                let eb = scalar_width(insn, 0)?;
                if eb != 64 {
                    return Err(EncodeError::InvalidOperand);
                }
                let shift = imm_u(insn, 2)? as u32;
                let (immh, immb) = left_immhb(64, shift)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                Ok(scalar_shift_word(u, immh, immb, 0b01010, rn, rd))
            }
            // Saturating left-shift, any size.
            SqshluImmScalar | SqshlImmScalar | UqshlImmScalar => {
                let (u, opcode) = match code {
                    SqshluImmScalar => (1u32, 0b01100u32),
                    SqshlImmScalar => (0, 0b01110),
                    _ => (1, 0b01110), // UqshlImmScalar
                };
                let eb = scalar_width(insn, 0)? as u32;
                let shift = imm_u(insn, 2)? as u32;
                let (immh, immb) = left_immhb(eb, shift)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                Ok(scalar_shift_word(u, immh, immb, opcode, rn, rd))
            }
            // Narrowing saturating shift-right.
            SqshrnScalar | SqrshrnScalar | SqshrunScalar | SqrshrunScalar | UqshrnScalar
            | UqrshrnScalar => {
                let (u, opcode) = match code {
                    SqshrnScalar => (0u32, 0b10010u32),
                    SqrshrnScalar => (0, 0b10011),
                    SqshrunScalar => (1, 0b10000),
                    SqrshrunScalar => (1, 0b10001),
                    UqshrnScalar => (1, 0b10010),
                    _ => (1, 0b10011), // UqrshrnScalar
                };
                // dst = half-width scalar (operand0), src = full width (operand1).
                let dst_eb = scalar_width(insn, 0)? as u32;
                let shift = imm_u(insn, 2)? as u32;
                let (immh, immb) = narrow_immhb(dst_eb, shift)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                Ok(scalar_shift_word(u, immh, immb, opcode, rn, rd))
            }
            // Fixed-point convert.
            ScvtfFixedScalar | UcvtfFixedScalar | FcvtzsFixedScalar | FcvtzuFixedScalar => {
                let (u, opcode) = match code {
                    ScvtfFixedScalar => (0u32, 0b11100u32),
                    UcvtfFixedScalar => (1, 0b11100),
                    FcvtzsFixedScalar => (0, 0b11111),
                    _ => (1, 0b11111),
                };
                let eb = scalar_width(insn, 0)? as u32;
                if eb == 8 {
                    return Err(EncodeError::InvalidOperand);
                }
                let shift = imm_u(insn, 2)? as u32;
                let (immh, immb) = right_immhb(eb, shift)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                Ok(scalar_shift_word(u, immh, immb, opcode, rn, rd))
            }
            _ => Err(EncodeError::Unsupported),
        }
    }

    /// Scalar shift word: `0 1 U 111110 immh immb opcode 1 Rn Rd`.
    #[inline]
    fn scalar_shift_word(u: u32, immh: u32, immb: u32, opcode: u32, rn: u32, rd: u32) -> u32 {
        (1 << 30)
            | (u << 29)
            | (0b111110 << 23)
            | (immh << 19)
            | (immb << 16)
            | (opcode << 11)
            | (1 << 10)
            | (rn << 5)
            | rd
    }
}
