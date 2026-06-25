// Included into `simd_fp.rs` — Scalar floating-point data-processing encoders.
//
// Inverse of `crate::decode::simd_fp::scalar_fp`. The precision (`ftype`) is
// recovered from the per-precision `Code` suffix (S/D/H); register numbers,
// GP width, immediates, conditions and the `#fbits` are read from the operands.

mod scalar_fp {
    use super::*;

    /// Scalar-FP precision, recovered from the `Code` suffix.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum P {
        S,
        D,
        H,
    }

    impl P {
        /// `ftype` field value.
        #[inline]
        fn ftype(self) -> u32 {
            match self {
                P::S => 0b00,
                P::D => 0b01,
                P::H => 0b11,
            }
        }
    }

    /// Encode a scalar-FP instruction. Returns `Ok(None)` if `code` is not a
    /// scalar-FP code (so the dispatcher tries the next family).
    pub(super) fn encode(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
        use Code::*;
        let w = match code {
            // --- fp<->fixed ---
            ScvtfFixedS32 | ScvtfFixedS64 | ScvtfFixedD32 | ScvtfFixedD64 | ScvtfFixedH32
            | ScvtfFixedH64 | UcvtfFixedS32 | UcvtfFixedS64 | UcvtfFixedD32 | UcvtfFixedD64
            | UcvtfFixedH32 | UcvtfFixedH64 | FcvtzsFixedS32 | FcvtzsFixedS64 | FcvtzsFixedD32
            | FcvtzsFixedD64 | FcvtzsFixedH32 | FcvtzsFixedH64 | FcvtzuFixedS32 | FcvtzuFixedS64
            | FcvtzuFixedD32 | FcvtzuFixedD64 | FcvtzuFixedH32 | FcvtzuFixedH64 => {
                enc_float2fix(insn, code)?
            }
            // --- fp<->int (FCVT*/SCVTF/UCVTF) ---
            ScvtfS32 | ScvtfS64 | ScvtfD32 | ScvtfD64 | ScvtfH32 | ScvtfH64 | UcvtfS32
            | UcvtfS64 | UcvtfD32 | UcvtfD64 | UcvtfH32 | UcvtfH64 | FcvtzsScalarS32
            | FcvtzsScalarS64 | FcvtzsScalarD32 | FcvtzsScalarD64 | FcvtzsScalarH32
            | FcvtzsScalarH64 | FcvtzuScalarS32 | FcvtzuScalarS64 | FcvtzuScalarD32
            | FcvtzuScalarD64 | FcvtzuScalarH32 | FcvtzuScalarH64 | FcvtnsScalar | FcvtnuScalar
            | FcvtasScalar | FcvtauScalar | FcvtpsScalar | FcvtpuScalar | FcvtmsScalar
            | FcvtmuScalar => enc_float2int(insn, code)?,
            Fjcvtzs => enc_fjcvtzs(insn)?,
            // --- FMOV general<->FP and top-half ---
            FmovToGp32 | FmovFromGp32 | FmovToGp64 | FmovFromGp64 | FmovToGpH32 | FmovFromGpH32
            | FmovToGpH64 | FmovFromGpH64 => enc_fmov_general(insn, code)?,
            FmovTopToGp | FmovTopFromGp => enc_fmov_top(insn, code)?,
            // --- dp1 ---
            FmovS | FmovD | FmovH | FabsS | FabsD | FabsH | FnegS | FnegD | FnegH | FsqrtS
            | FsqrtD | FsqrtH | FrintnS | FrintnD | FrintnH | FrintpS | FrintpD | FrintpH
            | FrintmS | FrintmD | FrintmH | FrintzS | FrintzD | FrintzH | FrintaS | FrintaD
            | FrintaH | FrintxS | FrintxD | FrintxH | FrintiS | FrintiD | FrintiH | Frint32zS
            | Frint32zD | Frint32xS | Frint32xD | Frint64zS | Frint64zD | Frint64xS
            | Frint64xD => enc_dp1_same(insn, code)?,
            FcvtSD | FcvtSH | FcvtDS | FcvtDH | FcvtHS | FcvtHD => enc_fcvt(insn, code)?,
            Bfcvt => enc_bfcvt(insn)?,
            // --- dp2 ---
            FmulS | FmulD | FmulH | FdivS | FdivD | FdivH | FaddS | FaddD | FaddH | FsubS
            | FsubD | FsubH | FmaxS | FmaxD | FmaxH | FminS | FminD | FminH | FmaxnmS
            | FmaxnmD | FmaxnmH | FminnmS | FminnmD | FminnmH | FnmulS | FnmulD | FnmulH => {
                enc_dp2(insn, code)?
            }
            // --- dp3 ---
            FmaddS | FmaddD | FmaddH | FmsubS | FmsubD | FmsubH | FnmaddS | FnmaddD | FnmaddH
            | FnmsubS | FnmsubD | FnmsubH => enc_dp3(insn, code)?,
            // --- compare ---
            FcmpS | FcmpD | FcmpH | FcmpeS | FcmpeD | FcmpeH => enc_cmp(insn, code)?,
            // --- ccmp ---
            FccmpS | FccmpD | FccmpH | FccmpeS | FccmpeD | FccmpeH => enc_ccmp(insn, code)?,
            // --- csel ---
            FcselS | FcselD | FcselH => enc_csel(insn, code)?,
            // --- imm ---
            FmovImmS | FmovImmD | FmovImmH => enc_imm(insn, code)?,
            _ => return Ok(None),
        };
        Ok(Some(w))
    }

    // -----------------------------------------------------------------------
    // Code -> (precision, sf, op-fields) recovery tables.
    // -----------------------------------------------------------------------

    /// `(precision, sf)` for a fixed-point convert code.
    fn fix_p_sf(code: Code) -> (P, u32) {
        use Code::*;
        match code {
            ScvtfFixedS32 | UcvtfFixedS32 | FcvtzsFixedS32 | FcvtzuFixedS32 => (P::S, 0),
            ScvtfFixedS64 | UcvtfFixedS64 | FcvtzsFixedS64 | FcvtzuFixedS64 => (P::S, 1),
            ScvtfFixedD32 | UcvtfFixedD32 | FcvtzsFixedD32 | FcvtzuFixedD32 => (P::D, 0),
            ScvtfFixedD64 | UcvtfFixedD64 | FcvtzsFixedD64 | FcvtzuFixedD64 => (P::D, 1),
            ScvtfFixedH32 | UcvtfFixedH32 | FcvtzsFixedH32 | FcvtzuFixedH32 => (P::H, 0),
            _ => (P::H, 1),
        }
    }

    /// `(precision, sf)` for an fp<->int convert code (the `*Scalar*` family and
    /// SCVTF/UCVTF int).
    fn int_p_sf(code: Code) -> (P, u32) {
        use Code::*;
        match code {
            ScvtfS32 | UcvtfS32 | FcvtzsScalarS32 | FcvtzuScalarS32 => (P::S, 0),
            ScvtfS64 | UcvtfS64 | FcvtzsScalarS64 | FcvtzuScalarS64 => (P::S, 1),
            ScvtfD32 | UcvtfD32 | FcvtzsScalarD32 | FcvtzuScalarD32 => (P::D, 0),
            ScvtfD64 | UcvtfD64 | FcvtzsScalarD64 | FcvtzuScalarD64 => (P::D, 1),
            ScvtfH32 | UcvtfH32 | FcvtzsScalarH32 | FcvtzuScalarH32 => (P::H, 0),
            ScvtfH64 | UcvtfH64 | FcvtzsScalarH64 | FcvtzuScalarH64 => (P::H, 1),
            // FCVT{N,A,P,M}{S,U} carry sf from the GP operand; precision is via
            // the scalar source. Resolve sf later from operands; provide a
            // precision placeholder selected by caller. These are handled
            // separately so this arm is unreachable for them.
            _ => (P::S, 0),
        }
    }

    /// Precision of a per-precision `Code` whose name ends in S/D/H (dp1/dp2/dp3
    /// /compare/ccmp/csel/imm). Returns `None` if the code is not in that family.
    fn suffix_prec(code: Code) -> Option<P> {
        use Code::*;
        let p = match code {
            FmovS | FabsS | FnegS | FsqrtS | FrintnS | FrintpS | FrintmS | FrintzS | FrintaS
            | FrintxS | FrintiS | Frint32zS | Frint32xS | Frint64zS | Frint64xS | FmulS | FdivS
            | FaddS | FsubS | FmaxS | FminS | FmaxnmS | FminnmS | FnmulS | FmaddS | FmsubS
            | FnmaddS | FnmsubS | FcmpS | FcmpeS | FccmpS | FccmpeS | FcselS | FmovImmS => P::S,
            FmovD | FabsD | FnegD | FsqrtD | FrintnD | FrintpD | FrintmD | FrintzD | FrintaD
            | FrintxD | FrintiD | Frint32zD | Frint32xD | Frint64zD | Frint64xD | FmulD | FdivD
            | FaddD | FsubD | FmaxD | FminD | FmaxnmD | FminnmD | FnmulD | FmaddD | FmsubD
            | FnmaddD | FnmsubD | FcmpD | FcmpeD | FccmpD | FccmpeD | FcselD | FmovImmD => P::D,
            FmovH | FabsH | FnegH | FsqrtH | FrintnH | FrintpH | FrintmH | FrintzH | FrintaH
            | FrintxH | FrintiH | FmulH | FdivH | FaddH | FsubH | FmaxH | FminH | FmaxnmH
            | FminnmH | FnmulH | FmaddH | FmsubH | FnmaddH | FnmsubH | FcmpH | FcmpeH | FccmpH
            | FccmpeH | FcselH | FmovImmH => P::H,
            _ => return None,
        };
        Some(p)
    }

    // -----------------------------------------------------------------------
    // Encoders.
    // -----------------------------------------------------------------------

    /// Base word for the scalar-FP `M=0 0 S=0 11110 ftype 1 ...` skeleton with a
    /// given `ftype`. Bits 24..21 are `0b11110` then bit21 set is part of the
    /// "floating-point" subgroup signature shared by these encodings.
    #[inline]
    fn fp_base(ftype: u32) -> u32 {
        // 0_0_0_11110_ftype_1_... : word<28:24>=11110, word<21>=1.
        (0b11110 << 24) | (ftype << 22) | (1 << 21)
    }

    /// fp<->fixed: `sf 0 0 11110 ftype 0 rmode opcode scale Rn Rd` (word<21>=0).
    fn enc_float2fix(insn: &Instruction, code: Code) -> R {
        use Code::*;
        let (p, sf) = fix_p_sf(code);
        let (rmode, opcode, int_to_fp) = match code {
            ScvtfFixedS32 | ScvtfFixedS64 | ScvtfFixedD32 | ScvtfFixedD64 | ScvtfFixedH32
            | ScvtfFixedH64 => (0b00u32, 0b010u32, true),
            UcvtfFixedS32 | UcvtfFixedS64 | UcvtfFixedD32 | UcvtfFixedD64 | UcvtfFixedH32
            | UcvtfFixedH64 => (0b00, 0b011, true),
            FcvtzsFixedS32 | FcvtzsFixedS64 | FcvtzsFixedD32 | FcvtzsFixedD64 | FcvtzsFixedH32
            | FcvtzsFixedH64 => (0b11, 0b000, false),
            _ => (0b11, 0b001, false), // FCVTZU fixed
        };
        // #fbits = 64 - scale; operand is at index 2.
        let fbits = imm_u(insn, 2)?;
        if fbits == 0 || fbits > 64 {
            return Err(EncodeError::InvalidImmediate);
        }
        let scale = 64 - fbits as u32;
        // 32-bit forms require scale<5>==1.
        if sf == 0 && (scale >> 5) & 1 == 0 {
            return Err(EncodeError::InvalidImmediate);
        }
        let (rd, rn) = if int_to_fp {
            // SCVTF/UCVTF Sd/Dd/Hd, Wn/Xn, #fbits.
            (fp_dst(insn, 0, p)?, reg_num(insn, 1)?)
        } else {
            // FCVTZ* Wd/Xd, Sn/Dn/Hn, #fbits.
            (reg_num(insn, 0)?, fp_src(insn, 1, p)?)
        };
        let word = (sf << 31)
            | (0b11110 << 24)
            | (p.ftype() << 22)
            // word<21>=0 (fixed-point), rmode at <20:19>, opcode at <18:16>.
            | (rmode << 19)
            | (opcode << 16)
            | (scale << 10)
            | (rn << 5)
            | rd;
        Ok(word)
    }

    /// fp<->int (FCVT*/SCVTF/UCVTF int): `sf 0 0 11110 ftype 1 rmode opcode 000000 Rn Rd`.
    fn enc_float2int(insn: &Instruction, code: Code) -> R {
        use Code::*;
        // FCVT{N,A,P,M}{S,U} carry precision in the scalar operand and sf in the
        // GP operand; SCVTF/UCVTF/FCVTZ carry both in the typed Code.
        let (rmode, opcode) = match code {
            FcvtnsScalar => (0b00u32, 0b000u32),
            FcvtnuScalar => (0b00, 0b001),
            ScvtfS32 | ScvtfS64 | ScvtfD32 | ScvtfD64 | ScvtfH32 | ScvtfH64 => (0b00, 0b010),
            UcvtfS32 | UcvtfS64 | UcvtfD32 | UcvtfD64 | UcvtfH32 | UcvtfH64 => (0b00, 0b011),
            FcvtasScalar => (0b00, 0b100),
            FcvtauScalar => (0b00, 0b101),
            FcvtpsScalar => (0b01, 0b000),
            FcvtpuScalar => (0b01, 0b001),
            FcvtmsScalar => (0b10, 0b000),
            FcvtmuScalar => (0b10, 0b001),
            FcvtzsScalarS32 | FcvtzsScalarS64 | FcvtzsScalarD32 | FcvtzsScalarD64
            | FcvtzsScalarH32 | FcvtzsScalarH64 => (0b11, 0b000),
            _ => (0b11, 0b001), // FCVTZU int
        };
        let int_to_fp = matches!(opcode, 0b010 | 0b011);

        // Recover precision and sf.
        let (p, sf, rd, rn) = match code {
            ScvtfS32 | ScvtfS64 | ScvtfD32 | ScvtfD64 | ScvtfH32 | ScvtfH64 | UcvtfS32
            | UcvtfS64 | UcvtfD32 | UcvtfD64 | UcvtfH32 | UcvtfH64 | FcvtzsScalarS32
            | FcvtzsScalarS64 | FcvtzsScalarD32 | FcvtzsScalarD64 | FcvtzsScalarH32
            | FcvtzsScalarH64 | FcvtzuScalarS32 | FcvtzuScalarS64 | FcvtzuScalarD32
            | FcvtzuScalarD64 | FcvtzuScalarH32 | FcvtzuScalarH64 => {
                let (p, sf) = int_p_sf(code);
                if int_to_fp {
                    (p, sf, fp_dst(insn, 0, p)?, reg_num(insn, 1)?)
                } else {
                    (p, sf, reg_num(insn, 0)?, fp_src(insn, 1, p)?)
                }
            }
            // FCVT{N,A,P,M}{S,U}: int dst, fp src. sf from GP dst width; precision
            // from the scalar source register.
            _ => {
                let sf = gp_sf(insn, 0)?;
                let p = prec_of_scalar(insn, 1)?;
                (p, sf, reg_num(insn, 0)?, fp_src(insn, 1, p)?)
            }
        };
        let word = fp_base(p.ftype())
            | (sf << 31)
            | (rmode << 19)
            | (opcode << 16)
            | (rn << 5)
            | rd;
        Ok(word)
    }

    /// FJCVTZS: `0 0 0 11110 01 1 11 110 000000 Rn Rd` (sf=0, ftype=01, rmode=11,
    /// opcode=110).
    fn enc_fjcvtzs(insn: &Instruction) -> R {
        let rd = reg_num(insn, 0)?; // Wd
        let rn = fp_src(insn, 1, P::D)?;
        let word = fp_base(0b01) | (0b11 << 19) | (0b110 << 16) | (rn << 5) | rd;
        Ok(word)
    }

    /// FMOV general<->FP (rmode=00, opcode 6/7).
    fn enc_fmov_general(insn: &Instruction, code: Code) -> R {
        use Code::*;
        let (p, sf, from_fp) = match code {
            FmovToGp32 => (P::S, 0, true),
            FmovFromGp32 => (P::S, 0, false),
            FmovToGp64 => (P::D, 1, true),
            FmovFromGp64 => (P::D, 1, false),
            FmovToGpH32 => (P::H, 0, true),
            FmovFromGpH32 => (P::H, 0, false),
            FmovToGpH64 => (P::H, 1, true),
            _ => (P::H, 1, false), // FmovFromGpH64
        };
        let opcode = if from_fp { 0b110u32 } else { 0b111 };
        let (rd, rn) = if from_fp {
            (reg_num(insn, 0)?, fp_src(insn, 1, p)?)
        } else {
            (fp_dst(insn, 0, p)?, reg_num(insn, 1)?)
        };
        let word = fp_base(p.ftype()) | (sf << 31) | (opcode << 16) | (rn << 5) | rd;
        Ok(word)
    }

    /// FMOV between GP and the high 64 bits of a vector (sf=1, ftype=10,
    /// rmode=01, opcode 6/7).
    fn enc_fmov_top(insn: &Instruction, code: Code) -> R {
        let (opcode, rd, rn) = if code == Code::FmovTopToGp {
            // FMOV Xd, Vn.D[1].
            (0b110u32, reg_num(insn, 0)?, reg_num(insn, 1)?)
        } else {
            // FMOV Vd.D[1], Xn.
            (0b111u32, reg_num(insn, 0)?, reg_num(insn, 1)?)
        };
        let word = (1 << 31) | fp_base(0b10) | (0b01 << 19) | (opcode << 16) | (rn << 5) | rd;
        Ok(word)
    }

    /// dp1 same-precision ops (FMOV/FABS/FNEG/FSQRT/FRINT*).
    fn enc_dp1_same(insn: &Instruction, code: Code) -> R {
        use Code::*;
        let p = suffix_prec_dp1(code)?;
        let opcode = match code {
            FmovS | FmovD | FmovH => 0b000000u32,
            FabsS | FabsD | FabsH => 0b000001,
            FnegS | FnegD | FnegH => 0b000010,
            FsqrtS | FsqrtD | FsqrtH => 0b000011,
            FrintnS | FrintnD | FrintnH => 0b001000,
            FrintpS | FrintpD | FrintpH => 0b001001,
            FrintmS | FrintmD | FrintmH => 0b001010,
            FrintzS | FrintzD | FrintzH => 0b001011,
            FrintaS | FrintaD | FrintaH => 0b001100,
            FrintxS | FrintxD | FrintxH => 0b001110,
            FrintiS | FrintiD | FrintiH => 0b001111,
            Frint32zS | Frint32zD => 0b010000,
            Frint32xS | Frint32xD => 0b010001,
            Frint64zS | Frint64zD => 0b010010,
            _ => 0b010011, // Frint64x
        };
        let rd = fp_dst(insn, 0, p)?;
        let rn = fp_src(insn, 1, p)?;
        // dp1 fixed signature: word<14:10> == 10000.
        let word = fp_base(p.ftype()) | (opcode << 15) | (0b10000 << 10) | (rn << 5) | rd;
        Ok(word)
    }

    /// dp1 precision for the non-suffix-S/D/H frint32/64 forms (S/D only).
    fn suffix_prec_dp1(code: Code) -> Result<P, EncodeError> {
        use Code::*;
        if let Some(p) = suffix_prec(code) {
            return Ok(p);
        }
        Ok(match code {
            Frint32zS | Frint32xS | Frint64zS | Frint64xS => P::S,
            Frint32zD | Frint32xD | Frint64zD | Frint64xD => P::D,
            _ => return Err(EncodeError::Unsupported),
        })
    }

    /// FCVT between precisions: opcode 0b0001xx, dst from opcode<1:0>, src=ftype.
    fn enc_fcvt(insn: &Instruction, code: Code) -> R {
        use Code::*;
        let (src, dst) = match code {
            FcvtSD => (P::S, P::D),
            FcvtSH => (P::S, P::H),
            FcvtDS => (P::D, P::S),
            FcvtDH => (P::D, P::H),
            FcvtHS => (P::H, P::S),
            _ => (P::H, P::D), // FcvtHD
        };
        let opcode = 0b000100u32
            | match dst {
                P::S => 0b00,
                P::D => 0b01,
                P::H => 0b11,
            };
        let rd = fp_dst(insn, 0, dst)?;
        let rn = fp_src(insn, 1, src)?;
        let word = fp_base(src.ftype()) | (opcode << 15) | (0b10000 << 10) | (rn << 5) | rd;
        Ok(word)
    }

    /// BFCVT Hd, Sn: ftype=01 (double slot), opcode 0b000110.
    fn enc_bfcvt(insn: &Instruction) -> R {
        // dst is Hd, src is Sn.
        if scalar_width(insn, 0)? != 16 || scalar_width(insn, 1)? != 32 {
            return Err(EncodeError::InvalidOperand);
        }
        let rd = reg_num(insn, 0)?;
        let rn = reg_num(insn, 1)?;
        let word = fp_base(0b01) | (0b000110 << 15) | (0b10000 << 10) | (rn << 5) | rd;
        Ok(word)
    }

    /// dp2: `... ftype 1 Rm opcode 10 Rn Rd` (word<11:10>=10).
    fn enc_dp2(insn: &Instruction, code: Code) -> R {
        use Code::*;
        let p = suffix_prec(code).ok_or(EncodeError::Unsupported)?;
        let opcode = match code {
            FmulS | FmulD | FmulH => 0b0000u32,
            FdivS | FdivD | FdivH => 0b0001,
            FaddS | FaddD | FaddH => 0b0010,
            FsubS | FsubD | FsubH => 0b0011,
            FmaxS | FmaxD | FmaxH => 0b0100,
            FminS | FminD | FminH => 0b0101,
            FmaxnmS | FmaxnmD | FmaxnmH => 0b0110,
            FminnmS | FminnmD | FminnmH => 0b0111,
            _ => 0b1000, // FNMUL
        };
        let rd = fp_dst(insn, 0, p)?;
        let rn = fp_src(insn, 1, p)?;
        let rm = fp_src(insn, 2, p)?;
        let word = fp_base(p.ftype()) | (rm << 16) | (opcode << 12) | (0b10 << 10) | (rn << 5) | rd;
        Ok(word)
    }

    /// dp3: `M 0 S 11111 ftype o1 Rm o0 Ra Rn Rd` (word<28:24>=11111).
    fn enc_dp3(insn: &Instruction, code: Code) -> R {
        use Code::*;
        let p = suffix_prec(code).ok_or(EncodeError::Unsupported)?;
        let (o1, o0) = match code {
            FmaddS | FmaddD | FmaddH => (0u32, 0u32),
            FmsubS | FmsubD | FmsubH => (0, 1),
            FnmaddS | FnmaddD | FnmaddH => (1, 0),
            _ => (1, 1), // FNMSUB
        };
        let rd = fp_dst(insn, 0, p)?;
        let rn = fp_src(insn, 1, p)?;
        let rm = fp_src(insn, 2, p)?;
        let ra = fp_src(insn, 3, p)?;
        let word = (0b11111 << 24)
            | (p.ftype() << 22)
            | (o1 << 21)
            | (rm << 16)
            | (o0 << 15)
            | (ra << 10)
            | (rn << 5)
            | rd;
        Ok(word)
    }

    /// FCMP/FCMPE (register and `#0.0`): `... ftype 1 Rm 00 1000 Rn opcode2`.
    /// Operand 0 is `Rn`; operand 1 is `Rm` (register form) or `#0.0` (zero form,
    /// whose `Rm` field is architecturally unused — emitted as 0).
    fn enc_cmp(insn: &Instruction, code: Code) -> R {
        use Code::*;
        let p = suffix_prec(code).ok_or(EncodeError::Unsupported)?;
        let is_e = matches!(code, FcmpeS | FcmpeD | FcmpeH);
        let rn = fp_src(insn, 0, p)?;
        let (rm, is_zero) = match insn.op(1) {
            Operand::Reg { .. } => (fp_src(insn, 1, p)?, false),
            Operand::FpImm(_) => (0u32, true),
            _ => return Err(EncodeError::InvalidOperand),
        };
        let mut opcode2 = 0u32;
        if is_zero {
            opcode2 |= 0b01000;
        }
        if is_e {
            opcode2 |= 0b10000;
        }
        // op (word<15:14>) = 00, the 0b001000 group is encoded by <13:10>=1000.
        let word = fp_base(p.ftype()) | (rm << 16) | (0b1000 << 10) | (rn << 5) | opcode2;
        Ok(word)
    }

    /// FCCMP/FCCMPE: `... ftype 1 Rm cond 01 Rn op nzcv`.
    fn enc_ccmp(insn: &Instruction, code: Code) -> R {
        use Code::*;
        let p = suffix_prec(code).ok_or(EncodeError::Unsupported)?;
        let op = if matches!(code, FccmpeS | FccmpeD | FccmpeH) { 1u32 } else { 0 };
        let rn = fp_src(insn, 0, p)?;
        let rm = fp_src(insn, 1, p)?;
        let nzcv = imm_u(insn, 2)? as u32;
        let cond = cond_of(insn, 3)?.as_u4() as u32;
        if nzcv > 0xf {
            return Err(EncodeError::InvalidImmediate);
        }
        let word = fp_base(p.ftype())
            | (rm << 16)
            | (cond << 12)
            | (0b01 << 10)
            | (rn << 5)
            | (op << 4)
            | nzcv;
        Ok(word)
    }

    /// FCSEL: `... ftype 1 Rm cond 11 Rn Rd`.
    fn enc_csel(insn: &Instruction, code: Code) -> R {
        let p = suffix_prec(code).ok_or(EncodeError::Unsupported)?;
        let rd = fp_dst(insn, 0, p)?;
        let rn = fp_src(insn, 1, p)?;
        let rm = fp_src(insn, 2, p)?;
        let cond = cond_of(insn, 3)?.as_u4() as u32;
        let word = fp_base(p.ftype())
            | (rm << 16)
            | (cond << 12)
            | (0b11 << 10)
            | (rn << 5)
            | rd;
        Ok(word)
    }

    /// FMOV (scalar, immediate): `... ftype 1 imm8 100 00000 Rd`.
    fn enc_imm(insn: &Instruction, code: Code) -> R {
        let p = suffix_prec(code).ok_or(EncodeError::Unsupported)?;
        let rd = fp_dst(insn, 0, p)?;
        let f = fpimm_of(insn, 1)?;
        let n = match p {
            P::S => 32,
            P::D => 64,
            P::H => 16,
        };
        let imm8 = encode_vfp_imm(f, n).ok_or(EncodeError::InvalidImmediate)?;
        // word<12:5> region: imm8 at <20:13>, then `100` at <12:10>, imm5=0.
        let word = fp_base(p.ftype()) | (imm8 << 13) | (0b100 << 10) | rd;
        Ok(word)
    }

    // -----------------------------------------------------------------------
    // Operand precision/width validation helpers.
    // -----------------------------------------------------------------------

    /// Validate operand `n` is a scalar-FP register of precision `p`, returning
    /// its number.
    fn fp_dst(insn: &Instruction, n: usize, p: P) -> Result<u32, EncodeError> {
        check_prec(insn, n, p)?;
        reg_num(insn, n)
    }
    fn fp_src(insn: &Instruction, n: usize, p: P) -> Result<u32, EncodeError> {
        check_prec(insn, n, p)?;
        reg_num(insn, n)
    }

    /// Confirm scalar register `n` has the width implied by precision `p`.
    fn check_prec(insn: &Instruction, n: usize, p: P) -> Result<(), EncodeError> {
        let want = match p {
            P::S => 32u16,
            P::D => 64,
            P::H => 16,
        };
        if scalar_width(insn, n)? == want {
            Ok(())
        } else {
            Err(EncodeError::InvalidOperand)
        }
    }

    /// Recover the precision from the scalar-FP register at operand `n`.
    fn prec_of_scalar(insn: &Instruction, n: usize) -> Result<P, EncodeError> {
        Ok(match scalar_width(insn, n)? {
            32 => P::S,
            64 => P::D,
            16 => P::H,
            _ => return Err(EncodeError::InvalidOperand),
        })
    }

    /// `sf` bit (0/1) from the GP register width at operand `n` (W=0, X=1).
    fn gp_sf(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
        let r = reg_of(insn, n)?;
        if r.class() != RegClass::Gp {
            return Err(EncodeError::InvalidOperand);
        }
        Ok(if r.width_bits() == 64 { 1 } else { 0 })
    }
}
