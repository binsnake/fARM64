// Included into `simd_fp.rs` — Advanced SIMD arithmetic encoders.
//
// Inverse of `crate::decode::simd_fp::simd_arith`. For each `Code` we recover the
// `(U, opcode)` selectors and family from a table, then recover `size`/`Q` from
// the operand arrangement (vector) or scalar element width, plus any index /
// immediate, and pack the word.

mod simd_arith {
    use super::*;

    /// The fixed top nibble/byte signature pieces.
    /// Vector three-X/two-reg-misc/across: word<28:24> = 0b01110 (vector) or
    /// 0b11110 (scalar). By-element: 0b01111 / 0b11111.
    #[inline]
    fn top(scalar: bool, by_elem: bool) -> u32 {
        let base = if by_elem { 0b01111 } else { 0b01110 };
        let base = if scalar { base | 0b10000 } else { base };
        // The scalar (asisd) encodings additionally fix word<30>==1 (their
        // `01` bits<31:30> prefix). Vector forms leave word<30> as the caller's
        // `q` bit, so we only contribute it for scalar here.
        let bit30 = if scalar { 1u32 << 30 } else { 0 };
        (base << 24) | bit30
    }

    /// Element width (bits) -> 2-bit integer `size` field.
    #[inline]
    fn size_for_eb(eb: u16) -> Result<u32, EncodeError> {
        Ok(match eb {
            8 => 0b00,
            16 => 0b01,
            32 => 0b10,
            64 => 0b11,
            _ => return Err(EncodeError::InvalidOperand),
        })
    }

    pub(super) fn encode(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
        // Try each arithmetic family in turn; the first that recognizes the code
        // returns the word.
        if let Some(w) = three_same(insn, code)? {
            return Ok(Some(w));
        }
        if let Some(w) = three_same_logical(insn, code)? {
            return Ok(Some(w));
        }
        if let Some(w) = three_same_fp(insn, code)? {
            return Ok(Some(w));
        }
        if let Some(w) = extra_complex_dot(insn, code)? {
            return Ok(Some(w));
        }
        if let Some(w) = ext_dot_mlal_vec(insn, code)? {
            return Ok(Some(w));
        }
        if let Some(w) = three_different(insn, code)? {
            return Ok(Some(w));
        }
        if let Some(w) = two_reg_misc(insn, code)? {
            return Ok(Some(w));
        }
        if let Some(w) = across_or_pairwise(insn, code)? {
            return Ok(Some(w));
        }
        if let Some(w) = fmlal_vec(insn, code)? {
            return Ok(Some(w));
        }
        if let Some(w) = by_element(insn, code)? {
            return Ok(Some(w));
        }
        Ok(None)
    }

    /// FMLAL/FMLSL/FMLAL2/FMLSL2 vector (three-same widening) form:
    /// `fmlal Vd.<2s/4s>, Vn.<2h/4h>, Vm.<2h/4h>`. These share the FP three-same
    /// opcode space but use the widening single-from-half opcodes (not covered by
    /// `fp_three_same_fields`). The by-element form (`Vm.h[idx]`) is handled in
    /// `by_element`. Layout: `0 Q U 01110 size 1 Rm opcode 1 Rn Rd`, with
    /// `size = a<<1` (sz==0, single-from-half), `a` selecting add vs sub.
    fn fmlal_vec(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
        use Code::*;
        let (u, a, opcode) = match code {
            FmlalVec => (0u32, 0u32, 0b11101u32),
            FmlslVec => (0, 1, 0b11101),
            Fmlal2Vec => (1, 0, 0b11001),
            Fmlsl2Vec => (1, 1, 0b11001),
            _ => return Ok(None),
        };
        // Only the vector three-register form here; the by-element form has an
        // indexed last operand and is handled separately.
        if is_by_element(insn) {
            return Ok(None);
        }
        let q = match arr_of(insn, 0)? {
            VA::V2S => 0u32,
            VA::V4S => 1,
            _ => return Err(EncodeError::InvalidOperand),
        };
        let size = a << 1;
        let rm = reg_num(insn, 2)?;
        let rn = reg_num(insn, 1)?;
        let rd = reg_num(insn, 0)?;
        let word = (q << 30)
            | (u << 29)
            | top(false, false)
            | (size << 22)
            | (1 << 21)
            | (rm << 16)
            | (opcode << 11)
            | (1 << 10)
            | (rn << 5)
            | rd;
        Ok(Some(word))
    }

    // =======================================================================
    // Three-same integer.
    // =======================================================================

    /// `(U, opcode)` for an integer three-same code; `None` if not in the family.
    fn int_three_same_fields(code: Code) -> Option<(u32, u32)> {
        use Code::*;
        let v = match code {
            ShaddVec => (0, 0b00000),
            SqaddVec => (0, 0b00001),
            SrhaddVec => (0, 0b00010),
            ShsubVec => (0, 0b00100),
            SqsubVec => (0, 0b00101),
            CmgtVec => (0, 0b00110),
            CmgeVec => (0, 0b00111),
            SshlVec => (0, 0b01000),
            SqshlVec => (0, 0b01001),
            SrshlVec => (0, 0b01010),
            SqrshlVec => (0, 0b01011),
            SmaxVec => (0, 0b01100),
            SminVec => (0, 0b01101),
            SabdVec => (0, 0b01110),
            SabaVec => (0, 0b01111),
            AddVec => (0, 0b10000),
            CmtstVec => (0, 0b10001),
            MlaVec => (0, 0b10010),
            MulVec => (0, 0b10011),
            SmaxpVec => (0, 0b10100),
            SminpVec => (0, 0b10101),
            SqdmulhVec => (0, 0b10110),
            AddpVec => (0, 0b10111),
            UhaddVec => (1, 0b00000),
            UqaddVec => (1, 0b00001),
            UrhaddVec => (1, 0b00010),
            UhsubVec => (1, 0b00100),
            UqsubVec => (1, 0b00101),
            CmhiVec => (1, 0b00110),
            CmhsVec => (1, 0b00111),
            UshlVec => (1, 0b01000),
            UqshlVec => (1, 0b01001),
            UrshlVec => (1, 0b01010),
            UqrshlVec => (1, 0b01011),
            UmaxVec => (1, 0b01100),
            UminVec => (1, 0b01101),
            UabdVec => (1, 0b01110),
            UabaVec => (1, 0b01111),
            SubVec => (1, 0b10000),
            CmeqVec => (1, 0b10001),
            MlsVec => (1, 0b10010),
            PmulVec => (1, 0b10011),
            UmaxpVec => (1, 0b10100),
            UminpVec => (1, 0b10101),
            SqrdmulhVec => (1, 0b10110),
            _ => return None,
        };
        Some(v)
    }

    fn three_same(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
        let Some((u, opcode)) = int_three_same_fields(code) else {
            return Ok(None);
        };
        // Several compare codes are shared with the two-reg-misc compare-against-
        // zero forms (`Vd, Vn, #0`); those are not three-same. Defer to the misc
        // encoder when the third operand is not a register.
        if !is_three_reg(insn) {
            return Ok(None);
        }
        // Scalar vs vector: scalar if operand 0 is a scalar-FP register.
        let scalar = reg_of(insn, 0)?.class() == RegClass::ScalarFp;
        let (size, q) = if scalar {
            let eb = scalar_width(insn, 0)?;
            (size_for_eb(eb)?, 0u32)
        } else {
            let a = arr_of(insn, 0)?;
            (size_of_arr(a)?, q_of_arr(a)?)
        };
        let rm = reg_num(insn, 2)?;
        let rn = reg_num(insn, 1)?;
        let rd = reg_num(insn, 0)?;
        let word = (q << 30)
            | (u << 29)
            | top(scalar, false)
            | (size << 22)
            | (1 << 21)
            | (rm << 16)
            | (opcode << 11)
            | (1 << 10)
            | (rn << 5)
            | rd;
        Ok(Some(word))
    }

    // =======================================================================
    // Three-same logical (AND/BIC/ORR/ORN/EOR/BSL/BIT/BIF + MOV alias).
    // =======================================================================

    fn three_same_logical(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
        use Code::*;
        let (u, sz) = match code {
            AndVec => (0u32, 0b00u32),
            BicVec => (0, 0b01),
            OrrVec => (0, 0b10),
            OrnVec => (0, 0b11),
            EorVec => (1, 0b00),
            BslVec => (1, 0b01),
            BitVec => (1, 0b10),
            BifVec => (1, 0b11),
            _ => return Ok(None),
        };
        let a = arr_of(insn, 0)?;
        let q = match a {
            VA::V8B => 0u32,
            VA::V16B => 1,
            _ => return Err(EncodeError::InvalidOperand),
        };
        let rd = reg_num(insn, 0)?;
        let rn = reg_num(insn, 1)?;
        // MOV alias (ORR with Vn==Vm): only two operands.
        let rm = if code == OrrVec && insn.mnemonic() == Mnemonic::Mov {
            rn
        } else {
            reg_num(insn, 2)?
        };
        // opcode field is fixed 0b00011; `size` field carries the `sz` selector.
        let word = (q << 30)
            | (u << 29)
            | top(false, false)
            | (sz << 22)
            | (1 << 21)
            | (rm << 16)
            | (0b00011 << 11)
            | (1 << 10)
            | (rn << 5)
            | rd;
        Ok(Some(word))
    }

    // =======================================================================
    // Three-same FP / FP16.
    // =======================================================================

    /// `(U, o1, opcode5)` for the single/double FP three-same family, where `o1`
    /// is `size<1>`. The precision bit `size<0>` is recovered from operands. The
    /// FP16 family reuses the same `(U,a,opcode3)` but a 3-bit opcode; we encode
    /// both by mapping each `Code` to the SP/DP 5-bit opcode and then converting
    /// to the FP16 3-bit form when the operand element is half.
    fn fp_three_same_fields(code: Code) -> Option<(u32, u32, u32)> {
        use Code::*;
        // (U, o1=size<1>, opcode<4:0>)
        let v = match code {
            FmaxnmVec => (0, 0, 0b11000),
            FmlaVec => (0, 0, 0b11001),
            FaddVec => (0, 0, 0b11010),
            FmulxVec => (0, 0, 0b11011),
            FcmeqVec => (0, 0, 0b11100),
            FmaxVec => (0, 0, 0b11110),
            FrecpsVec => (0, 0, 0b11111),
            FminnmVec => (0, 1, 0b11000),
            FmlsVec => (0, 1, 0b11001),
            FsubVec => (0, 1, 0b11010),
            FminVec => (0, 1, 0b11110),
            FrsqrtsVec => (0, 1, 0b11111),
            FmaxnmpVec => (1, 0, 0b11000),
            FaddpVec => (1, 0, 0b11010),
            FmulVec => (1, 0, 0b11011),
            FcmgeVec => (1, 0, 0b11100),
            FacgeVec => (1, 0, 0b11101),
            FmaxpVec => (1, 0, 0b11110),
            FdivVec => (1, 0, 0b11111),
            FminnmpVec => (1, 1, 0b11000),
            FabdVec => (1, 1, 0b11010),
            FcmgtVec => (1, 1, 0b11100),
            FacgtVec => (1, 1, 0b11101),
            FminpVec => (1, 1, 0b11110),
            // FEAT_FAMINMAX / FEAT_FP8 (o1==size<1>==1 group).
            FamaxVec => (0, 1, 0b11011),
            FaminVec => (1, 1, 0b11011),
            FscaleVec => (1, 1, 0b11111),
            _ => return None,
        };
        Some(v)
    }

    /// FP16 three-same 3-bit opcode for a `Code` (the opcode<2:0> used by the
    /// half-precision encoding). Mirrors `fp16_three_same` in the decoder.
    fn fp16_three_same_op(code: Code, u: u32, a: u32) -> Option<u32> {
        use Code::*;
        let v = match (u, a, code) {
            (0, 0, FmaxnmVec) => 0b000,
            (0, 0, FmlaVec) => 0b001,
            (0, 0, FaddVec) => 0b010,
            (0, 0, FmulxVec) => 0b011,
            (0, 0, FcmeqVec) => 0b100,
            (0, 0, FmaxVec) => 0b110,
            (0, 0, FrecpsVec) => 0b111,
            (0, 1, FminnmVec) => 0b000,
            (0, 1, FmlsVec) => 0b001,
            (0, 1, FsubVec) => 0b010,
            (0, 1, FminVec) => 0b110,
            (0, 1, FrsqrtsVec) => 0b111,
            (1, 0, FmaxnmpVec) => 0b000,
            (1, 0, FaddpVec) => 0b010,
            (1, 0, FmulVec) => 0b011,
            (1, 0, FcmgeVec) => 0b100,
            (1, 0, FacgeVec) => 0b101,
            (1, 0, FmaxpVec) => 0b110,
            (1, 0, FdivVec) => 0b111,
            (1, 1, FminnmpVec) => 0b000,
            (1, 1, FabdVec) => 0b010,
            (1, 1, FcmgtVec) => 0b100,
            (1, 1, FacgtVec) => 0b101,
            (1, 1, FminpVec) => 0b110,
            // FEAT_FAMINMAX / FEAT_FP8 (half), a==size<1>==1 group.
            (0, 1, FamaxVec) => 0b011,
            (1, 1, FaminVec) => 0b011,
            (1, 1, FscaleVec) => 0b111,
            _ => return None,
        };
        Some(v)
    }

    fn three_same_fp(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
        let Some((u, o1, opcode)) = fp_three_same_fields(code) else {
            return Ok(None);
        };
        // FCMGT/FCMGE/FCMEQ are shared with the two-reg-misc compare-against-zero
        // forms (`Vd, Vn, #0.0`); the FP scalar-pairwise reductions (FADDP etc.)
        // are shared too (`Vd, Vn.2x`). Both have a non-register or absent third
        // operand — defer to the other family encoders.
        if !is_three_reg(insn) {
            return Ok(None);
        }
        let scalar = reg_of(insn, 0)?.class() == RegClass::ScalarFp;

        // Determine element width: half (FP16 family), single, or double.
        let half = is_half(insn, 0)?;
        if half {
            // FP16 three-same: word<21>==0, opcode at <13:11>, a=word<23>.
            let a = o1; // a == size<1> selects the variant
            let op3 = fp16_three_same_op(code, u, a).ok_or(EncodeError::InvalidOperand)?;
            let q = if scalar { 0 } else {
                q_of_arr(arr_of(insn, 0)?)?
            };
            let rm = reg_num(insn, 2)?;
            let rn = reg_num(insn, 1)?;
            let rd = reg_num(insn, 0)?;
            // word<23>=a, word<22>=1 (FP16 marker), word<21>=0, word<15:14>=00,
            // word<13:11>=op3, word<10>=1.
            let word = (q << 30)
                | (u << 29)
                | top(scalar, false)
                | (a << 23)
                | (1 << 22)
                | (op3 << 11)
                | (1 << 10)
                | (rm << 16)
                | (rn << 5)
                | rd;
            return Ok(Some(word));
        }

        // Single/double: size = o1<<1 | sz, sz from element width (S=0, D=1).
        let sz = if scalar {
            match scalar_width(insn, 0)? {
                32 => 0u32,
                64 => 1,
                _ => return Err(EncodeError::InvalidOperand),
            }
        } else {
            let a = arr_of(insn, 0)?;
            match a {
                VA::V2S | VA::V4S => 0,
                VA::V2D => 1,
                _ => return Err(EncodeError::InvalidOperand),
            }
        };
        let size = (o1 << 1) | sz;
        let q = if scalar { 0 } else { q_of_arr(arr_of(insn, 0)?)? };
        let rm = reg_num(insn, 2)?;
        let rn = reg_num(insn, 1)?;
        let rd = reg_num(insn, 0)?;
        let word = (q << 30)
            | (u << 29)
            | top(scalar, false)
            | (size << 22)
            | (1 << 21)
            | (rm << 16)
            | (opcode << 11)
            | (1 << 10)
            | (rn << 5)
            | rd;
        Ok(Some(word))
    }

    // =======================================================================
    // Three-same "extra" (SQRDMLAH/SQRDMLSH), complex (FCMLA/FCADD), dot
    // (SDOT/UDOT vector). All sit in the word<21>==0 region.
    // =======================================================================

    fn extra_complex_dot(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
        use Code::*;
        match code {
            // SQRDMLAH/SQRDMLSH three-same-extra: but these codes ALSO appear in
            // by-element. Disambiguate by operand count / indexed last operand.
            SqrdmlahVec | SqrdmlshVec => {
                if is_by_element(insn) {
                    return Ok(None); // handled by by_element
                }
                let scalar = reg_of(insn, 0)?.class() == RegClass::ScalarFp;
                let opcode = if code == SqrdmlahVec { 0b10000u32 } else { 0b10001 };
                let (size, q) = if scalar {
                    (size_for_eb(scalar_width(insn, 0)?)?, 0)
                } else {
                    let a = arr_of(insn, 0)?;
                    (size_of_arr(a)?, q_of_arr(a)?)
                };
                let rm = reg_num(insn, 2)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                // U==1, word<21>==0, opcode at <15:11>, word<10>==1.
                let word = (q << 30)
                    | (1 << 29)
                    | top(scalar, false)
                    | (size << 22)
                    | (rm << 16)
                    | (opcode << 11)
                    | (1 << 10)
                    | (rn << 5)
                    | rd;
                Ok(Some(word))
            }
            // FCMLA / FCADD: complex. FCMLA also has a by-element form.
            FcmlaVec => {
                if is_by_element(insn) {
                    return Ok(None);
                }
                // three-same complex FCMLA: U=1, word<15:13>=110, rot=word<12:11>,
                // word<10>=1. operands: Vd, Vn, Vm, #rot.
                let a = arr_of(insn, 0)?;
                let (size, q) = complex_size_q(a)?;
                let rot = (imm_u(insn, 3)? / 90) as u32;
                if rot > 3 {
                    return Err(EncodeError::InvalidImmediate);
                }
                let rm = reg_num(insn, 2)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                let word = (q << 30)
                    | (1 << 29)
                    | top(false, false)
                    | (size << 22)
                    | (rm << 16)
                    | (0b110 << 13)
                    | (rot << 11)
                    | (1 << 10)
                    | (rn << 5)
                    | rd;
                Ok(Some(word))
            }
            FcaddVec => {
                let a = arr_of(insn, 0)?;
                let (size, q) = complex_size_q(a)?;
                let deg = imm_u(insn, 3)?;
                let rot1 = match deg {
                    90 => 0u32,
                    270 => 1,
                    _ => return Err(EncodeError::InvalidImmediate),
                };
                let rm = reg_num(insn, 2)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                // word<15:13>=111, rot1=word<12>, word<11>=0, word<10>=1.
                let word = (q << 30)
                    | (1 << 29)
                    | top(false, false)
                    | (size << 22)
                    | (rm << 16)
                    | (0b111 << 13)
                    | (rot1 << 12)
                    | (1 << 10)
                    | (rn << 5)
                    | rd;
                Ok(Some(word))
            }
            // SDOT/UDOT vector (three-same region, word<21>==0): also has Idx forms
            // (distinct codes SdotIdx/UdotIdx handled in by_element).
            SdotVec | UdotVec => {
                let u = if code == SdotVec { 0u32 } else { 1 };
                let q = q_of_arr(arr_of(insn, 0)?)?;
                let rm = reg_num(insn, 2)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                // size==10, word<15:10>=100101.
                let word = (q << 30)
                    | (u << 29)
                    | top(false, false)
                    | (0b10 << 22)
                    | (rm << 16)
                    | (0b100101 << 10)
                    | (rn << 5)
                    | rd;
                Ok(Some(word))
            }
            _ => Ok(None),
        }
    }

    /// FP8 / I8MM / BF16 Advanced-SIMD "three-register extension" dot-product
    /// and widening MLAL — *vector* forms (the by-element forms are encoded in
    /// `by_element`). Inverse of `decode::simd_fp::simd_arith::simd_three_reg_ext`.
    /// Layout: `0 Q U 01110 size 0 Rm <word15:10> Rn Rd`.
    fn ext_dot_mlal_vec(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
        use Code::*;
        // The MLAL codes (FmlalbVec/Bfmlalb.../Fmlall*) also carry a by-element
        // form under the same Code; defer those to `by_element`.
        if is_by_element(insn) {
            return Ok(None);
        }
        let (u, size, q, lo) = match code {
            FdotVec => {
                // size + Q from the destination arrangement (to-single .2s/.4s ->
                // size 00; to-half .4h/.8h -> size 01).
                let (size, q) = match arr_of(insn, 0)? {
                    VA::V2S => (0b00u32, 0u32),
                    VA::V4S => (0b00, 1),
                    VA::V4H => (0b01, 0),
                    VA::V8H => (0b01, 1),
                    _ => return Err(EncodeError::InvalidOperand),
                };
                (0u32, size, q, 0b111111u32)
            }
            UsdotVec => (0, 0b10, q_of_arr(arr_of(insn, 0)?)?, 0b100111),
            BfdotVec => (1, 0b01, q_of_arr(arr_of(insn, 0)?)?, 0b111111),
            FmlalbVec => (0, 0b11, 0, 0b111111),
            FmlaltVec => (0, 0b11, 1, 0b111111),
            BfmlalbVec => (1, 0b11, 0, 0b111111),
            BfmlaltVec => (1, 0b11, 1, 0b111111),
            FmlallbbVec => (0, 0b00, 0, 0b110001),
            FmlallbtVec => (0, 0b01, 0, 0b110001),
            FmlalltbVec => (0, 0b00, 1, 0b110001),
            FmlallttVec => (0, 0b01, 1, 0b110001),
            // FP8 FCVTN (FEAT_FP8): two-source convert+narrow to FP8 (lo=111101).
            // size==00 -> FP32 sources `.4s`; size==01 -> FP16 sources. Q selects
            // the destination width (.8b/.16b). FCVTN2 is the size==00, Q==1 form.
            FcvtnFp8 => {
                let (size, q) = match (arr_of(insn, 0)?, arr_of(insn, 1)?) {
                    (VA::V8B, VA::V4S) => (0b00u32, 0u32),
                    (VA::V8B, VA::V4H) => (0b01, 0),
                    (VA::V16B, VA::V8H) => (0b01, 1),
                    _ => return Err(EncodeError::InvalidOperand),
                };
                (0u32, size, q, 0b111101u32)
            }
            Fcvtn2Fp8 => (0, 0b00, 1, 0b111101),
            // FEAT_I8MM integer matrix multiply-accumulate (Q=1, size=10).
            //   SMMLA  lo=101001 U=0; UMMLA lo=101001 U=1; USMMLA lo=101011 U=0.
            SmmlaVec => (0, 0b10, 1, 0b101001),
            UmmlaVec => (1, 0b10, 1, 0b101001),
            UsmmlaVec => (0, 0b10, 1, 0b101011),
            _ => return Ok(None),
        };
        let rm = reg_num(insn, 2)?;
        let rn = reg_num(insn, 1)?;
        let rd = reg_num(insn, 0)?;
        let word = (q << 30)
            | (u << 29)
            | top(false, false)
            | (size << 22)
            | (rm << 16)
            | (lo << 10)
            | (rn << 5)
            | rd;
        Ok(Some(word))
    }

    /// `(size, q)` for a complex FCMLA/FCADD vector arrangement.
    fn complex_size_q(a: VA) -> Result<(u32, u32), EncodeError> {
        Ok(match a {
            VA::V4H => (0b01, 0),
            VA::V8H => (0b01, 1),
            VA::V2S => (0b10, 0),
            VA::V4S => (0b10, 1),
            VA::V2D => (0b11, 1),
            _ => return Err(EncodeError::InvalidOperand),
        })
    }

    // =======================================================================
    // Three-different.
    // =======================================================================

    /// `(U, opcode4, base_code_for_q0)` for a three-different code. The `2` form
    /// (Q==1) and non-`2` (Q==0) share the encoding; we recover Q from operand
    /// arrangement and select the matching mnemonic by the code passed in.
    fn three_diff_fields(code: Code) -> Option<(u32, u32)> {
        use Code::*;
        let v = match code {
            SaddlVec | Saddl2Vec => (0, 0b0000),
            SaddwVec | Saddw2Vec => (0, 0b0001),
            SsublVec | Ssubl2Vec => (0, 0b0010),
            SsubwVec | Ssubw2Vec => (0, 0b0011),
            AddhnVec | Addhn2Vec => (0, 0b0100),
            SabalVec | Sabal2Vec => (0, 0b0101),
            SubhnVec | Subhn2Vec => (0, 0b0110),
            SabdlVec | Sabdl2Vec => (0, 0b0111),
            SmlalVec | Smlal2Vec => (0, 0b1000),
            SqdmlalVec | Sqdmlal2Vec => (0, 0b1001),
            SmlslVec | Smlsl2Vec => (0, 0b1010),
            SqdmlslVec | Sqdmlsl2Vec => (0, 0b1011),
            SmullVec | Smull2Vec => (0, 0b1100),
            SqdmullVec | Sqdmull2Vec => (0, 0b1101),
            PmullVec | Pmull2Vec => (0, 0b1110),
            UaddlVec | Uaddl2Vec => (1, 0b0000),
            UaddwVec | Uaddw2Vec => (1, 0b0001),
            UsublVec | Usubl2Vec => (1, 0b0010),
            UsubwVec | Usubw2Vec => (1, 0b0011),
            RaddhnVec | Raddhn2Vec => (1, 0b0100),
            UabalVec | Uabal2Vec => (1, 0b0101),
            RsubhnVec | Rsubhn2Vec => (1, 0b0110),
            UabdlVec | Uabdl2Vec => (1, 0b0111),
            UmlalVec | Umlal2Vec => (1, 0b1000),
            UmlslVec | Umlsl2Vec => (1, 0b1010),
            UmullVec | Umull2Vec => (1, 0b1100),
            _ => return None,
        };
        Some(v)
    }

    /// Shape of a three-different code (mirrors `Shape3D` in the decoder).
    #[derive(Clone, Copy, PartialEq)]
    enum Shape {
        LongL,
        LongLSat,
        WideW,
        NarrowHN,
        Pmull,
    }

    fn three_diff_shape(code: Code) -> Shape {
        use Code::*;
        match code {
            SaddwVec | Saddw2Vec | SsubwVec | Ssubw2Vec | UaddwVec | Uaddw2Vec | UsubwVec
            | Usubw2Vec => Shape::WideW,
            AddhnVec | Addhn2Vec | SubhnVec | Subhn2Vec | RaddhnVec | Raddhn2Vec | RsubhnVec
            | Rsubhn2Vec => Shape::NarrowHN,
            SqdmlalVec | Sqdmlal2Vec | SqdmlslVec | Sqdmlsl2Vec | SqdmullVec | Sqdmull2Vec => {
                Shape::LongLSat
            }
            PmullVec | Pmull2Vec => Shape::Pmull,
            _ => Shape::LongL,
        }
    }

    /// `true` if the code is a `…2` (hi-half) three-different / narrow / long
    /// variant.
    fn is_2_form(code: Code) -> bool {
        use Code::*;
        matches!(
            code,
            Saddl2Vec
                | Saddw2Vec
                | Ssubl2Vec
                | Ssubw2Vec
                | Addhn2Vec
                | Sabal2Vec
                | Subhn2Vec
                | Sabdl2Vec
                | Smlal2Vec
                | Sqdmlal2Vec
                | Smlsl2Vec
                | Sqdmlsl2Vec
                | Smull2Vec
                | Sqdmull2Vec
                | Pmull2Vec
                | Uaddl2Vec
                | Uaddw2Vec
                | Usubl2Vec
                | Usubw2Vec
                | Raddhn2Vec
                | Uabal2Vec
                | Rsubhn2Vec
                | Uabdl2Vec
                | Umlal2Vec
                | Umlsl2Vec
                | Umull2Vec
                | Xtn2Vec
                | Sqxtn2Vec
                | Uqxtn2Vec
                | Sqxtun2Vec
                | Shrn2Vec
                | Rshrn2Vec
                | Sqshrn2Vec
                | Sqrshrn2Vec
                | Sqshrun2Vec
                | Sqrshrun2Vec
                | Uqshrn2Vec
                | Uqrshrn2Vec
                | Sshll2Vec
                | Ushll2Vec
                | Sxtl2Vec
                | Uxtl2Vec
                | Fcvtl2Vec
                | Fcvtn2Vec
                | Fcvtxn2Vec
        )
    }

    fn three_different(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
        let Some((u, opcode)) = three_diff_fields(code) else {
            return Ok(None);
        };
        // SMLAL/SMULL/SQDMULL/... share their `Code` with the by-element long
        // forms; those carry an indexed last operand and belong to `by_element`.
        if is_by_element(insn) {
            return Ok(None);
        }
        let shape = three_diff_shape(code);

        // Scalar three-different (SQDMULL/SQDMLAL/SQDMLSL only): operand 0 scalar.
        let scalar = reg_of(insn, 0)?.class() == RegClass::ScalarFp;
        if scalar {
            // Source element width from operand 1 (Hn/Sn); size = that width.
            let eb = scalar_width(insn, 1)?;
            let size = size_for_eb(eb)?;
            let rm = reg_num(insn, 2)?;
            let rn = reg_num(insn, 1)?;
            let rd = reg_num(insn, 0)?;
            // Scalar three-different: top=11110, word<21>=1, word<11:10>=00.
            let word = (u << 29)
                | top(true, false)
                | (size << 22)
                | (1 << 21)
                | (rm << 16)
                | (opcode << 12)
                | (rn << 5)
                | rd;
            return Ok(Some(word));
        }

        // Vector: recover size + Q from the narrow operand's arrangement.
        // The narrow side differs by shape; the `size` field is the *narrow*
        // element size, and Q selects the hi/lo half (matching the `2` form).
        let q = if is_2_form(code) { 1u32 } else { 0 };
        let size = match shape {
            Shape::Pmull => {
                // dst Ta (.8h or .1q), src Tb (.8b/.16b or .1d/.2d). size from Tb.
                let tb = arr_of(insn, 1)?;
                match tb {
                    VA::V8B | VA::V16B => 0b00,
                    VA::V1D | VA::V2D => 0b11,
                    _ => return Err(EncodeError::InvalidOperand),
                }
            }
            Shape::WideW => {
                // Vd.Ta, Vn.Ta, Vm.Tb. size from the narrow Tb (operand 2).
                size_from_narrow(arr_of(insn, 2)?)?
            }
            Shape::NarrowHN => {
                // Vd.Tb, Vn.Ta, Vm.Ta. size from narrow result Tb (operand 0).
                size_from_narrow(arr_of(insn, 0)?)?
            }
            _ => {
                // Long: Vd.Ta, Vn.Tb, Vm.Tb. size from narrow Tb (operand 1).
                size_from_narrow(arr_of(insn, 1)?)?
            }
        };
        let _ = shape;
        let rm = reg_num(insn, 2)?;
        let rn = reg_num(insn, 1)?;
        let rd = reg_num(insn, 0)?;
        let word = (q << 30)
            | (u << 29)
            | top(false, false)
            | (size << 22)
            | (1 << 21)
            | (rm << 16)
            | (opcode << 12)
            | (rn << 5)
            | rd;
        Ok(Some(word))
    }

    /// The narrow 2-bit `size` for an arrangement that is the *narrow* side of a
    /// long/wide/narrow three-different op. Both the `.8b` and `.16b` (etc.)
    /// halves map to the same size.
    fn size_from_narrow(a: VA) -> Result<u32, EncodeError> {
        Ok(match a {
            VA::V8B | VA::V16B => 0b00,
            VA::V4H | VA::V8H => 0b01,
            VA::V2S | VA::V4S => 0b10,
            _ => return Err(EncodeError::InvalidOperand),
        })
    }

    // =======================================================================
    // Two-register miscellaneous.
    // =======================================================================

    fn two_reg_misc(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
        if let Some(w) = int_two_reg_misc(insn, code)? {
            return Ok(Some(w));
        }
        if let Some(w) = fp_two_reg_misc(insn, code)? {
            return Ok(Some(w));
        }
        Ok(None)
    }

    /// Base for a two-reg-misc: `q u 01110 size 1 0000 opcode 10 Rn Rd`.
    /// (word<21:17>=10000.)
    #[inline]
    fn misc_word(scalar: bool, q: u32, u: u32, size: u32, opcode: u32, rn: u32, rd: u32) -> u32 {
        // word<21:17> == 10000 is exactly word<21>==1 with word<20:17>==0; the
        // opcode occupies word<16:12>. (Do NOT also set bit 20.)
        (q << 30)
            | (u << 29)
            | top(scalar, false)
            | (size << 22)
            | (1 << 21)
            | (opcode << 12)
            | (0b10 << 10)
            | (rn << 5)
            | rd
    }

    fn int_two_reg_misc(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
        use Code::*;
        // (U, opcode5) for the integer two-reg-misc family.
        let (u, opcode) = match code {
            Rev64Vec => (0, 0b00000),
            Rev16Vec => (0, 0b00001),
            SaddlpVec => (0, 0b00010),
            SuqaddVec => (0, 0b00011),
            ClsVec => (0, 0b00100),
            CntVec => (0, 0b00101),
            SadalpVec => (0, 0b00110),
            SqabsVec => (0, 0b00111),
            CmgtVec => (0, 0b01000), // CMGT #0
            CmeqVec => (0, 0b01001), // CMEQ #0
            CmltVec => (0, 0b01010), // CMLT #0
            AbsVec => (0, 0b01011),
            XtnVec | Xtn2Vec => (0, 0b10010),
            SqxtnVec | Sqxtn2Vec => (0, 0b10100),
            Rev32Vec => (1, 0b00000),
            UaddlpVec => (1, 0b00010),
            UsqaddVec => (1, 0b00011),
            ClzVec => (1, 0b00100),
            MvnVec => (1, 0b00101),  // size==00
            RbitVec => (1, 0b00101), // size==01
            UadalpVec => (1, 0b00110),
            SqnegVec => (1, 0b00111),
            CmgeVec => (1, 0b01000), // CMGE #0
            CmleVec => (1, 0b01001), // CMLE #0
            NegVec => (1, 0b01011),
            SqxtunVec | Sqxtun2Vec => (1, 0b10010),
            Shll2Vec | ShllVec => (1, 0b10011),
            UqxtnVec | Uqxtn2Vec => (1, 0b10100),
            _ => return Ok(None),
        };

        let scalar = reg_of(insn, 0)?.class() == RegClass::ScalarFp;

        // --- families needing special operand shapes ---
        match code {
            // Compare-against-zero (have a trailing #0). Same arrangement in/out.
            CmgtVec | CmeqVec | CmltVec | CmgeVec | CmleVec => {
                let (size, q) = if scalar {
                    (size_for_eb(scalar_width(insn, 0)?)?, 0)
                } else {
                    let a = arr_of(insn, 0)?;
                    (size_of_arr(a)?, q_of_arr(a)?)
                };
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                return Ok(Some(misc_word(scalar, q, u, size, opcode, rn, rd)));
            }
            // MVN / RBIT: byte arrangement; size selects which (00=MVN, 01=RBIT).
            MvnVec | RbitVec => {
                let a = arr_of(insn, 0)?;
                let q = match a {
                    VA::V8B => 0u32,
                    VA::V16B => 1,
                    _ => return Err(EncodeError::InvalidOperand),
                };
                let size = if code == MvnVec { 0b00u32 } else { 0b01 };
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                return Ok(Some(misc_word(false, q, u, size, opcode, rn, rd)));
            }
            // Narrowing XTN/SQXTN/UQXTN/SQXTUN: Vd.Tb, Vn.Ta; size from result Tb.
            XtnVec | Xtn2Vec | SqxtnVec | Sqxtn2Vec | UqxtnVec | Uqxtn2Vec | SqxtunVec
            | Sqxtun2Vec => {
                if scalar {
                    // result eb (operand0), source 2*eb (operand1).
                    let eb = scalar_width(insn, 0)?;
                    let size = size_for_eb(eb)?;
                    let rn = reg_num(insn, 1)?;
                    let rd = reg_num(insn, 0)?;
                    return Ok(Some(misc_word(true, 0, u, size, opcode, rn, rd)));
                }
                let tb = arr_of(insn, 0)?;
                let size = size_from_narrow(tb)?;
                let q = if is_2_form(code) { 1 } else { 0 };
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                return Ok(Some(misc_word(false, q, u, size, opcode, rn, rd)));
            }
            // SHLL/SHLL2: Vd.Ta, Vn.Tb, #shift. size from narrow Tb (operand1).
            ShllVec | Shll2Vec => {
                let tb = arr_of(insn, 1)?;
                let size = size_from_narrow(tb)?;
                let q = if code == Shll2Vec { 1u32 } else { 0 };
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                return Ok(Some(misc_word(false, q, u, size, opcode, rn, rd)));
            }
            // Pairwise-long SADDLP/UADDLP/SADALP/UADALP: Vd.Ta, Vn.Tb; size from
            // the narrow source Tb (operand 1).
            SaddlpVec | UaddlpVec | SadalpVec | UadalpVec => {
                let tb = arr_of(insn, 1)?;
                let size = size_of_arr(tb)?;
                let q = q_of_arr(tb)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                return Ok(Some(misc_word(false, q, u, size, opcode, rn, rd)));
            }
            _ => {}
        }

        // --- plain same-arrangement ops (REV*, CLS/CLZ/CNT, SUQADD/USQADD,
        //     SQABS/SQNEG, ABS/NEG) ---
        let (size, q) = if scalar {
            (size_for_eb(scalar_width(insn, 0)?)?, 0)
        } else {
            let a = arr_of(insn, 0)?;
            (size_of_arr(a)?, q_of_arr(a)?)
        };
        let rn = reg_num(insn, 1)?;
        let rd = reg_num(insn, 0)?;
        Ok(Some(misc_word(scalar, q, u, size, opcode, rn, rd)))
    }

    /// FP two-reg-misc + FP16 two-reg-misc + the FCVTL/FCVTN/FCVTXN specials.
    fn fp_two_reg_misc(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
        use Code::*;

        // Widen/narrow specials first (bespoke shapes).
        match code {
            FcvtlVec | Fcvtl2Vec => return Ok(Some(enc_fcvtl(insn, code)?)),
            FcvtnVec | Fcvtn2Vec => return Ok(Some(enc_fcvtn(insn, code)?)),
            FcvtxnVec | Fcvtxn2Vec => return Ok(Some(enc_fcvtxn(insn, code)?)),
            BfcvtnVec | Bfcvtn2Vec => return Ok(Some(enc_bfcvtn(insn, code)?)),
            F1cvtlVec | F1cvtl2Vec | F2cvtlVec | F2cvtl2Vec | Bf1cvtlVec | Bf1cvtl2Vec
            | Bf2cvtlVec | Bf2cvtl2Vec => return Ok(Some(enc_fp8_cvtl(insn, code)?)),
            _ => {}
        }

        // (U, a=word<23>, opcode5). a is the opcode-group bit; precision (sz =
        // word<22>) is recovered from operands separately.
        let Some((u, a, opcode)) = fp_misc_fields(code) else {
            return Ok(None);
        };
        let scalar = reg_of(insn, 0)?.class() == RegClass::ScalarFp;
        let half = is_half(insn, 0)?;

        let is_cmp_zero = matches!(code, FcmgtVec | FcmgeVec | FcmeqVec | FcmleVec | FcmltVec);

        if half {
            // FP16 two-reg-misc: bits<21:17>==11100. opcode at <16:12>, a=word<23>.
            let q = if scalar { 0 } else { q_of_arr(arr_of(insn, 0)?)? };
            let rn = reg_num(insn, 1)?;
            let rd = reg_num(insn, 0)?;
            let word = (q << 30)
                | (u << 29)
                | top(scalar, false)
                | (a << 23)
                // word<22> == 1 (the half-precision marker), word<21:17> == 11100.
                | (1 << 22)
                | (0b11100 << 17)
                | (opcode << 12)
                | (0b10 << 10)
                | (rn << 5)
                | rd;
            let _ = is_cmp_zero;
            return Ok(Some(word));
        }

        // Single/double: sz = word<22> (S=0, D=1).
        let sz = if scalar {
            match scalar_width(insn, 0)? {
                32 => 0u32,
                64 => 1,
                _ => return Err(EncodeError::InvalidOperand),
            }
        } else {
            match arr_of(insn, 0)? {
                VA::V2S | VA::V4S => 0,
                VA::V2D => 1,
                _ => return Err(EncodeError::InvalidOperand),
            }
        };
        let size = (a << 1) | sz;
        let q = if scalar { 0 } else { q_of_arr(arr_of(insn, 0)?)? };
        let rn = reg_num(insn, 1)?;
        let rd = reg_num(insn, 0)?;
        Ok(Some(misc_word(scalar, q, u, size, opcode, rn, rd)))
    }

    /// FP two-reg-misc `(U, a, opcode5)` table (excludes the widen/narrow
    /// specials and FRECPX which is scalar-only & handled here too).
    fn fp_misc_fields(code: Code) -> Option<(u32, u32, u32)> {
        use Code::*;
        let v = match code {
            FrintnVec => (0, 0, 0b11000),
            FrintmVec => (0, 0, 0b11001),
            FcvtnsVec => (0, 0, 0b11010),
            FcvtmsVec => (0, 0, 0b11011),
            FcvtasVec => (0, 0, 0b11100),
            ScvtfVec => (0, 0, 0b11101),
            Frint32zVec => (0, 0, 0b11110),
            Frint64zVec => (0, 0, 0b11111),
            FcmgtVec => (0, 1, 0b01100),
            FcmeqVec => (0, 1, 0b01101),
            FcmltVec => (0, 1, 0b01110),
            FabsVec => (0, 1, 0b01111),
            FrintpVec => (0, 1, 0b11000),
            FrintzVec => (0, 1, 0b11001),
            FcvtpsVec => (0, 1, 0b11010),
            FcvtzsVec => (0, 1, 0b11011),
            UrecpeVec => (0, 1, 0b11100),
            FrecpeVec => (0, 1, 0b11101),
            FrintaVec => (1, 0, 0b11000),
            FrintxVec => (1, 0, 0b11001),
            FcvtnuVec => (1, 0, 0b11010),
            FcvtmuVec => (1, 0, 0b11011),
            FcvtauVec => (1, 0, 0b11100),
            UcvtfVec => (1, 0, 0b11101),
            Frint32xVec => (1, 0, 0b11110),
            Frint64xVec => (1, 0, 0b11111),
            FcmgeVec => (1, 1, 0b01100),
            FcmleVec => (1, 1, 0b01101),
            FnegVec => (1, 1, 0b01111),
            FrintiVec => (1, 1, 0b11001),
            FcvtpuVec => (1, 1, 0b11010),
            FcvtzuVec => (1, 1, 0b11011),
            UrsqrteVec => (1, 1, 0b11100),
            FrsqrteVec => (1, 1, 0b11101),
            FsqrtVec => (1, 1, 0b11111),
            FrecpxVec => (0, 1, 0b11111), // scalar-only
            _ => return None,
        };
        Some(v)
    }

    /// FCVTL{2}: U=0,a=0,opcode=10111. Vd.Ta(wide), Vn.Tb(narrow). sz from Tb.
    fn enc_fcvtl(insn: &Instruction, code: Code) -> R {
        let tb = arr_of(insn, 1)?;
        let (sz, q) = match tb {
            VA::V4H => (0u32, 0u32),
            VA::V8H => (0, 1),
            VA::V2S => (1, 0),
            VA::V4S => (1, 1),
            _ => return Err(EncodeError::InvalidOperand),
        };
        let _ = code;
        let rn = reg_num(insn, 1)?;
        let rd = reg_num(insn, 0)?;
        Ok(misc_word(false, q, 0, sz, 0b10111, rn, rd))
    }

    /// FCVTN{2}: U=0,a=0,opcode=10110. Vd.Tb(narrow), Vn.Ta(wide). sz from Tb.
    fn enc_fcvtn(insn: &Instruction, code: Code) -> R {
        let tb = arr_of(insn, 0)?;
        let (sz, q) = match tb {
            VA::V4H => (0u32, 0u32),
            VA::V8H => (0, 1),
            VA::V2S => (1, 0),
            VA::V4S => (1, 1),
            _ => return Err(EncodeError::InvalidOperand),
        };
        let _ = code;
        let rn = reg_num(insn, 1)?;
        let rd = reg_num(insn, 0)?;
        Ok(misc_word(false, q, 0, sz, 0b10110, rn, rd))
    }

    /// FCVTXN{2}: U=1,opcode=10110. Vd.Tb(.2s/.4s), Vn.2d. sz from Tb (always
    /// single, sz=1). Scalar form: Sd, Dn.
    fn enc_fcvtxn(insn: &Instruction, code: Code) -> R {
        let scalar = reg_of(insn, 0)?.class() == RegClass::ScalarFp;
        if scalar {
            // FCVTXN Sd, Dn: U=1, sz=1, opcode 10110.
            let rn = reg_num(insn, 1)?;
            let rd = reg_num(insn, 0)?;
            return Ok(misc_word(true, 0, 1, 0b1, 0b10110, rn, rd));
        }
        let tb = arr_of(insn, 0)?;
        let q = match tb {
            VA::V2S => 0u32,
            VA::V4S => 1,
            _ => return Err(EncodeError::InvalidOperand),
        };
        let _ = code;
        let rn = reg_num(insn, 1)?;
        let rd = reg_num(insn, 0)?;
        // sz=1 (single<->double family).
        Ok(misc_word(false, q, 1, 0b1, 0b10110, rn, rd))
    }

    /// BFCVTN{2}: U=0, opcode 10110, size=10 (a=1, sz=0). Vd.4h/.8h, Vn.4s.
    /// Q selects the low (`.4h`, BFCVTN) vs high (`.8h`, BFCVTN2) result half.
    fn enc_bfcvtn(insn: &Instruction, code: Code) -> R {
        let q = if code == Code::Bfcvtn2Vec { 1u32 } else { 0 };
        let rn = reg_num(insn, 1)?;
        let rd = reg_num(insn, 0)?;
        Ok(misc_word(false, q, 0, 0b10, 0b10110, rn, rd))
    }

    /// F1CVTL/F2CVTL/BF1CVTL/BF2CVTL (+L2): U=1, opcode 10111. Vd.8h, Vn.8b/.16b.
    /// `size` selects the variant; Q selects the low (`.8b`) vs high (`.16b`, the
    /// `2`-suffixed) source half.
    fn enc_fp8_cvtl(insn: &Instruction, code: Code) -> R {
        use Code::*;
        let (size, q) = match code {
            F1cvtlVec => (0b00u32, 0u32),
            F1cvtl2Vec => (0b00, 1),
            F2cvtlVec => (0b01, 0),
            F2cvtl2Vec => (0b01, 1),
            Bf1cvtlVec => (0b10, 0),
            Bf1cvtl2Vec => (0b10, 1),
            Bf2cvtlVec => (0b11, 0),
            Bf2cvtl2Vec => (0b11, 1),
            _ => return Err(EncodeError::Unsupported),
        };
        let rn = reg_num(insn, 1)?;
        let rd = reg_num(insn, 0)?;
        Ok(misc_word(false, q, 1, size, 0b10111, rn, rd))
    }

    // =======================================================================
    // Across lanes + scalar pairwise.
    // =======================================================================

    fn across_or_pairwise(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
        use Code::*;
        // Scalar pairwise: dest is scalar, source is a 2-element vector.
        let dst_scalar = reg_of(insn, 0)?.class() == RegClass::ScalarFp;

        match code {
            // ADDP (scalar pairwise) vs ADDV/... Need to distinguish AddpVec used
            // here (scalar pairwise) — but AddpVec is also three-same. Three-same
            // AddpVec has 3 register operands; scalar-pairwise has dest scalar +
            // one vector source (2 operands). int_three_same handled the 3-op
            // case already, so AddpVec reaching here with 2 ops is scalar-pairwise.
            AddpVec => {
                if insn.op_count() != 2 {
                    return Ok(None);
                }
                // scalar ADDP: U=0, opcode 11011, size=11, source .2d.
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                let word = scalar_pairwise_word(0, 0b11011, 0b11, rn, rd);
                Ok(Some(word))
            }
            // FP scalar pairwise reductions.
            FaddpVec | FmaxpVec | FminpVec | FmaxnmpVec | FminnmpVec => {
                // These codes also occur in FP three-same (3 operands). Scalar
                // pairwise has exactly 2 operands (scalar dst + vector src).
                if insn.op_count() != 2 || !dst_scalar {
                    return Ok(None);
                }
                let arr = arr_of(insn, 1)?;
                // Half (V2H, U==0) vs single (V2S, sz=0) vs double (V2D, sz=1).
                let (u, size, opcode) = scalar_pairwise_fp_fields(code, arr)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                Ok(Some(scalar_pairwise_word(u, opcode, size, rn, rd)))
            }
            // Integer across-lanes.
            SaddlvVec | SmaxvVec | SminvVec | AddvVec | UaddlvVec | UmaxvVec | UminvVec => {
                let (u, opcode) = match code {
                    SaddlvVec => (0u32, 0b00011u32),
                    SmaxvVec => (0, 0b01010),
                    SminvVec => (0, 0b11010),
                    AddvVec => (0, 0b11011),
                    UaddlvVec => (1, 0b00011),
                    UmaxvVec => (1, 0b01010),
                    _ => (1, 0b11010), // UminvVec
                };
                let src = arr_of(insn, 1)?;
                let size = size_of_arr(src)?;
                let q = q_of_arr(src)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                Ok(Some(across_word(q, u, size, opcode, rn, rd)))
            }
            // FP across-lanes.
            FmaxnmvVec | FmaxvVec | FminnmvVec | FminvVec => {
                let (a, is_nm) = match code {
                    FmaxnmvVec => (0u32, true),
                    FmaxvVec => (0, false),
                    FminnmvVec => (1, true),
                    _ => (1, false), // FminvVec
                };
                let opcode = if is_nm { 0b01100u32 } else { 0b01111 };
                let src = arr_of(insn, 1)?;
                // U==0 -> half (.4h/.8h, dest H); U==1 -> single (.4s, dest S).
                let (u, q) = match src {
                    VA::V4H => (0u32, 0u32),
                    VA::V8H => (0, 1),
                    VA::V4S => (1, 1),
                    _ => return Err(EncodeError::InvalidOperand),
                };
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                let word = (q << 30)
                    | (u << 29)
                    | top(false, false)
                    | (a << 23)
                    // word<21:17> == 11000 (bits 21,20 set).
                    | (0b11000 << 17)
                    | (opcode << 12)
                    | (0b10 << 10)
                    | (rn << 5)
                    | rd;
                Ok(Some(word))
            }
            _ => Ok(None),
        }
    }

    /// Scalar-pairwise word skeleton: `01 U 11110 size 11000 opcode 10 Rn Rd`.
    /// (scalar top=11110, word<21:17>=11000.)
    #[inline]
    fn scalar_pairwise_word(u: u32, opcode: u32, size: u32, rn: u32, rd: u32) -> u32 {
        (u << 29)
            | top(true, false)
            | (size << 22)
            | (0b11000 << 17)
            | (opcode << 12)
            | (0b10 << 10)
            | (rn << 5)
            | rd
    }

    /// `(U, size, opcode)` for an FP scalar-pairwise reduction, given the 2-elem
    /// source arrangement (V2H=half, V2S=single, V2D=double).
    fn scalar_pairwise_fp_fields(code: Code, arr: VA) -> Result<(u32, u32, u32), EncodeError> {
        use Code::*;
        // half: U==0; single/double: U==1. size<1> selects min vs max; size<0>
        // selects S vs D (for the U==1 family).
        let is_min = matches!(code, FminpVec | FminnmpVec);
        let opcode = match code {
            FaddpVec => 0b01101u32,
            FmaxnmpVec | FminnmpVec => 0b01100,
            _ => 0b01111, // FMAXP / FMINP
        };
        match arr {
            VA::V2H => {
                let size = if is_min { 0b10 } else { 0b00 };
                Ok((0, size, opcode))
            }
            VA::V2S => {
                let size = if is_min { 0b10 } else { 0b00 };
                Ok((1, size, opcode))
            }
            VA::V2D => {
                let size = if is_min { 0b11 } else { 0b01 };
                Ok((1, size, opcode))
            }
            _ => Err(EncodeError::InvalidOperand),
        }
    }

    /// Across-lanes word skeleton: `q u 01110 size 11000 opcode 10 Rn Rd`.
    #[inline]
    fn across_word(q: u32, u: u32, size: u32, opcode: u32, rn: u32, rd: u32) -> u32 {
        (q << 30)
            | (u << 29)
            | top(false, false)
            | (size << 22)
            | (0b11000 << 17)
            | (opcode << 12)
            | (0b10 << 10)
            | (rn << 5)
            | rd
    }

    // =======================================================================
    // By element.
    // =======================================================================

    fn by_element(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
        use Code::*;
        // The codes that have by-element forms. Some share codes with three-same
        // (e.g. MulVec/FmulVec/SqdmulhVec/FcmlaVec/SqrdmlahVec); by_element only
        // applies when the last operand is an indexed element.
        if !is_by_element(insn) {
            return Ok(None);
        }

        // FCMLA by element (has trailing #rot) — handle first.
        if code == FcmlaVec {
            return Ok(Some(enc_by_element_fcmla(insn)?));
        }

        // FP8 / I8MM / BF16 by-element dot-product and widening MLAL.
        if let Some(w) = ext_dot_mlal_byel(insn, code)? {
            return Ok(Some(w));
        }

        // (U, opcode4) for by-element.
        let fields = by_element_fields(code);
        let Some((u, opcode)) = fields else {
            return Ok(None);
        };

        let scalar = reg_of(insn, 0)?.class() == RegClass::ScalarFp;

        // Determine the kind to drive operand shaping + index encoding.
        match code {
            // SDOT/UDOT by element: Vm.4b[index], size==10, H:L index, 5-bit Vm.
            SdotIdx | UdotIdx => {
                let q = q_of_arr(arr_of(insn, 0)?)?;
                let (vm, index) = idx_dot(insn)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                // Vm.4b[index]: index = H:L (2-bit), Vm is 5-bit so M = Vm<4>.
                let (h, l) = ((index >> 1) & 1, index & 1);
                let m = (vm >> 4) & 1;
                let word = by_elem_word(q, u, 0b10, opcode, h as u32, l as u32, m, vm, rn, rd);
                return Ok(Some(word));
            }
            // FMLAL/FMLSL/FMLAL2/FMLSL2 by element: half-index (H:L:M, 4-bit Vm),
            // Vd.<2s/4s>, Vn.<2h/4h>.
            FmlalVec | FmlslVec | Fmlal2Vec | Fmlsl2Vec => {
                let ta = arr_of(insn, 0)?;
                let q = match ta {
                    VA::V2S => 0u32,
                    VA::V4S => 1,
                    _ => return Err(EncodeError::InvalidOperand),
                };
                let (vm, index) = idx_half(insn)?;
                let rn = reg_num(insn, 1)?;
                let rd = reg_num(insn, 0)?;
                let (h, l, m) = ((index >> 2) & 1, (index >> 1) & 1, index & 1);
                // FMLAL/FMLSL by element fix `sz` so word<23:22> == 10 (the
                // widening single-from-half slot); the index uses the H:L:M
                // half-element form (4-bit Vm).
                let word =
                    by_elem_word(q, u, 0b10, opcode, h as u32, l as u32, m as u32, vm, rn, rd);
                return Ok(Some(word));
            }
            _ => {}
        }

        // The remaining families: SameInt (MUL/MLA/MLS), SameFp (FMLA/FMLS/FMUL/
        // FMULX), LongInt/LongSat, SatSame. The element size + index come from
        // the indexed operand.
        let is_fp = matches!(code, FmlaVec | FmlsVec | FmulVec | FmulxVec);

        if is_fp {
            return Ok(Some(enc_by_element_fp(insn, u, opcode, scalar)?));
        }

        // Integer by-element: size from the indexed element arrangement (the
        // indexed operand carries a `.h`/`.s`/`.d` element-size arrangement).
        let idx_arr = arr_of(insn, insn.op_count() - 1)?;
        let size = idx_arr_size(idx_arr)?;
        let (vm, index) = idx_for_size(insn, size)?;
        let q = if scalar { 0 } else {
            // For long forms, arrangement of dst is the wide one; recover Q from
            // whether code is a 2-form. For same-int, Q from dst arrangement.
            by_element_q(insn, code)?
        };
        let (h, l, m) = idx_hlm(size, vm, index);
        let rn = reg_num(insn, 1)?;
        let rd = reg_num(insn, 0)?;
        let word = if scalar {
            by_elem_word_scalar(u, size, opcode, h, l, m, vm, rn, rd)
        } else {
            by_elem_word(q, u, size, opcode, h, l, m, vm, rn, rd)
        };
        Ok(Some(word))
    }

    /// FP8 / I8MM / BF16 by-element dot-product and widening MLAL. Inverse of
    /// `decode::simd_fp::simd_arith::by_element_ext`. Returns `Ok(None)` for codes
    /// outside this family. Caller guarantees the operand is by-element.
    fn ext_dot_mlal_byel(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
        use Code::*;
        // Bail out before touching operands for codes outside this family — other
        // by-element-shaped instructions (e.g. INS `mov v.h[i], w`) share this
        // dispatch and must not have their operands mis-read.
        if !matches!(
            code,
            FdotIdx
                | SudotIdx
                | UsdotIdx
                | BfdotIdx
                | BfmlalbVec
                | BfmlaltVec
                | FmlalbVec
                | FmlaltVec
                | FmlallbbVec
                | FmlallbtVec
                | FmlalltbVec
                | FmlallttVec
        ) {
            return Ok(None);
        }
        let last = insn.op_count() - 1;
        let rn = reg_num(insn, 1)?;
        let rd = reg_num(insn, 0)?;
        let vm = reg_num(insn, last)?;
        let index = lane_of(insn, last)?;
        let w = match code {
            FdotIdx => {
                // to-single (.2s/.4s) -> size00, .4b index (H:L, 5-bit Vm);
                // to-half (.4h/.8h) -> size01, .2b index (H:L:M, 4-bit Vm).
                match arr_of(insn, 0)? {
                    VA::V2S | VA::V4S => {
                        let q = q_of_arr(arr_of(insn, 0)?)?;
                        let (h, l, m) = idx_hlm(0b10, vm, index);
                        by_elem_word(q, 0, 0b00, 0b0000, h, l, m, vm, rn, rd)
                    }
                    VA::V4H | VA::V8H => {
                        let q = if arr_of(insn, 0)? == VA::V8H { 1 } else { 0 };
                        if vm > 0xf {
                            return Err(EncodeError::InvalidOperand);
                        }
                        let (h, l, m) = split_index(0b01, index);
                        by_elem_word(q, 0, 0b01, 0b0000, h, l, m, vm, rn, rd)
                    }
                    _ => return Err(EncodeError::InvalidOperand),
                }
            }
            SudotIdx | UsdotIdx => {
                let q = q_of_arr(arr_of(insn, 0)?)?;
                let (h, l, m) = idx_hlm(0b10, vm, index);
                let size = if code == SudotIdx { 0b00 } else { 0b10 };
                by_elem_word(q, 0, size, 0b1111, h, l, m, vm, rn, rd)
            }
            BfdotIdx => {
                let q = q_of_arr(arr_of(insn, 0)?)?;
                let (h, l, m) = idx_hlm(0b10, vm, index);
                by_elem_word(q, 0, 0b01, 0b1111, h, l, m, vm, rn, rd)
            }
            BfmlalbVec | BfmlaltVec => {
                // size11, opcode1111, .h index (H:L:M, 4-bit Vm); B/T = Q.
                let q = if code == BfmlaltVec { 1 } else { 0 };
                if vm > 0xf {
                    return Err(EncodeError::InvalidOperand);
                }
                let (h, l, m) = split_index(0b01, index);
                by_elem_word(q, 0, 0b11, 0b1111, h, l, m, vm, rn, rd)
            }
            FmlalbVec | FmlaltVec => {
                // size11, opcode0000, .b byte index (4-bit), 3-bit Vm; B/T = Q.
                let q = if code == FmlaltVec { 1 } else { 0 };
                byte_idx_word(q, 0, 0b11, 0b0000, index, vm, rn, rd)?
            }
            FmlallbbVec | FmlallbtVec | FmlalltbVec | FmlallttVec => {
                // U=1, opcode1000, .b byte index (4-bit), 3-bit Vm.
                let (q, size) = match code {
                    FmlallbbVec => (0, 0b00),
                    FmlallbtVec => (0, 0b01),
                    FmlalltbVec => (1, 0b00),
                    _ => (1, 0b01),
                };
                byte_idx_word(q, 1, size, 0b1000, index, vm, rn, rd)?
            }
            _ => return Ok(None),
        };
        Ok(Some(w))
    }

    /// By-element word builder for the FP8 `.b` byte-indexed forms (FMLALB/T and
    /// FMLALL*): a 4-bit index packed as `H(word<11>) : word<21> : word<20> :
    /// word<19>` with a 3-bit `Vm` in `word<18:16>`.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn byte_idx_word(
        q: u32,
        u: u32,
        size: u32,
        opcode: u32,
        index: u8,
        vm: u32,
        rn: u32,
        rd: u32,
    ) -> Result<u32, EncodeError> {
        if vm > 0x7 {
            return Err(EncodeError::InvalidOperand); // 3-bit Vm only.
        }
        let idx = index as u32;
        if idx > 0xf {
            return Err(EncodeError::InvalidImmediate);
        }
        let h = (idx >> 3) & 1; // word<11>
        let i2 = (idx >> 2) & 1; // word<21>
        let i1 = (idx >> 1) & 1; // word<20>
        let i0 = idx & 1; // word<19>
        Ok((q << 30)
            | (u << 29)
            | (0b01111 << 24)
            | (size << 22)
            | (i2 << 21)
            | (i1 << 20)
            | (i0 << 19)
            | ((vm & 0x7) << 16)
            | (opcode << 12)
            | (h << 11)
            | (rn << 5)
            | rd)
    }

    /// `(U, opcode4)` for a by-element code.
    fn by_element_fields(code: Code) -> Option<(u32, u32)> {
        use Code::*;
        let v = match code {
            FmlalVec => (0, 0b0000),
            FmlaVec => (0, 0b0001),
            SmlalVec | Smlal2Vec => (0, 0b0010),
            SqdmlalVec | Sqdmlal2Vec => (0, 0b0011),
            FmlslVec => (0, 0b0100),
            FmlsVec => (0, 0b0101),
            SmlslVec | Smlsl2Vec => (0, 0b0110),
            SqdmlslVec | Sqdmlsl2Vec => (0, 0b0111),
            MulVec => (0, 0b1000),
            FmulVec => (0, 0b1001),
            SmullVec | Smull2Vec => (0, 0b1010),
            SqdmullVec | Sqdmull2Vec => (0, 0b1011),
            SqdmulhVec => (0, 0b1100),
            SqrdmulhVec => (0, 0b1101),
            SdotIdx => (0, 0b1110),
            UdotIdx => (1, 0b1110),
            MlaVec => (1, 0b0000),
            UmlalVec | Umlal2Vec => (1, 0b0010),
            MlsVec => (1, 0b0100),
            UmlslVec | Umlsl2Vec => (1, 0b0110),
            Fmlal2Vec => (1, 0b1000),
            FmulxVec => (1, 0b1001),
            UmullVec | Umull2Vec => (1, 0b1010),
            Fmlsl2Vec => (1, 0b1100),
            SqrdmlahVec => (1, 0b1101),
            SqrdmlshVec => (1, 0b1111),
            _ => return None,
        };
        Some(v)
    }

    /// By-element word skeleton: `0 Q U 01111 size L M Rm opcode H 0 Rn Rd`.
    /// (word<28:24>=01111 vector / 11111 scalar; word<10>=0.)
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn by_elem_word(
        q: u32,
        u: u32,
        size: u32,
        opcode: u32,
        h: u32,
        l: u32,
        m: u32,
        vm: u32,
        rn: u32,
        rd: u32,
    ) -> u32 {
        // Vm low 4 bits go to word<19:16>; bit M is word<20>.
        (q << 30)
            | (u << 29)
            | (0b01111 << 24)
            | (size << 22)
            | (l << 21)
            | (m << 20)
            | ((vm & 0xf) << 16)
            | (opcode << 12)
            | (h << 11)
            | (rn << 5)
            | rd
    }

    /// By-element word for the scalar (asisd) forms (top 11111).
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn by_elem_word_scalar(
        u: u32,
        size: u32,
        opcode: u32,
        h: u32,
        l: u32,
        m: u32,
        vm: u32,
        rn: u32,
        rd: u32,
    ) -> u32 {
        // The scalar (asisd) by-element forms fix word<30>==1 (their `01`
        // bits<31:30> prefix).
        (1 << 30)
            | (u << 29)
            | (0b11111 << 24)
            | (size << 22)
            | (l << 21)
            | (m << 20)
            | ((vm & 0xf) << 16)
            | (opcode << 12)
            | (h << 11)
            | (rn << 5)
            | rd
    }

    /// FP by-element (FMLA/FMLS/FMUL/FMULX): the FP precision is carried by the
    /// indexed element's element-size arrangement (`.h`->size 00, `.s`->size 10,
    /// `.d`->size 11). Vd/Vn use the matching FP vector/scalar view; `Q` comes
    /// from the destination arrangement (vector) or is 0 (scalar).
    fn enc_by_element_fp(insn: &Instruction, u: u32, opcode: u32, scalar: bool) -> R {
        let last = insn.op_count() - 1;
        let idx_arr = arr_of(insn, last)?;
        let rn = reg_num(insn, 1)?;
        let rd = reg_num(insn, 0)?;
        let vm = reg_num(insn, last)?;
        let index = lane_of(insn, last)?;
        let q = if scalar { 0 } else { q_of_arr(arr_of(insn, 0)?)? };
        // The FP precision element size determines size + index packing.
        match idx_arr.element_bits() {
            16 => {
                // half: size==00, index H:L:M (3-bit), 4-bit Vm.
                if vm > 0xf {
                    return Err(EncodeError::InvalidOperand);
                }
                let (h, l, m) = split_index(0b01, index); // .h index packing
                if scalar {
                    Ok(by_elem_word_scalar(u, 0b00, opcode, h, l, m, vm, rn, rd))
                } else {
                    Ok(by_elem_word(q, u, 0b00, opcode, h, l, m, vm, rn, rd))
                }
            }
            32 => {
                // single: size==10, index H:L, 5-bit Vm (M = Vm<4>).
                let (h, l, m) = idx_hlm(0b10, vm, index);
                if scalar {
                    Ok(by_elem_word_scalar(u, 0b10, opcode, h, l, m, vm, rn, rd))
                } else {
                    Ok(by_elem_word(q, u, 0b10, opcode, h, l, m, vm, rn, rd))
                }
            }
            64 => {
                // double: size==11, index H, 5-bit Vm (M = Vm<4>).
                let (h, l, m) = idx_hlm(0b11, vm, index);
                if scalar {
                    Ok(by_elem_word_scalar(u, 0b11, opcode, h, l, m, vm, rn, rd))
                } else {
                    Ok(by_elem_word(q, u, 0b11, opcode, h, l, m, vm, rn, rd))
                }
            }
            _ => Err(EncodeError::InvalidOperand),
        }
    }

    /// FCMLA by element: `0 Q 1 01111 size L M Rm 0 RR 1 H 0 Rn Rd` (rot RR =
    /// word<14:13>, H=word<11>, L=word<21>). Arrangements `.4h/.8h` (size=01) or
    /// `.4s` (size=10, Q=1).
    fn enc_by_element_fcmla(insn: &Instruction) -> R {
        let a = arr_of(insn, 0)?;
        let rot = (imm_u(insn, 3)? / 90) as u32;
        if rot > 3 {
            return Err(EncodeError::InvalidImmediate);
        }
        let rn = reg_num(insn, 1)?;
        let rd = reg_num(insn, 0)?;
        let idx = lane_of(insn, 2)?;
        let vm = reg_num(insn, 2)?;
        let (size, q, l, h, m) = match a {
            VA::V4H | VA::V8H => {
                // size=01, index=H:L. vm 5-bit. l=index<0>, h=index<1>.
                let q = if a == VA::V8H { 1u32 } else { 0 };
                ((0b01u32), q, (idx as u32) & 1, ((idx as u32) >> 1) & 1, (vm >> 4) & 1)
            }
            VA::V4S => {
                // size=10, Q=1, index=H. h=index<0>, l=0, m=vm<4>.
                (0b10u32, 1u32, 0u32, (idx as u32) & 1, (vm >> 4) & 1)
            }
            _ => return Err(EncodeError::InvalidOperand),
        };
        // word<15:13> structure: word<15>=0, rot=word<14:13>. word<12>=0,
        // word<11>=H? Actually layout: 0 RR 1 H. So word<15>=0, word<14:13>=rot,
        // word<12>=1, word<11>=H, word<10>=0. But wait decoder reads rot=word<14:13>
        // and h=word<11>, and the opcode pattern for fcmla-by-element is
        // (opcode & 0b1001)==0b0001 where opcode=word<15:12>. So word<15:12> =
        // 0b0RR1 -> word<15>=0, word<14:13>=rot, word<12>=1.
        let word = (q << 30)
            | (1 << 29)
            | (0b01111 << 24)
            | (size << 22)
            | (l << 21)
            | (m << 20)
            | ((vm & 0xf) << 16)
            | (rot << 13)
            | (1 << 12)
            | (h << 11)
            | (rn << 5)
            | rd;
        Ok(word)
    }

    /// Recover Q for an integer by-element form. Long (`…l`) forms read Q from the
    /// `2`-ness of the code; same-int read Q from the dst arrangement.
    fn by_element_q(insn: &Instruction, code: Code) -> Result<u32, EncodeError> {
        use Code::*;
        let is_long = matches!(
            code,
            SmlalVec
                | Smlal2Vec
                | SmlslVec
                | Smlsl2Vec
                | SmullVec
                | Smull2Vec
                | UmlalVec
                | Umlal2Vec
                | UmlslVec
                | Umlsl2Vec
                | UmullVec
                | Umull2Vec
                | SqdmlalVec
                | Sqdmlal2Vec
                | SqdmlslVec
                | Sqdmlsl2Vec
                | SqdmullVec
                | Sqdmull2Vec
        );
        if is_long {
            Ok(if is_2_form(code) { 1 } else { 0 })
        } else {
            q_of_arr(arr_of(insn, 0)?)
        }
    }

    /// Decode the indexed `(vm, index)` for a `.s`/`.d` (5-bit Vm) by-element op.
    /// `size==01` uses the 4-bit/half path instead (see [`idx_half`]).
    fn idx_for_size(insn: &Instruction, size: u32) -> Result<(u32, u8), EncodeError> {
        let last = insn.op_count() - 1;
        let vm = reg_num(insn, last)?;
        let index = lane_of(insn, last)?;
        // size==01 -> 4-bit Vm + 3-bit index; else 5-bit Vm.
        if size == 0b01 {
            if vm > 0xf {
                return Err(EncodeError::InvalidOperand);
            }
        } else if vm > 0x1f {
            return Err(EncodeError::InvalidOperand);
        }
        Ok((vm, index))
    }

    /// Decode the half-precision indexed element (H:L:M, 4-bit Vm).
    fn idx_half(insn: &Instruction) -> Result<(u32, u8), EncodeError> {
        let last = insn.op_count() - 1;
        let vm = reg_num(insn, last)?;
        let index = lane_of(insn, last)?;
        if vm > 0xf {
            return Err(EncodeError::InvalidOperand);
        }
        Ok((vm, index))
    }

    /// Decode the dot-product indexed element (Vm.4b[index], H:L 2-bit index,
    /// 5-bit Vm).
    fn idx_dot(insn: &Instruction) -> Result<(u32, u8), EncodeError> {
        let last = insn.op_count() - 1;
        let vm = reg_num(insn, last)?;
        let index = lane_of(insn, last)?;
        Ok((vm, index))
    }

    /// The 2-bit element `size` for a by-element indexed operand's element-size
    /// arrangement (`.h`->01, `.s`->10, `.d`->11). The indexed operand carries a
    /// scalable element-size arrangement (`Sh`/`Ss`/`Sd`) or the full `V*` form.
    fn idx_arr_size(a: VA) -> Result<u32, EncodeError> {
        Ok(match a.element_bits() {
            16 => 0b01,
            32 => 0b10,
            64 => 0b11,
            _ => return Err(EncodeError::InvalidOperand),
        })
    }

    /// Split an index into `(H, L, M)` bits per the element `size` (mirrors
    /// `decode_index`): `.h`(01)=H:L:M, `.s`(10)=H:L, `.d`(11)=H.
    fn split_index(size: u32, index: u8) -> (u32, u32, u32) {
        let i = index as u32;
        match size {
            0b01 => ((i >> 2) & 1, (i >> 1) & 1, i & 1),
            0b10 => ((i >> 1) & 1, i & 1, 0),
            _ => (i & 1, 0, 0),
        }
    }

    /// Compute the `(H, L, M)` bits for a by-element op with a *5-bit* `Vm`
    /// (`.s`/`.d` and the dot-product `.4b` forms). The architecture packs `Vm`
    /// as `M:Rmlo`, so word<20> (`M`) is `Vm<4>`; the element index supplies only
    /// `H` (`.d`/dot) or `H:L` (`.s`). For `.h` (size 01) `Vm` is 4-bit and the
    /// index supplies `H:L:M`, so we defer to [`split_index`].
    fn idx_hlm(size: u32, vm: u32, index: u8) -> (u32, u32, u32) {
        if size == 0b01 {
            return split_index(size, index);
        }
        let (h, l, _) = split_index(size, index);
        (h, l, (vm >> 4) & 1)
    }

    // =======================================================================
    // Shared predicates.
    // =======================================================================

    /// `true` if operand `n` is a half-precision element (scalar H or `.4h`/`.8h`
    /// /`.2h` vector).
    fn is_half(insn: &Instruction, n: usize) -> Result<bool, EncodeError> {
        match insn.op(n) {
            Operand::Reg { reg, arr, .. } => {
                if let Some(a) = arr {
                    Ok(matches!(a, VA::V4H | VA::V8H | VA::V2H))
                } else {
                    Ok(reg.class() == RegClass::ScalarFp && reg.width_bits() == 16)
                }
            }
            _ => Err(EncodeError::InvalidOperand),
        }
    }

    /// `true` if operand 2 exists and is a plain (non-indexed) register — the
    /// shape of a genuine three-same / three-different form, as opposed to the
    /// two-reg-misc compare-against-zero and scalar-pairwise forms that share a
    /// `Code` but have a `#0`/`#0.0` immediate or a vector reduction source.
    fn is_three_reg(insn: &Instruction) -> bool {
        if insn.op_count() < 3 {
            return false;
        }
        matches!(insn.op(2), Operand::Reg { lane: None, .. })
    }

    /// `true` if any operand is an indexed vector element (`Vm.Ts[i]`), which
    /// marks a by-element form. (Most by-element ops carry the index in the last
    /// operand, but FCMLA-by-element has a trailing `#rot`, so we scan all slots.)
    fn is_by_element(insn: &Instruction) -> bool {
        (0..insn.op_count()).any(|i| matches!(insn.op(i), Operand::Reg { lane: Some(_), .. }))
    }
}
