//! Encoder for Data Processing -- Scalar Floating-Point and Advanced SIMD — the
//! inverse of [`crate::decode::simd_fp`].
//!
//! Dispatches on [`Instruction::code`] (the canonical encoding identity), then
//! recovers the raw bit-fields (element `size`, `Q`, `U`, opcode selectors,
//! register numbers, indexes and immediates) from the structured operands the
//! decoder produced, and packs the word in reverse. It reconstructs the word
//! purely from the instruction's semantics — it never reads
//! [`Instruction::word`].
//!
//! The group is enormous (scalar FP conv/dp/compare/select/imm, Advanced SIMD
//! three-same / three-different / two-reg-misc / across-lanes / by-element /
//! copy / permute / extract / table / modified-immediate / shift-by-immediate,
//! and the crypto block). Each family encoder mirrors its decoder one-to-one:
//! the same field math, run backwards. Anything inconsistent returns
//! [`EncodeError`]; nothing panics.

use crate::encode::bits::{encode_advsimd_movi64, encode_vfp_imm};
use crate::encode::EncodeError;
use crate::enums::{Condition, VectorArrangement as VA};
use crate::instruction::Instruction;
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::Operand;
use crate::register::{Register, RegClass};

type R = Result<u32, EncodeError>;

// ===========================================================================
// Group predicate + top-level dispatch.
// ===========================================================================

/// `true` for every [`Code`] produced by the Scalar-FP / Advanced-SIMD group
/// ([`crate::decode::simd_fp`]); the set this encoder covers.
#[inline]
pub fn is_simd_fp(code: Code) -> bool {
    use Code::*;
    matches!(
        code,
        // --- scalar FP: conversions fp<->fixed / fp<->int ---
        ScvtfFixedS32 | ScvtfFixedS64 | ScvtfFixedD32 | ScvtfFixedD64 | ScvtfFixedH32
        | ScvtfFixedH64 | UcvtfFixedS32 | UcvtfFixedS64 | UcvtfFixedD32 | UcvtfFixedD64
        | UcvtfFixedH32 | UcvtfFixedH64 | FcvtzsFixedS32 | FcvtzsFixedS64 | FcvtzsFixedD32
        | FcvtzsFixedD64 | FcvtzsFixedH32 | FcvtzsFixedH64 | FcvtzuFixedS32 | FcvtzuFixedS64
        | FcvtzuFixedD32 | FcvtzuFixedD64 | FcvtzuFixedH32 | FcvtzuFixedH64
        | ScvtfS32 | ScvtfS64 | ScvtfD32 | ScvtfD64 | ScvtfH32 | ScvtfH64
        | UcvtfS32 | UcvtfS64 | UcvtfD32 | UcvtfD64 | UcvtfH32 | UcvtfH64
        | FcvtzsScalarS32 | FcvtzsScalarS64 | FcvtzsScalarD32 | FcvtzsScalarD64
        | FcvtzsScalarH32 | FcvtzsScalarH64 | FcvtzuScalarS32 | FcvtzuScalarS64
        | FcvtzuScalarD32 | FcvtzuScalarD64 | FcvtzuScalarH32 | FcvtzuScalarH64
        | FcvtnsScalar | FcvtnuScalar | FcvtasScalar | FcvtauScalar | FcvtpsScalar
        | FcvtpuScalar | FcvtmsScalar | FcvtmuScalar | Fjcvtzs
        | FmovToGp32 | FmovFromGp32 | FmovToGp64 | FmovFromGp64 | FmovToGpH32 | FmovFromGpH32
        | FmovToGpH64 | FmovFromGpH64 | FmovTopToGp | FmovTopFromGp
        // --- scalar FP: dp1 ---
        | FmovS | FmovD | FmovH | FabsS | FabsD | FabsH | FnegS | FnegD | FnegH | FsqrtS
        | FsqrtD | FsqrtH | FcvtSD | FcvtSH | FcvtDS | FcvtDH | FcvtHS | FcvtHD | Bfcvt
        | FrintnS | FrintnD | FrintnH | FrintpS | FrintpD | FrintpH | FrintmS | FrintmD
        | FrintmH | FrintzS | FrintzD | FrintzH | FrintaS | FrintaD | FrintaH | FrintxS
        | FrintxD | FrintxH | FrintiS | FrintiD | FrintiH
        | Frint32zS | Frint32zD | Frint32xS | Frint32xD | Frint64zS | Frint64zD | Frint64xS
        | Frint64xD
        // --- scalar FP: dp2 / dp3 / compare / ccmp / sel / imm ---
        | FmulS | FmulD | FmulH | FdivS | FdivD | FdivH | FaddS | FaddD | FaddH | FsubS
        | FsubD | FsubH | FmaxS | FmaxD | FmaxH | FminS | FminD | FminH | FmaxnmS | FmaxnmD
        | FmaxnmH | FminnmS | FminnmD | FminnmH | FnmulS | FnmulD | FnmulH
        | FmaddS | FmaddD | FmaddH | FmsubS | FmsubD | FmsubH | FnmaddS | FnmaddD | FnmaddH
        | FnmsubS | FnmsubD | FnmsubH
        | FcmpS | FcmpD | FcmpH | FcmpeS | FcmpeD | FcmpeH | FccmpS | FccmpD | FccmpH
        | FccmpeS | FccmpeD | FccmpeH | FcselS | FcselD | FcselH | FmovImmS | FmovImmD
        | FmovImmH
        // --- Advanced SIMD: three-same int ---
        | ShaddVec | SqaddVec | SrhaddVec | ShsubVec | SqsubVec | CmgtVec | CmgeVec | SshlVec
        | SqshlVec | SrshlVec | SqrshlVec | SmaxVec | SminVec | SabdVec | SabaVec | AddVec
        | CmtstVec | MlaVec | MulVec | SmaxpVec | SminpVec | SqdmulhVec | AddpVec | UhaddVec
        | UqaddVec | UrhaddVec | UhsubVec | UqsubVec | CmhiVec | CmhsVec | UshlVec | UqshlVec
        | UrshlVec | UqrshlVec | UmaxVec | UminVec | UabdVec | UabaVec | SubVec | CmeqVec
        | MlsVec | PmulVec | UmaxpVec | UminpVec | SqrdmulhVec
        // --- three-same logical ---
        | AndVec | BicVec | OrrVec | OrnVec | EorVec | BslVec | BitVec | BifVec
        // --- three-same FP / FP16 ---
        | FmaxnmVec | FmlaVec | FaddVec | FmulxVec | FcmeqVec | FmaxVec | FrecpsVec
        | FminnmVec | FmlsVec | FsubVec | FminVec | FrsqrtsVec | FmaxnmpVec | FaddpVec
        | FmulVec | FcmgeVec | FacgeVec | FmaxpVec | FdivVec | FminnmpVec | FabdVec | FcmgtVec
        | FacgtVec | FminpVec
        // --- three-same widening / extra / complex ---
        | FmlalVec | FmlslVec | Fmlal2Vec | Fmlsl2Vec | SqrdmlahVec | SqrdmlshVec | FcmlaVec
        | FcaddVec | SdotVec | UdotVec | SdotIdx | UdotIdx
        // --- FP8 / I8MM / BF16 dot-product & widening MLAL (NEON) ---
        | FdotVec | FdotIdx | UsdotVec | UsdotIdx | SudotIdx | BfdotVec | BfdotIdx
        | NeonFdotF16Vec | NeonFdotF16Idx
        | BfmlalbVec | BfmlaltVec | FmlalbVec | FmlaltVec
        | FmlallbbVec | FmlallbtVec | FmlalltbVec | FmlallttVec
        | SmmlaVec | UmmlaVec | UsmmlaVec
        // --- FP/BF16/FP8 matrix multiply-accumulate (FMMLA/BFMMLA, NEON) ---
        | FmmlaVecF16F32 | FmmlaVecF16 | FmmlaVecF8F16 | FmmlaVecF8F32 | BfmmlaVec
        // --- FEAT_FAMINMAX / FEAT_FP8 / FEAT_LUT (NEON) ---
        | FamaxVec | FaminVec | FscaleVec | FcvtnFp8 | Fcvtn2Fp8 | BfcvtnVec | Bfcvtn2Vec
        | F1cvtlVec | F1cvtl2Vec | F2cvtlVec | F2cvtl2Vec | Bf1cvtlVec | Bf1cvtl2Vec
        | Bf2cvtlVec | Bf2cvtl2Vec | Luti2Vec | Luti4Vec | Luti4TwoVec
        // --- three-different ---
        | SaddlVec | Saddl2Vec | SaddwVec | Saddw2Vec | SsublVec | Ssubl2Vec | SsubwVec
        | Ssubw2Vec | AddhnVec | Addhn2Vec | SabalVec | Sabal2Vec | SubhnVec | Subhn2Vec
        | SabdlVec | Sabdl2Vec | SmlalVec | Smlal2Vec | SqdmlalVec | Sqdmlal2Vec | SmlslVec
        | Smlsl2Vec | SqdmlslVec | Sqdmlsl2Vec | SmullVec | Smull2Vec | SqdmullVec
        | Sqdmull2Vec | PmullVec | Pmull2Vec | UaddlVec | Uaddl2Vec | UaddwVec | Uaddw2Vec
        | UsublVec | Usubl2Vec | UsubwVec | Usubw2Vec | RaddhnVec | Raddhn2Vec | UabalVec
        | Uabal2Vec | RsubhnVec | Rsubhn2Vec | UabdlVec | Uabdl2Vec | UmlalVec | Umlal2Vec
        | UmlslVec | Umlsl2Vec | UmullVec | Umull2Vec
        // --- two-reg-misc int ---
        | Rev64Vec | Rev16Vec | Rev32Vec | SaddlpVec | UaddlpVec | SuqaddVec | UsqaddVec
        | ClsVec | ClzVec | CntVec | SadalpVec | UadalpVec | SqabsVec | SqnegVec | AbsVec
        | NegVec | CmltVec | CmleVec | XtnVec | Xtn2Vec | SqxtnVec | Sqxtn2Vec | SqxtunVec
        | Sqxtun2Vec | UqxtnVec | Uqxtn2Vec | MvnVec | RbitVec | ShllVec | Shll2Vec
        // --- two-reg-misc FP ---
        | FrintnVec | FrintmVec | FcvtnsVec | FcvtmsVec | FcvtasVec | ScvtfVec | Frint32zVec
        | Frint64zVec | FabsVec | FrintpVec | FrintzVec | FcvtpsVec | FcvtzsVec | UrecpeVec
        | FrecpeVec | FrintaVec | FrintxVec | FcvtnuVec | FcvtmuVec | FcvtauVec | UcvtfVec
        | Frint32xVec | Frint64xVec | FnegVec | FrintiVec | FcvtpuVec | FcvtzuVec | UrsqrteVec
        | FrsqrteVec | FsqrtVec | FcmleVec | FcmltVec | FrecpxVec | FcvtlVec | Fcvtl2Vec
        | FcvtnVec | Fcvtn2Vec | FcvtxnVec | Fcvtxn2Vec | ScvtfFixedVec | UcvtfFixedVec
        | FcvtzsFixedVec | FcvtzuFixedVec
        // --- across lanes ---
        | SaddlvVec | SmaxvVec | SminvVec | AddvVec | UaddlvVec | UmaxvVec | UminvVec
        | FmaxnmvVec | FmaxvVec | FminnmvVec | FminvVec
        // --- copy / permute / ext / table ---
        | DupElement | DupElementScalar | DupGeneral | InsGeneral | InsElement | Smov | Umov | Uzp1 | Uzp2
        | Trn1 | Trn2 | Zip1 | Zip2 | Ext | Tbl | Tbx
        // --- modified immediate ---
        | MoviVector | MvniVector | MoviScalarD | MoviVec2D | OrrVecImm | BicVecImm
        | FmovVecImmS | FmovVecImmH | FmovVecImmD2
        // --- shift by immediate (vector) ---
        | SshrVec | UshrVec | SsraVec | UsraVec | SrshrVec | UrshrVec | SrsraVec | UrsraVec
        | SriVec | ShlVec | SliVec | SqshluImmVec | SqshlImmVec | UqshlImmVec | ShrnVec
        | Shrn2Vec | RshrnVec | Rshrn2Vec | SqshrnVec | Sqshrn2Vec | SqrshrnVec | Sqrshrn2Vec
        | SqshrunVec | Sqshrun2Vec | SqrshrunVec | Sqrshrun2Vec | UqshrnVec | Uqshrn2Vec
        | UqrshrnVec | Uqrshrn2Vec | SshllVec | Sshll2Vec | UshllVec | Ushll2Vec | SxtlVec
        | Sxtl2Vec | UxtlVec | Uxtl2Vec
        // --- shift by immediate (scalar) ---
        | SshrScalar | UshrScalar | SsraScalar | UsraScalar | SrshrScalar | UrshrScalar
        | SrsraScalar | UrsraScalar | SriScalar | ShlScalar | SliScalar | SqshluImmScalar
        | SqshlImmScalar | UqshlImmScalar | SqshrnScalar | SqrshrnScalar | SqshrunScalar
        | SqrshrunScalar | UqshrnScalar | UqrshrnScalar | ScvtfFixedScalar | UcvtfFixedScalar
        | FcvtzsFixedScalar | FcvtzuFixedScalar
        // --- crypto ---
        | AdvAese | AdvAesd | AdvAesmc | AdvAesimc | AdvSha1c | AdvSha1p | AdvSha1m
        | AdvSha1su0 | AdvSha256h | AdvSha256h2 | AdvSha256su1 | AdvSha1h | AdvSha1su1
        | AdvSha256su0 | AdvSha512h | AdvSha512h2 | AdvSha512su1 | AdvRax1 | AdvSm3partw1
        | AdvSm3partw2 | AdvSm4ekey | AdvSha512su0 | AdvSm4e | AdvEor3 | AdvBcax | AdvSm3ss1
        | AdvSm3tt1a | AdvSm3tt1b | AdvSm3tt2a | AdvSm3tt2b | AdvXar
    )
}

/// Encode a Scalar-FP / Advanced-SIMD instruction.
#[inline]
pub fn encode(insn: &Instruction) -> R {
    let code = insn.code();
    if let Some(w) = scalar_fp::encode(insn, code)? {
        return Ok(w);
    }
    if let Some(w) = crypto::encode(insn, code)? {
        return Ok(w);
    }
    if let Some(w) = simd_arith::encode(insn, code)? {
        return Ok(w);
    }
    if let Some(w) = simd_data::encode(insn, code)? {
        return Ok(w);
    }
    Err(EncodeError::Unsupported)
}

// ===========================================================================
// Shared operand-reading helpers.
// ===========================================================================

/// The 5-bit register number of operand `n`, or `InvalidOperand` if it is not a
/// plain register-bearing operand.
#[inline]
fn reg_num(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    match insn.op(n) {
        Operand::Reg { reg, .. } => Ok(reg.number() as u32),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The register of operand `n` (any kind), or `InvalidOperand`.
#[inline]
fn reg_of(insn: &Instruction, n: usize) -> Result<Register, EncodeError> {
    match insn.op(n) {
        Operand::Reg { reg, .. } => Ok(reg),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The unsigned immediate value of operand `n`.
#[inline]
fn imm_u(insn: &Instruction, n: usize) -> Result<u64, EncodeError> {
    match insn.op(n) {
        Operand::ImmUnsigned(v) | Operand::ImmLogical(v) => Ok(v),
        Operand::ImmSigned(v) => Ok(v as u64),
        Operand::ShiftAmount(v) => Ok(v as u64),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The arrangement of register operand `n`, or `InvalidOperand` if absent.
#[inline]
fn arr_of(insn: &Instruction, n: usize) -> Result<VA, EncodeError> {
    match insn.op(n) {
        Operand::Reg { arr: Some(a), .. } => Ok(a),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The lane index of register operand `n`, or `InvalidOperand` if absent.
#[inline]
fn lane_of(insn: &Instruction, n: usize) -> Result<u8, EncodeError> {
    match insn.op(n) {
        Operand::Reg { lane: Some(l), .. } => Ok(l),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The `Condition` of operand `n`.
#[inline]
fn cond_of(insn: &Instruction, n: usize) -> Result<Condition, EncodeError> {
    match insn.op(n) {
        Operand::Cond(c) => Ok(c),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The `f32` value of an `FpImm` operand `n`.
#[inline]
fn fpimm_of(insn: &Instruction, n: usize) -> Result<f32, EncodeError> {
    match insn.op(n) {
        Operand::FpImm(f) => Ok(f),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// `Q` bit (0/1) for a SIMD arrangement: the 128-bit-wide arrangements map to
/// `Q == 1`, the 64-bit-wide ones to `Q == 0`. `InvalidImmediate` for the
/// scalable / `None` / `.1q`-like forms with no defined `Q`.
#[inline]
fn q_of_arr(a: VA) -> Result<u32, EncodeError> {
    Ok(match a {
        VA::V8B | VA::V4H | VA::V2S | VA::V1D | VA::V2H => 0,
        VA::V16B | VA::V8H | VA::V4S | VA::V2D => 1,
        _ => return Err(EncodeError::InvalidImmediate),
    })
}

/// The 2-bit integer `size` field for an arrangement element width
/// (`B->00 H->01 S->10 D->11`).
#[inline]
fn size_of_arr(a: VA) -> Result<u32, EncodeError> {
    Ok(match a {
        VA::V8B | VA::V16B => 0b00,
        VA::V4H | VA::V8H | VA::V2H => 0b01,
        VA::V2S | VA::V4S => 0b10,
        VA::V1D | VA::V2D => 0b11,
        _ => return Err(EncodeError::InvalidImmediate),
    })
}

/// The scalar-FP element width (bits) of register operand `n` (B=8 … Q=128),
/// from the register view. `InvalidOperand` if not a scalar-FP register.
#[inline]
fn scalar_width(insn: &Instruction, n: usize) -> Result<u16, EncodeError> {
    let r = reg_of(insn, n)?;
    if r.class() != RegClass::ScalarFp {
        return Err(EncodeError::InvalidOperand);
    }
    Ok(r.width_bits())
}

include!("simd_fp_scalar.rs");
include!("simd_fp_crypto.rs");
include!("simd_fp_arith.rs");
include!("simd_fp_data.rs");

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

    /// Decode then re-encode, requiring only that the re-encoded word decodes to
    /// a semantically-equal instruction (for the forms whose raw encoding carries
    /// bits the decoder legitimately discards, so the exact word is irrecoverable).
    #[track_caller]
    fn rt_semantic(word: u32) {
        let mut a = Instruction::default();
        crate::decode::decode_into(word, 0x1000, FeatureSet::ALL, &mut a);
        assert!(!a.is_invalid(), "word {word:#010x} failed to decode");
        let got = a.encode().expect("encode");
        let mut b = Instruction::default();
        crate::decode::decode_into(got, 0x1000, FeatureSet::ALL, &mut b);
        assert!(!b.is_invalid(), "re-encoded {got:#010x} did not decode");
        assert_eq!(a.code(), b.code());
        assert_eq!(a.mnemonic(), b.mnemonic());
        assert_eq!(a.op_count(), b.op_count());
        for i in 0..a.op_count() {
            assert_eq!(a.op(i), b.op(i), "operand {i} differs for {word:#010x}");
        }
    }

    #[test]
    fn scalar_fp_words() {
        rt(0x1E604101); // fmov d1, d8
        rt(0x1E20C2CF); // fabs s15, s22
        rt(0x1E22C0E0); // fcvt d0, s7
        rt(0x1E602800); // fadd d0, d0, d0
        rt(0x1F4E3209); // fmadd d9, d16, d14, d12
        rt(0x1E212090); // fcmpe s4, s1
        rt(0x1E202108); // fcmp s8, #0.0 (Rm field already zero)
        rt(0x1E743417); // fccmpe d0, d20, #0x7, lo
        rt(0x1E746C6A); // fcsel d10, d3, d20, vs
        rt(0x1E66700B); // fmov d11, #19.0
        rt(0x1E2602EA); // fmov w10, s23
        rt(0x9E660041); // fmov x1, d2
        rt(0x9EAE0041); // fmov x1, v2.d[1]
        rt(0x1E7802E5); // fcvtzs w5, d23
        rt(0x1E58ABAC); // fcvtzs w12, d29, #0x16 (fixed)
        rt(0x1E634041); // bfcvt h1, s2
    }

    #[test]
    fn advsimd_three_same_words() {
        rt(0x4EAE86AF); // add v15.4s, v21.4s, v14.4s
        rt(0x5EE687A0); // add d0, d29, d6 (scalar)
        rt(0x4E391F6A); // and v10.16b, v27.16b, v25.16b
        rt(0x0EAB1D68); // mov v8.8b, v11.8b (orr alias)
        rt(0x6EF7E4B8); // fcmgt v24.2d, v5.2d, v23.2d
        rt(0x6ED21776); // fabd v22.8h, ... (fp16)
    }

    #[test]
    fn advsimd_misc_diff_across_words() {
        rt(0x0E2343B2); // addhn v18.8b, v29.8h, v3.8h
        rt(0x4E78C1A1); // smull2 v1.4s, v13.8h, v24.8h
        rt(0x0E2098FE); // cmeq v30.8b, v7.8b, #0
        rt(0x4EA14A31); // sqxtn2 v17.4s, v17.2d
        rt(0x6E205A52); // mvn v18.16b, v18.16b
        rt(0x2E213961); // shll v1.8h, v11.8b, #8
        rt(0x4E31B9A0); // addv b0, v13.16b
        rt(0x5EF1B94B); // addp d11, v10.2d (scalar pairwise)
        rt(0x4EE1D95A); // frecpe v26.2d, v10.2d
        rt(0x7E6168A4); // fcvtxn s4, d5 (scalar)
    }

    #[test]
    fn advsimd_by_element_words() {
        rt(0x4FA39897); // fmul v23.4s, v4.4s, v3.s[3]
        rt(0x4F2913F6); // fmla v22.8h, v31.8h, v9.h[2]
        rt(0x4F8E00D4); // fmlal v20.4s, v6.4h, v14.h[0]
        rt(0x0F9BE85D); // sdot v29.2s, v2.8b, v27.4b[2]
        rt(0x6F52302B); // fcmla v11.8h, v1.8h, v18.h[0], #0x5a
    }

    #[test]
    fn advsimd_data_words() {
        rt(0x4E0506E3); // dup v3.16b, v23.b[2]
        rt(0x0E062EE2); // smov w2, v23.h[1]
        rt(0x0E4B3AF4); // zip1 v20.4h, v23.4h, v11.4h
        rt(0x6E080B29); // ext v9.16b, v25.16b, v8.16b, #1
        rt(0x4E1101A3); // tbl v3.16b, {v13.16b}, v17.16b
        rt(0x4F04E52A); // movi v10.16b, #0x89
        rt(0x0F04D6D1); // movi v17.2s, #0x96, msl #0x10
        rt(0x2F05E64B); // movi d11, #...
        rt(0x4F00150A); // orr v10.4s, #8
        rt(0x0F04F40D); // fmov v13.2s, #-2.0
        rt(0x4F4256B3); // shl v19.2d, v21.2d, #2
        rt(0x0F08A790); // sxtl v16.8h, v28.8b
        rt(0x5F450550); // sshr d16, d10, #0x3b (scalar)
    }

    #[cfg(feature = "crypto")]
    #[test]
    fn crypto_words() {
        rt(0x4E284BE7); // aese v7.16b, v31.16b
        rt(0x5E1B00ED); // sha1c q13, s7, v27.4s
        rt(0xCE6A81DE); // sha512h q30, q14, v10.2d
        rt(0xCE0B75E2); // eor3 v2.16b, v15.16b, v11.16b, v29.16b
        rt(0xCE8FCBA8); // xar v8.2d, v29.2d, v15.2d, #0x32
        rt(0xCE5FB3F5); // sm3tt1a v21.4s, v31.4s, v31.s[3]
    }

    #[test]
    fn discarded_bits_are_semantic() {
        // DUP (general): the index bits of imm5 are don't-care.
        rt_semantic(0x4E030D41);
        // FCMP #0.0 with a non-zero (ignored) Rm field.
        rt_semantic(0x1E7F22F8);
        // INS (element): the low sub-bits of imm4 are don't-care.
        rt_semantic(0x6E060CE4);
    }
}
