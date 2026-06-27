//! Encoder for Loads and Stores — the exact inverse of [`crate::decode::ldst`].
//!
//! Dispatches on [`Instruction::code`] (the canonical encoding identity), then
//! recovers the raw bit-fields the decoder produced. Memory operands
//! ([`Operand::MemImm`] / [`Operand::MemExt`]) are unpacked back into their
//! `Rn`/`imm`/`option`/`S` fields; ip-relative literal loads recover the 19-bit
//! immediate from the absolute [`Operand::Label`] and [`Instruction::ip`]. It
//! reconstructs the word purely from semantics — it never reads
//! [`Instruction::word`].
//!
//! ## How the inversion is organized
//!
//! The decoder classifies a single load/store register form from `(size, V,
//! opc)` into a [`Code`]; the encoder carries the inverse table — each `Code`
//! maps back to `(size, v, opc)` plus the addressing-mode "variant" (unsigned,
//! unscaled, imm pre/post, unprivileged, register-offset). The pair, exclusive,
//! ordered, atomic, CAS/CASP, PAC, RCpc-unscaled and memory-tagging families are
//! inverted directly from their fixed `Code`->fields mapping.

use crate::encode::ldst_simd;
use crate::encode::EncodeError;
use crate::instruction::Instruction;
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::{MemIndexMode, Operand};
use crate::register::Register;
use crate::sysop::SysToken;

type R = Result<u32, EncodeError>;

/// `true` for every [`Code`] handled by this Loads-and-Stores encoder (including
/// the Advanced SIMD load/store *structure* forms routed to [`ldst_simd`]). Used
/// by [`crate::encode::encode`] to dispatch the whole group here.
#[inline]
pub fn is_ldst(code: Code) -> bool {
    use Code::*;
    if matches!(
        code,
        LdrLit32 | LdrLit64 | LdrswLit | PrfmLit | LdrLitFp32 | LdrLitFp64 | LdrLitFp128
            | RprfmReg
            | Ldaprb | Ldaprh | Ldapr32 | Ldapr64
            | LdraaOff | LdraaPre | LdrabOff | LdrabPre
            | Ld64b | St64b | St64bv | St64bv0
            // FEAT_LRCPC3 SIMD&FP LDAPUR/STLUR + LDIAPP/STILP pair + writeback
            // STLR/LDAPR.
            | LdapurFp8 | LdapurFp16 | LdapurFp32 | LdapurFp64 | LdapurFp128
            | StlurFp8 | StlurFp16 | StlurFp32 | StlurFp64 | StlurFp128
            | LdiappOff | LdiappPost | StilpOff | StilpPre
            | StlrPre32 | StlrPre64 | LdaprPost32 | LdaprPost64
            | LdappPair | LdapPair | StlpPair
            // FEAT_GCS stores.
            | Gcsstr | Gcssttr
    ) {
        return true;
    }
    reg_form(code).is_some()
        || unpriv_fields(code).is_some()
        || pair_kind(code).is_some()
        || excl_single_fields(code).is_some()
        || excl_pair_fields(code).is_some()
        || ordered_fields(code).is_some()
        || cas_fields(code).is_some()
        || casp_fields(code).is_some()
        || lsui_excl_single_fields(code).is_some()
        || lsui_cas_fields(code).is_some()
        || lsui_casp_fields(code).is_some()
        || swp_fields(code).is_some()
        || atomic_opc(code).is_some()
        || lsfe_fields(code).is_some()
        || rcw_single_fields(code).is_some()
        || the_atomic_fields(code).is_some()
        || ldapstl_fields(code).is_some()
        || tag_is(code)
        || ldst_simd::is_ldst_simd(code)
}

/// Encode a Load/Store instruction.
#[inline]
pub fn encode(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();
    match code {
        // --- FEAT_RPRFM range prefetch. ---
        RprfmReg => enc_rprfm(insn),

        // --- Load register (literal). ---
        LdrLit32 | LdrLit64 | LdrswLit | PrfmLit | LdrLitFp32 | LdrLitFp64 | LdrLitFp128 => {
            enc_literal(insn)
        }

        // --- Load/store register: unsigned immediate offset. ---
        _ if reg_variant(code) == Some(RegVariant::Unsigned) => {
            // The FP *unsigned* codes are reused by the unscaled and pre/post
            // addressing forms (the decoder re-tags only the mnemonic + memory
            // operand). Intercept those before the unsigned-offset path.
            if let Some(r) = enc_fp_unscaled_or_idx(insn) {
                return r;
            }
            enc_reg_unsigned(insn)
        }
        // --- Load/store register: register offset. ---
        _ if reg_variant(code) == Some(RegVariant::RegOff) => enc_reg_offset(insn),
        // --- Load/store register: unscaled (LDUR/STUR/PRFUM). ---
        _ if reg_variant(code) == Some(RegVariant::Unscaled) => enc_reg_unscaled(insn),
        // --- Load/store register: immediate pre/post-index. ---
        _ if reg_variant(code) == Some(RegVariant::ImmPost)
            || reg_variant(code) == Some(RegVariant::ImmPre) =>
        {
            enc_reg_immidx(insn)
        }
        // --- Load/store register: unprivileged (LDTR/STTR). ---
        _ if unpriv_fields(code).is_some() => enc_reg_unpriv(insn),

        // --- Load/store pair (incl. NP, SIMD&FP, STGP). ---
        _ if pair_kind(code).is_some() => enc_pair(insn),

        // --- Load/store exclusive / ordered. ---
        _ if excl_single_fields(code).is_some() => enc_excl_single(insn),
        _ if excl_pair_fields(code).is_some() => enc_excl_pair(insn),
        _ if ordered_fields(code).is_some() => enc_ordered(insn),

        // --- CAS / CASP (LSE). ---
        _ if cas_fields(code).is_some() => enc_cas(insn),
        _ if casp_fields(code).is_some() => enc_casp(insn),

        // --- FEAT_LSUI unprivileged atomics. ---
        _ if lsui_excl_single_fields(code).is_some() => enc_lsui_excl_single(insn),
        _ if lsui_cas_fields(code).is_some() => enc_lsui_cas(insn),
        _ if lsui_casp_fields(code).is_some() => enc_lsui_casp(insn),

        // --- LSE atomics: SWP and LD<op>/ST<op>. ---
        _ if swp_fields(code).is_some() => enc_swp(insn),
        _ if atomic_fields(code).is_some() => enc_atomic(insn),
        // --- FEAT_LSFE atomic floating-point in-memory (LDF*/STF*/LDBF*/STBF*). ---
        _ if lsfe_fields(code).is_some() => enc_lsfe(insn),
        // --- FEAT_THE single-register RCW RMW (RCWCLR/RCWSWP/RCWSET + RCWS*). ---
        _ if rcw_single_fields(code).is_some() => enc_rcw_single(insn),
        // --- FEAT_THE / FEAT_LSE128 atomics (LDTADD/SWPT, RCW*, LDCLRP/...). ---
        _ if the_atomic_fields(code).is_some() => enc_the_atomic(insn),
        // --- LDAPR/LDAPRB/LDAPRH (FEAT_LRCPC). ---
        Ldaprb | Ldaprh | Ldapr32 | Ldapr64 => enc_ldapr(insn),

        // --- FEAT_LS64 single-copy atomic 64-byte ops. ---
        Ld64b | St64b | St64bv | St64bv0 => enc_ls64(insn),

        // --- FEAT_GCS stores (GCSSTR/GCSSTTR). ---
        Gcsstr | Gcssttr => enc_gcsstr(insn),

        // --- Pointer-authenticated LDRAA/LDRAB. ---
        LdraaOff | LdraaPre | LdrabOff | LdrabPre => enc_pac(insn),

        // --- LDAPUR/STLUR (RCpc unscaled). ---
        _ if ldapstl_fields(code).is_some() => enc_ldapstl(insn),

        // --- FEAT_LRCPC3 SIMD&FP LDAPUR/STLUR. ---
        LdapurFp8 | LdapurFp16 | LdapurFp32 | LdapurFp64 | LdapurFp128 | StlurFp8 | StlurFp16
        | StlurFp32 | StlurFp64 | StlurFp128 => enc_fp_ldapstl(insn),

        // --- FEAT_LRCPC3 LDIAPP/STILP (ordered pair). ---
        LdiappOff | LdiappPost | StilpOff | StilpPre => enc_ldiapp_stilp(insn),

        // --- FEAT_LRCPC3 LDAPP/LDAP/STLP (ordered pair, X-only, no offset). ---
        LdappPair | LdapPair | StlpPair => enc_ldapp_stlp(insn),

        // --- FEAT_LRCPC3 writeback STLR (pre) / LDAPR (post). ---
        StlrPre32 | StlrPre64 | LdaprPost32 | LdaprPost64 => enc_stlr_ldapr_wb(insn),

        // --- Memory tagging (STG/LDG/...). ---
        _ if tag_is(code) => enc_tags(insn),

        // --- Advanced SIMD load/store structures live in ldst_simd. ---
        _ if ldst_simd::is_ldst_simd(code) => ldst_simd::encode(insn),

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

/// Unpack an immediate memory operand `n`: returns `(rn, imm, mode)`.
#[inline]
fn mem_imm(insn: &Instruction, n: usize) -> Result<(u32, i64, MemIndexMode), EncodeError> {
    match insn.op(n) {
        Operand::MemImm { base, imm, mode } => Ok((base.number() as u32, imm, mode)),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Recover the 5-bit `Rt` field of a prefetch operand (the named keyword or a
/// raw `#imm` fallback), inverting [`crate::decode::ldst::prefetch_op`].
#[inline]
fn prefetch_rt(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    match insn.op(n) {
        Operand::ImmUnsigned(v) => {
            if v > 0x1f {
                return Err(EncodeError::InvalidImmediate);
            }
            Ok(v as u32)
        }
        Operand::SysOp(tok) => prefetch_field_of(tok).ok_or(EncodeError::InvalidOperand),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Recover the 6-bit `rprfop` field of a range-prefetch operand (the named
/// keyword or a raw `#imm` fallback), inverting [`crate::decode::ldst::rprfop_op`].
#[inline]
fn rprfop_field(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    match insn.op(n) {
        Operand::ImmUnsigned(v) => {
            if v > 0x3f {
                return Err(EncodeError::InvalidImmediate);
            }
            Ok(v as u32)
        }
        Operand::SysOp(tok) => {
            // The named subset is `imm6<5:3>==0 && imm6<1>==0`, with bit0 the
            // pld/pst type and bit2 the keep/strm policy.
            let v = match tok.name() {
                "pldkeep" => 0b000000,
                "pstkeep" => 0b000001,
                "pldstrm" => 0b000100,
                "pststrm" => 0b000101,
                _ => return Err(EncodeError::InvalidOperand),
            };
            Ok(v)
        }
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Encode `RPRFM <rprfop>, <Xm>, [<Xn|SP>]` (FEAT_RPRFM). The slot is the PRFM
/// register-offset encoding (`size==11`, `V==0`, `opc==10`, `word<11:10>==10`)
/// with `imm6 = option<2> : option<0> : S : Rt<2:0>` and a forced `option<1>==1`.
fn enc_rprfm(insn: &Instruction) -> R {
    let imm6 = rprfop_field(insn, 0)?;
    let rm = reg_num(insn, 1)?;
    let (rn, off, mode) = mem_imm(insn, 2)?;
    if off != 0 || mode != MemIndexMode::Offset {
        return Err(EncodeError::InvalidOperand);
    }
    // Distribute imm6 across option<2>, option<0>, S, Rt<2:0>. `option<1>` is
    // fixed to 1; Rt<4:3> are fixed to 11 (matching the architectural form).
    let option = (((imm6 >> 5) & 1) << 2) | (1 << 1) | ((imm6 >> 4) & 1);
    let s = (imm6 >> 3) & 1;
    let rt = 0b11000 | (imm6 & 0b111);
    // Fixed bits: size==11 (word<31:30>), V==0 (word<26>), opc==10
    // (word<23:22>), word<21>==1, word<11:10>==10. (word<22> and word<10> are
    // zero and so contribute nothing to the OR.)
    let word = (0b11111000u32 << 24)
        | (1 << 23)
        | (1 << 21)
        | (rm << 16)
        | (option << 13)
        | (s << 12)
        | (1 << 11)
        | (rn << 5)
        | rt;
    Ok(word)
}

/// Map a prefetch keyword token back to its 5-bit `Rt` field
/// (`<type>:<target>:<policy>`), the inverse of the decoder's name table.
fn prefetch_field_of(tok: SysToken) -> Option<u32> {
    // Field = type<4:3> : target<2:1> : policy<0>. type 00 pld / 01 pli / 10 pst;
    // target 00 l1 / 01 l2 / 10 l3; policy 0 keep / 1 strm.
    const TABLE: [(&str, u32); 18] = [
        ("pldl1keep", 0b00000),
        ("pldl1strm", 0b00001),
        ("pldl2keep", 0b00010),
        ("pldl2strm", 0b00011),
        ("pldl3keep", 0b00100),
        ("pldl3strm", 0b00101),
        ("plil1keep", 0b01000),
        ("plil1strm", 0b01001),
        ("plil2keep", 0b01010),
        ("plil2strm", 0b01011),
        ("plil3keep", 0b01100),
        ("plil3strm", 0b01101),
        ("pstl1keep", 0b10000),
        ("pstl1strm", 0b10001),
        ("pstl2keep", 0b10010),
        ("pstl2strm", 0b10011),
        ("pstl3keep", 0b10100),
        ("pstl3strm", 0b10101),
    ];
    let name = tok.name();
    TABLE.iter().find(|(n, _)| *n == name).map(|(_, v)| *v)
}

// ---------------------------------------------------------------------------
// Inverse classification: Code -> (size, v, opc) for the register forms.
// ---------------------------------------------------------------------------

/// The addressing-mode variant of a load/store-register form (mirrors the
/// decoder's `RegVariant`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum RegVariant {
    Unsigned,
    Unscaled,
    ImmPost,
    ImmPre,
    RegOff,
}

/// Decoded register-form fields recovered from a [`Code`].
struct RegForm {
    size: u32,
    v: u32,
    opc: u32,
    /// `true` when the data register is scalar-FP/SIMD.
    is_fp: bool,
    /// For FP forms: the B/H/S/D/Q access code (0..4).
    fp_code: u32,
    /// `true` when the data slot is a prefetch op.
    is_prfm: bool,
}

/// The `(size, v, opc, ...)` for a register-form [`Code`] and its addressing
/// variant, or `None` if `code` is not a register form. `is_fp`/`fp_code`
/// describe the data register; `is_prfm` flags the prefetch carrier.
fn reg_variant(code: Code) -> Option<RegVariant> {
    reg_form(code).map(|(v, _)| v)
}

/// Full inverse of `classify_reg`: map a register-form `Code` back to its
/// `(variant, fields)`.
#[allow(clippy::too_many_lines)]
fn reg_form(code: Code) -> Option<(RegVariant, RegForm)> {
    use Code::*;
    // GP helper.
    let gp = |size: u32, opc: u32| RegForm {
        size,
        v: 0,
        opc,
        is_fp: false,
        fp_code: 0,
        is_prfm: false,
    };
    let prf = |size: u32, opc: u32| RegForm {
        size,
        v: 0,
        opc,
        is_fp: false,
        fp_code: 0,
        is_prfm: true,
    };
    // FP helper: access code `acc` (0..4) -> (size, opc<1>) with opc<0> = load.
    let fp = |acc: u32, load: bool| RegForm {
        size: acc & 0b11,
        v: 1,
        opc: ((acc >> 2) << 1) | (load as u32),
        is_fp: true,
        fp_code: acc,
        is_prfm: false,
    };
    Some(match code {
        // ---- Unsigned immediate offset ----
        StrbImmUnsigned => (RegVariant::Unsigned, gp(0, 0b00)),
        LdrbImmUnsigned => (RegVariant::Unsigned, gp(0, 0b01)),
        LdrsbImmUnsigned64 => (RegVariant::Unsigned, gp(0, 0b10)),
        LdrsbImmUnsigned32 => (RegVariant::Unsigned, gp(0, 0b11)),
        StrhImmUnsigned => (RegVariant::Unsigned, gp(1, 0b00)),
        LdrhImmUnsigned => (RegVariant::Unsigned, gp(1, 0b01)),
        LdrshImmUnsigned64 => (RegVariant::Unsigned, gp(1, 0b10)),
        LdrshImmUnsigned32 => (RegVariant::Unsigned, gp(1, 0b11)),
        StrImmUnsigned32 => (RegVariant::Unsigned, gp(2, 0b00)),
        LdrImmUnsigned32 => (RegVariant::Unsigned, gp(2, 0b01)),
        LdrswImmUnsigned => (RegVariant::Unsigned, gp(2, 0b10)),
        StrImmUnsigned64 => (RegVariant::Unsigned, gp(3, 0b00)),
        LdrImmUnsigned64 => (RegVariant::Unsigned, gp(3, 0b01)),
        PrfmImmUnsigned => (RegVariant::Unsigned, prf(3, 0b10)),
        StrFpImmUnsigned8 => (RegVariant::Unsigned, fp(0, false)),
        LdrFpImmUnsigned8 => (RegVariant::Unsigned, fp(0, true)),
        StrFpImmUnsigned16 => (RegVariant::Unsigned, fp(1, false)),
        LdrFpImmUnsigned16 => (RegVariant::Unsigned, fp(1, true)),
        StrFpImmUnsigned32 => (RegVariant::Unsigned, fp(2, false)),
        LdrFpImmUnsigned32 => (RegVariant::Unsigned, fp(2, true)),
        StrFpImmUnsigned64 => (RegVariant::Unsigned, fp(3, false)),
        LdrFpImmUnsigned64 => (RegVariant::Unsigned, fp(3, true)),
        StrFpImmUnsigned128 => (RegVariant::Unsigned, fp(4, false)),
        LdrFpImmUnsigned128 => (RegVariant::Unsigned, fp(4, true)),

        // ---- Register offset ----
        StrbReg => (RegVariant::RegOff, gp(0, 0b00)),
        LdrbReg => (RegVariant::RegOff, gp(0, 0b01)),
        LdrsbReg64 => (RegVariant::RegOff, gp(0, 0b10)),
        LdrsbReg32 => (RegVariant::RegOff, gp(0, 0b11)),
        StrhReg => (RegVariant::RegOff, gp(1, 0b00)),
        LdrhReg => (RegVariant::RegOff, gp(1, 0b01)),
        LdrshReg64 => (RegVariant::RegOff, gp(1, 0b10)),
        LdrshReg32 => (RegVariant::RegOff, gp(1, 0b11)),
        StrReg32 => (RegVariant::RegOff, gp(2, 0b00)),
        LdrReg32 => (RegVariant::RegOff, gp(2, 0b01)),
        LdrswReg => (RegVariant::RegOff, gp(2, 0b10)),
        StrReg64 => (RegVariant::RegOff, gp(3, 0b00)),
        LdrReg64 => (RegVariant::RegOff, gp(3, 0b01)),
        PrfmReg => (RegVariant::RegOff, prf(3, 0b10)),
        // The B/H register-offset FP forms reuse a 32-bit carrier code; the
        // access code is carried in the operand at encode time, so the fp_code
        // here is only a default. We recover the true access size from the data
        // register's width in `enc_reg_offset`.
        LdrFpReg32 => (RegVariant::RegOff, fp(2, true)),
        LdrFpReg64 => (RegVariant::RegOff, fp(3, true)),
        LdrFpReg128 => (RegVariant::RegOff, fp(4, true)),
        StrFpReg32 => (RegVariant::RegOff, fp(2, false)),
        StrFpReg64 => (RegVariant::RegOff, fp(3, false)),
        StrFpReg128 => (RegVariant::RegOff, fp(4, false)),

        // ---- Unscaled (LDUR/STUR/PRFUM) ----
        Sturb => (RegVariant::Unscaled, gp(0, 0b00)),
        Ldurb => (RegVariant::Unscaled, gp(0, 0b01)),
        Ldursb64 => (RegVariant::Unscaled, gp(0, 0b10)),
        Ldursb32 => (RegVariant::Unscaled, gp(0, 0b11)),
        Sturh => (RegVariant::Unscaled, gp(1, 0b00)),
        Ldurh => (RegVariant::Unscaled, gp(1, 0b01)),
        Ldursh64 => (RegVariant::Unscaled, gp(1, 0b10)),
        Ldursh32 => (RegVariant::Unscaled, gp(1, 0b11)),
        Stur32 => (RegVariant::Unscaled, gp(2, 0b00)),
        Ldur32 => (RegVariant::Unscaled, gp(2, 0b01)),
        Ldursw => (RegVariant::Unscaled, gp(2, 0b10)),
        Stur64 => (RegVariant::Unscaled, gp(3, 0b00)),
        Ldur64 => (RegVariant::Unscaled, gp(3, 0b01)),
        Prfum => (RegVariant::Unscaled, prf(3, 0b10)),

        // ---- Immediate post-index ----
        StrbImmPost => (RegVariant::ImmPost, gp(0, 0b00)),
        LdrbImmPost => (RegVariant::ImmPost, gp(0, 0b01)),
        LdrsbImmPost64 => (RegVariant::ImmPost, gp(0, 0b10)),
        LdrsbImmPost32 => (RegVariant::ImmPost, gp(0, 0b11)),
        StrhImmPost => (RegVariant::ImmPost, gp(1, 0b00)),
        LdrhImmPost => (RegVariant::ImmPost, gp(1, 0b01)),
        LdrshImmPost64 => (RegVariant::ImmPost, gp(1, 0b10)),
        LdrshImmPost32 => (RegVariant::ImmPost, gp(1, 0b11)),
        StrImmPost32 => (RegVariant::ImmPost, gp(2, 0b00)),
        LdrImmPost32 => (RegVariant::ImmPost, gp(2, 0b01)),
        LdrswImmPost => (RegVariant::ImmPost, gp(2, 0b10)),
        StrImmPost64 => (RegVariant::ImmPost, gp(3, 0b00)),
        LdrImmPost64 => (RegVariant::ImmPost, gp(3, 0b01)),

        // ---- Immediate pre-index ----
        StrbImmPre => (RegVariant::ImmPre, gp(0, 0b00)),
        LdrbImmPre => (RegVariant::ImmPre, gp(0, 0b01)),
        LdrsbImmPre64 => (RegVariant::ImmPre, gp(0, 0b10)),
        LdrsbImmPre32 => (RegVariant::ImmPre, gp(0, 0b11)),
        StrhImmPre => (RegVariant::ImmPre, gp(1, 0b00)),
        LdrhImmPre => (RegVariant::ImmPre, gp(1, 0b01)),
        LdrshImmPre64 => (RegVariant::ImmPre, gp(1, 0b10)),
        LdrshImmPre32 => (RegVariant::ImmPre, gp(1, 0b11)),
        StrImmPre32 => (RegVariant::ImmPre, gp(2, 0b00)),
        LdrImmPre32 => (RegVariant::ImmPre, gp(2, 0b01)),
        LdrswImmPre => (RegVariant::ImmPre, gp(2, 0b10)),
        StrImmPre64 => (RegVariant::ImmPre, gp(3, 0b00)),
        LdrImmPre64 => (RegVariant::ImmPre, gp(3, 0b01)),

        // FP unscaled & pre/post reuse the FP unsigned codes; they are
        // distinguished from the unsigned-offset form by mnemonic + operand
        // addressing mode at encode time, handled separately.
        _ => return None,
    })
}

/// The access-size log2 (0..4) for a scalar-FP/SIMD register, used to recover the
/// B/H/S/D/Q access code from the data register of a reg-offset FP form.
#[inline]
fn fp_acc_of(reg: Register) -> Option<u32> {
    Some(match reg.width_bits() {
        8 => 0,
        16 => 1,
        32 => 2,
        64 => 3,
        128 => 4,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Load register (literal).
// ---------------------------------------------------------------------------

/// `LDR (literal)` / `LDRSW` / `PRFM` (literal) and the SIMD&FP literal loads.
/// Encoding: `opc(31:30) 011 V(26) 00 imm19 Rt`.
fn enc_literal(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();
    let (opc, v) = match code {
        LdrLit32 => (0b00u32, 0u32),
        LdrLit64 => (0b01, 0),
        LdrswLit => (0b10, 0),
        PrfmLit => (0b11, 0),
        LdrLitFp32 => (0b00, 1),
        LdrLitFp64 => (0b01, 1),
        _ => (0b10, 1), // LdrLitFp128
    };

    let rt = if code == PrfmLit {
        prefetch_rt(insn, 0)?
    } else {
        reg_num(insn, 0)?
    };
    let target = match insn.op(1) {
        Operand::Label(v) => v,
        _ => return Err(EncodeError::InvalidOperand),
    };
    let imm19 = label_imm19(target, insn.ip())?;

    let word = (opc << 30) | (0b011 << 27) | (v << 26) | (imm19 << 5) | rt;
    Ok(word)
}

/// Recover the 19-bit `imm19` (the `imm19:00` byte offset / 4) from an absolute
/// label and `ip`. The target is `ip + SignExtend(imm19:00, 21)`.
fn label_imm19(target: u64, ip: u64) -> Result<u32, EncodeError> {
    let off = target.wrapping_sub(ip) as i64;
    if off & 0b11 != 0 {
        return Err(EncodeError::InvalidImmediate);
    }
    let imm = off >> 2;
    // 19-bit signed range.
    if (imm >> 18) != 0 && (imm >> 18) != -1 {
        return Err(EncodeError::InvalidImmediate);
    }
    Ok((imm as u32) & 0x7_ffff)
}

// ---------------------------------------------------------------------------
// Load/store register: unsigned immediate offset.
// ---------------------------------------------------------------------------

/// `LDR/STR (immediate, unsigned offset)` and the byte/half/sign/FP variants.
/// Encoding: `size(31:30) 111 V(26) 01 opc(23:22) imm12 Rn Rt`.
fn enc_reg_unsigned(insn: &Instruction) -> R {
    let (_, form) = reg_form(insn.code()).ok_or(EncodeError::Unsupported)?;
    let rt = data_reg_field(insn, &form)?;
    let (rn, imm, mode) = mem_imm(insn, 1)?;
    if mode != MemIndexMode::Offset {
        return Err(EncodeError::InvalidOperand);
    }
    // Unsigned-offset scale = access-size log2. For GP that is `size`; for FP it
    // is the full B/H/S/D/Q access code (`fp_code`), since the 128-bit form has
    // size==0 but a 16-byte access.
    let scale = if form.is_fp { form.fp_code } else { form.size };
    if imm < 0 {
        return Err(EncodeError::InvalidImmediate);
    }
    let imm12 = encode_scaled_imm(imm, scale)?;
    let word = (form.size << 30)
        | (0b111 << 27)
        | (form.v << 26)
        | (0b01 << 24)
        | (form.opc << 22)
        | (imm12 << 10)
        | (rn << 5)
        | rt;
    Ok(word)
}

/// Scale an absolute byte immediate down by `scale`, requiring exact alignment
/// and a 12-bit result.
fn encode_scaled_imm(imm: i64, scale: u32) -> Result<u32, EncodeError> {
    let step = 1i64 << scale;
    if imm % step != 0 {
        return Err(EncodeError::InvalidImmediate);
    }
    let v = imm >> scale;
    if !(0..=0xfff).contains(&v) {
        return Err(EncodeError::InvalidImmediate);
    }
    Ok(v as u32)
}

/// Recover the 5-bit `Rt` data field for a GP / PRFM / FP register form.
fn data_reg_field(insn: &Instruction, form: &RegForm) -> Result<u32, EncodeError> {
    if form.is_prfm {
        prefetch_rt(insn, 0)
    } else {
        reg_num(insn, 0)
    }
}

// ---------------------------------------------------------------------------
// Load/store register: unscaled (LDUR/STUR/PRFUM).
// ---------------------------------------------------------------------------

/// `LDUR/STUR (unscaled)` and friends. Encoding:
/// `size 111 V 00 opc 0 imm9 00 Rn Rt`. `imm9` is signed, unscaled.
fn enc_reg_unscaled(insn: &Instruction) -> R {
    let (_, form) = reg_form(insn.code()).ok_or(EncodeError::Unsupported)?;
    let rt = data_reg_field(insn, &form)?;
    let (rn, imm, mode) = mem_imm(insn, 1)?;
    if mode != MemIndexMode::Offset {
        return Err(EncodeError::InvalidOperand);
    }
    let imm9 = encode_imm9(imm)?;
    let word = (form.size << 30)
        | (0b111 << 27)
        | (form.v << 26)
        | (form.opc << 22)
        | (imm9 << 12)
        | (rn << 5)
        | rt;
    Ok(word)
}

/// Encode a signed 9-bit unscaled immediate.
fn encode_imm9(imm: i64) -> Result<u32, EncodeError> {
    if !(-256..=255).contains(&imm) {
        return Err(EncodeError::InvalidImmediate);
    }
    Ok((imm as u32) & 0x1ff)
}

// ---------------------------------------------------------------------------
// Load/store register: immediate post/pre-index.
// ---------------------------------------------------------------------------

/// Shared pre/post-index immediate encode. Encoding:
/// `size 111 V 00 opc 0 imm9 (01|11) Rn Rt`. `imm9` is signed, unscaled.
fn enc_reg_immidx(insn: &Instruction) -> R {
    let (variant, form) = reg_form(insn.code()).ok_or(EncodeError::Unsupported)?;
    let rt = data_reg_field(insn, &form)?;
    let (rn, imm, mode) = mem_imm(insn, 1)?;
    let want = if variant == RegVariant::ImmPre {
        MemIndexMode::PreIndex
    } else {
        MemIndexMode::PostImm
    };
    if mode != want {
        return Err(EncodeError::InvalidOperand);
    }
    let imm9 = encode_imm9(imm)?;
    let op4 = if variant == RegVariant::ImmPre { 0b11 } else { 0b01 };
    let word = (form.size << 30)
        | (0b111 << 27)
        | (form.v << 26)
        | (form.opc << 22)
        | (imm9 << 12)
        | (op4 << 10)
        | (rn << 5)
        | rt;
    Ok(word)
}

// ---------------------------------------------------------------------------
// FP unscaled & pre/post-index (LDUR/STUR/LDR/STR with B/H/S/D/Q register).
// ---------------------------------------------------------------------------
//
// These reuse the FP *unsigned* codes; the decoder re-tags the mnemonic
// (Ldur/Stur for unscaled, Ldr/Str for pre/post) and the addressing mode lives
// in the memory operand. They are dispatched ahead of the unsigned-offset path
// by inspecting the memory operand's mode.

/// Encode an FP register form whose code is one of the FP *unsigned* codes but
/// whose addressing is actually unscaled (`LDUR`/`STUR`) or pre/post-index. The
/// unsigned-offset form and the unscaled form both carry a `MemImm::Offset`
/// operand, so they are told apart by the **mnemonic** (`Ldur`/`Stur` vs
/// `Ldr`/`Str`). Returns `None` to defer to the unsigned-offset encoder.
fn enc_fp_unscaled_or_idx(insn: &Instruction) -> Option<R> {
    let (_, form) = reg_form(insn.code())?;
    if !form.is_fp {
        return None;
    }
    let (rn, imm, mode) = match insn.op(1) {
        Operand::MemImm { base, imm, mode } => (base.number() as u32, imm, mode),
        _ => return None,
    };
    let is_unscaled = matches!(insn.mnemonic(), Mnemonic::Ldur | Mnemonic::Stur);
    // Determine the op4 selector + whether this is one of the imm9 (unscaled /
    // pre / post) forms. A plain `Ldr`/`Str` with `Offset` mode is the
    // unsigned-offset form — defer.
    let op4 = match mode {
        MemIndexMode::Offset if is_unscaled => 0b00u32, // LDUR/STUR
        MemIndexMode::Offset => return None,            // unsigned-offset path
        MemIndexMode::PostImm => 0b01,
        MemIndexMode::PreIndex => 0b11,
        MemIndexMode::PostReg | MemIndexMode::PreNoOffset => {
            return Some(Err(EncodeError::InvalidOperand))
        }
    };
    let rt = match reg_num(insn, 0) {
        Ok(r) => r,
        Err(e) => return Some(Err(e)),
    };
    let imm9 = match encode_imm9(imm) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
    };
    let word = (form.size << 30)
        | (0b111 << 27)
        | (form.v << 26)
        | (form.opc << 22)
        | (imm9 << 12)
        | (op4 << 10)
        | (rn << 5)
        | rt;
    Some(Ok(word))
}

// ---------------------------------------------------------------------------
// Load/store register: unprivileged (LDTR/STTR).
// ---------------------------------------------------------------------------

/// Inverse of the unprivileged classification: `Code -> (size, opc, gp_x)`.
fn unpriv_fields(code: Code) -> Option<(u32, u32)> {
    use Code::*;
    Some(match code {
        Sttrb => (0, 0b00),
        Ldtrb => (0, 0b01),
        Ldtrsb64 => (0, 0b10),
        Ldtrsb32 => (0, 0b11),
        Sttrh => (1, 0b00),
        Ldtrh => (1, 0b01),
        Ldtrsh64 => (1, 0b10),
        Ldtrsh32 => (1, 0b11),
        Sttr32 => (2, 0b00),
        Ldtr32 => (2, 0b01),
        Ldtrsw => (2, 0b10),
        Sttr64 => (3, 0b00),
        Ldtr64 => (3, 0b01),
        _ => return None,
    })
}

/// `LDTR/STTR` (unprivileged). Encoding: `size 111 0 00 opc 0 imm9 10 Rn Rt`.
fn enc_reg_unpriv(insn: &Instruction) -> R {
    let (size, opc) = unpriv_fields(insn.code()).ok_or(EncodeError::Unsupported)?;
    let rt = reg_num(insn, 0)?;
    let (rn, imm, mode) = mem_imm(insn, 1)?;
    if mode != MemIndexMode::Offset {
        return Err(EncodeError::InvalidOperand);
    }
    let imm9 = encode_imm9(imm)?;
    let word = (size << 30)
        | (0b111 << 27)
        | (opc << 22)
        | (imm9 << 12)
        | (0b10 << 10)
        | (rn << 5)
        | rt;
    Ok(word)
}

// ---------------------------------------------------------------------------
// Load/store register: register offset.
// ---------------------------------------------------------------------------

/// `LDR/STR (register)` and friends. Encoding:
/// `size 111 V 00 opc 1 Rm option(15:13) S(12) 10 Rn Rt`.
fn enc_reg_offset(insn: &Instruction) -> R {
    let (_, mut form) = reg_form(insn.code()).ok_or(EncodeError::Unsupported)?;
    let rt = data_reg_field(insn, &form)?;

    // For FP forms, recover the true access size from the data register width
    // (the B/H reg-offset forms reuse the 32-bit carrier code).
    if form.is_fp {
        let acc = fp_acc_of(insn.op_register(0)).ok_or(EncodeError::InvalidOperand)?;
        form.fp_code = acc;
        form.size = acc & 0b11;
        form.opc = ((acc >> 2) << 1) | (form.opc & 1);
    }

    // The shift amount (when S==1) equals the access-size log2: `size` for GP,
    // the full B/H/S/D/Q code for FP (Q has size==0 but a 16-byte access).
    let scale = if form.is_fp { form.fp_code } else { form.size };
    let (rn, rm, option, s) = recover_mem_ext(insn, 1, scale)?;
    let word = (form.size << 30)
        | (0b111 << 27)
        | (form.v << 26)
        | (form.opc << 22)
        | (1 << 21)
        | (rm << 16)
        | (option << 13)
        | (s << 12)
        | (0b10 << 10)
        | (rn << 5)
        | rt;
    Ok(word)
}

/// Unpack a [`Operand::MemExt`] into `(rn, rm, option, s)`. The shift byte packs
/// `S` in bit7 and the amount in bits<6:0> (see the decoder's `pack_shift`); `S`
/// is what the encoder needs (the amount is recomputed from the scale by the
/// architecture, so it is only validated here against `scale`).
fn recover_mem_ext(
    insn: &Instruction,
    n: usize,
    scale: u32,
) -> Result<(u32, u32, u32, u32), EncodeError> {
    match insn.op(n) {
        Operand::MemExt {
            base,
            index,
            extend,
            shift,
        } => {
            let rn = base.number() as u32;
            let rm = index.number() as u32;
            let option = extend.as_bits() as u32;
            // option<1> must be set (uxtw/uxtx/sxtw/sxtx only).
            if option & 0b010 == 0 {
                return Err(EncodeError::InvalidOperand);
            }
            // bit7 of `shift` is the "S" flag; bits<6:0> the amount.
            let s = (shift >> 7) & 1;
            let amt = (shift & 0x7f) as u32;
            // When S==1 the amount must equal the access-size scale; when S==0 it
            // must be zero (the decoder emits amt = S ? scale : 0).
            let expected = if s == 1 { scale } else { 0 };
            if amt != expected {
                return Err(EncodeError::InvalidImmediate);
            }
            Ok((rn, rm, option, s as u32))
        }
        _ => Err(EncodeError::InvalidOperand),
    }
}

// ---------------------------------------------------------------------------
// Load/store register pair (and SIMD&FP pair, LDNP/STNP, STGP).
// ---------------------------------------------------------------------------

/// The pair family + addressing index recovered from a [`Code`]. Returns
/// `(opc, v, l, idx, scale, is_stgp)` where `idx` is the 2-bit selector
/// (00 NP, 01 post, 10 offset, 11 pre).
struct PairForm {
    opc: u32,
    v: u32,
    l: u32,
    idx: u32,
    scale: u32,
    /// `true` when the data registers are scalar-FP (Q/D/S by `fp_code`).
    is_fp: bool,
    fp_code: u32,
}

/// Inverse mapping for the pair family.
#[allow(clippy::too_many_lines)]
fn pair_kind(code: Code) -> Option<PairForm> {
    use Code::*;
    let gp = |opc: u32, l: u32, idx: u32, scale: u32| PairForm {
        opc,
        v: 0,
        l,
        idx,
        scale,
        is_fp: false,
        fp_code: 0,
    };
    let fp = |opc: u32, l: u32, idx: u32, scale: u32, fp_code: u32| PairForm {
        opc,
        v: 1,
        l,
        idx,
        scale,
        is_fp: true,
        fp_code,
    };
    Some(match code {
        // GP W (opc=00, scale=2).
        Stnp32 => gp(0b00, 0, 0b00, 2),
        Ldnp32 => gp(0b00, 1, 0b00, 2),
        Stp32Post => gp(0b00, 0, 0b01, 2),
        Ldp32Post => gp(0b00, 1, 0b01, 2),
        Stp32 => gp(0b00, 0, 0b10, 2),
        Ldp32 => gp(0b00, 1, 0b10, 2),
        Stp32Pre => gp(0b00, 0, 0b11, 2),
        Ldp32Pre => gp(0b00, 1, 0b11, 2),
        // GP X (opc=10, scale=3).
        Stnp64 => gp(0b10, 0, 0b00, 3),
        Ldnp64 => gp(0b10, 1, 0b00, 3),
        Stp64Post => gp(0b10, 0, 0b01, 3),
        Ldp64Post => gp(0b10, 1, 0b01, 3),
        Stp64 => gp(0b10, 0, 0b10, 3),
        Ldp64 => gp(0b10, 1, 0b10, 3),
        Stp64Pre => gp(0b10, 0, 0b11, 3),
        LdpPre64 => gp(0b10, 1, 0b11, 3),
        // LDPSW (opc=01, load-only, scale=2).
        LdpswPost => gp(0b01, 1, 0b01, 2),
        Ldpsw => gp(0b01, 1, 0b10, 2),
        LdpswPre => gp(0b01, 1, 0b11, 2),
        // STGP (opc=01, store, scale=4 [×16], MTE). The decoder routes this to a
        // dedicated emit, but the raw encoding is the generic pair format
        // (opc=01, V=0, L=0), so it packs through the same path.
        StgpPost => gp(0b01, 0, 0b01, 4),
        StgpOff => gp(0b01, 0, 0b10, 4),
        StgpPre => gp(0b01, 0, 0b11, 4),
        // SIMD&FP pair: opc 00->S(2), 01->D(3), 10->Q(4).
        StpFp32 => fp(0b00, 0, 0b10, 2, 2),
        LdpFp32 => fp(0b00, 1, 0b10, 2, 2),
        StpFp64 => fp(0b01, 0, 0b10, 3, 3),
        LdpFp64 => fp(0b01, 1, 0b10, 3, 3),
        StpFp128 => fp(0b10, 0, 0b10, 4, 4),
        LdpFp128 => fp(0b10, 1, 0b10, 4, 4),
        // FEAT_THE unprivileged translation-enhanced pair (opc=11, V=0, X regs,
        // scale=3). idx 00=NP, 01=post, 10=offset, 11=pre.
        Ldtnp => gp(0b11, 1, 0b00, 3),
        Sttnp => gp(0b11, 0, 0b00, 3),
        LdtpPost => gp(0b11, 1, 0b01, 3),
        SttpPost => gp(0b11, 0, 0b01, 3),
        LdtpOff => gp(0b11, 1, 0b10, 3),
        SttpOff => gp(0b11, 0, 0b10, 3),
        LdtpPre => gp(0b11, 1, 0b11, 3),
        SttpPre => gp(0b11, 0, 0b11, 3),
        // FEAT_LSUI quadword unprivileged translation-enhanced pair (opc=11,
        // V=1, Q regs, scale=4). idx 00=NP, 01=post, 10=offset, 11=pre.
        Ldtnpq => fp(0b11, 1, 0b00, 4, 4),
        Sttnpq => fp(0b11, 0, 0b00, 4, 4),
        LdtpqPost => fp(0b11, 1, 0b01, 4, 4),
        SttpqPost => fp(0b11, 0, 0b01, 4, 4),
        LdtpqOff => fp(0b11, 1, 0b10, 4, 4),
        SttpqOff => fp(0b11, 0, 0b10, 4, 4),
        LdtpqPre => fp(0b11, 1, 0b11, 4, 4),
        SttpqPre => fp(0b11, 0, 0b11, 4, 4),
        _ => return None,
    })
}

/// Load/store pair. Encoding:
/// `opc(31:30) 101 V(26) idx(24:23) L(22) imm7 Rt2 Rn Rt`.
fn enc_pair(insn: &Instruction) -> R {
    let mut form = pair_kind(insn.code()).ok_or(EncodeError::Unsupported)?;

    // The NP forms (LDNP/STNP) carry idx=00 but reuse the offset code; recover
    // the real addressing index from the memory operand for every pair form.
    let rt = reg_num(insn, 0)?;
    let rt2 = reg_num(insn, 1)?;
    let (rn, imm, mode) = mem_imm(insn, 2)?;

    // Recover the 2-bit addressing index (00 NP, 01 post, 10 offset, 11 pre).
    //
    // GP pairs use a DISTINCT `Code` per index, so `form.idx` is authoritative
    // and the operand mode must agree. FP pairs (and the FP NP forms) reuse a
    // single `LdpFp*`/`StpFp*` `Code` for every index — the decoder selects NP
    // via the `Ldnp`/`Stnp` mnemonic and the rest via the operand mode — so for
    // them the index is taken from the mnemonic + operand mode directly.
    let is_np_mnem = matches!(
        insn.mnemonic(),
        Mnemonic::Ldnp | Mnemonic::Stnp | Mnemonic::Ldtnp | Mnemonic::Sttnp
    );
    let mode_idx = if is_np_mnem {
        if mode != MemIndexMode::Offset {
            return Err(EncodeError::InvalidOperand);
        }
        0b00u32
    } else {
        match mode {
            MemIndexMode::PostImm => 0b01,
            MemIndexMode::PreIndex => 0b11,
            MemIndexMode::Offset => 0b10,
            MemIndexMode::PostReg | MemIndexMode::PreNoOffset => {
                return Err(EncodeError::InvalidOperand)
            }
        }
    };

    if form.is_fp {
        // Validate the data registers match the expected FP width.
        let acc = fp_acc_of(insn.op_register(0)).ok_or(EncodeError::InvalidOperand)?;
        if acc != form.fp_code {
            return Err(EncodeError::InvalidOperand);
        }
        form.idx = mode_idx;
    } else {
        // GP: the code pinned the index; the operand mode must be consistent.
        if mode_idx != form.idx {
            return Err(EncodeError::InvalidOperand);
        }
    }

    let imm7 = encode_scaled_imm7(imm, form.scale)?;
    let word = (form.opc << 30)
        | (0b101 << 27)
        | (form.v << 26)
        | (form.idx << 23)
        | (form.l << 22)
        | (imm7 << 15)
        | (rt2 << 10)
        | (rn << 5)
        | rt;
    Ok(word)
}

/// Scale + sign-encode a 7-bit pair immediate.
fn encode_scaled_imm7(imm: i64, scale: u32) -> Result<u32, EncodeError> {
    let step = 1i64 << scale;
    if imm % step != 0 {
        return Err(EncodeError::InvalidImmediate);
    }
    let v = imm >> scale;
    if !(-64..=63).contains(&v) {
        return Err(EncodeError::InvalidImmediate);
    }
    Ok((v as u32) & 0x7f)
}

// ---------------------------------------------------------------------------
// Load/store exclusive and ordered.
// ---------------------------------------------------------------------------

/// Single-register exclusive: `Code -> (sz, l, o0)`.
fn excl_single_fields(code: Code) -> Option<(u32, u32, u32)> {
    use Code::*;
    Some(match code {
        Ldxrb => (0, 1, 0),
        Ldaxrb => (0, 1, 1),
        Stxrb => (0, 0, 0),
        Stlxrb => (0, 0, 1),
        Ldxrh => (1, 1, 0),
        Ldaxrh => (1, 1, 1),
        Stxrh => (1, 0, 0),
        Stlxrh => (1, 0, 1),
        Ldxr32 => (2, 1, 0),
        Ldaxr32 => (2, 1, 1),
        Stxr32 => (2, 0, 0),
        Stlxr32 => (2, 0, 1),
        Ldxr64 => (3, 1, 0),
        Ldaxr64 => (3, 1, 1),
        Stxr64 => (3, 0, 0),
        Stlxr64 => (3, 0, 1),
        _ => return None,
    })
}

/// `sz 001000 o2=0 L o1=0 Rs o0 Rt2(11111) Rn Rt`.
fn enc_excl_single(insn: &Instruction) -> R {
    let (sz, l, o0) = excl_single_fields(insn.code()).ok_or(EncodeError::Unsupported)?;
    let load = l == 1;
    let (rs, rt, rn) = if load {
        // LDXR* Rt, [Xn]. Rs field is 0b11111.
        let rt = reg_num(insn, 0)?;
        let (rn, _, _) = mem_imm(insn, 1)?;
        (0b11111u32, rt, rn)
    } else {
        // STXR* Ws, Rt, [Xn].
        let rs = reg_num(insn, 0)?;
        let rt = reg_num(insn, 1)?;
        let (rn, _, _) = mem_imm(insn, 2)?;
        (rs, rt, rn)
    };
    let word = excl_word(sz, 0, l, 0, rs, o0, 0b11111, rn, rt);
    Ok(word)
}

/// Pair exclusive: `Code -> (sz, l, o0)`.
fn excl_pair_fields(code: Code) -> Option<(u32, u32, u32)> {
    use Code::*;
    Some(match code {
        Ldxp32 => (2, 1, 0),
        Ldxp64 => (3, 1, 0),
        Ldaxp32 => (2, 1, 1),
        Ldaxp64 => (3, 1, 1),
        Stxp32 => (2, 0, 0),
        Stxp64 => (3, 0, 0),
        Stlxp32 => (2, 0, 1),
        Stlxp64 => (3, 0, 1),
        _ => return None,
    })
}

/// `sz 001000 o2=0 L o1=1 Rs o0 Rt2 Rn Rt`.
fn enc_excl_pair(insn: &Instruction) -> R {
    let (sz, l, o0) = excl_pair_fields(insn.code()).ok_or(EncodeError::Unsupported)?;
    let load = l == 1;
    let (rs, rt, rt2, rn) = if load {
        // LDXP Rt, Rt2, [Xn]. Rs == 0b11111.
        let rt = reg_num(insn, 0)?;
        let rt2 = reg_num(insn, 1)?;
        let (rn, _, _) = mem_imm(insn, 2)?;
        (0b11111u32, rt, rt2, rn)
    } else {
        // STXP Ws, Rt, Rt2, [Xn].
        let rs = reg_num(insn, 0)?;
        let rt = reg_num(insn, 1)?;
        let rt2 = reg_num(insn, 2)?;
        let (rn, _, _) = mem_imm(insn, 3)?;
        (rs, rt, rt2, rn)
    };
    let word = excl_word(sz, 0, l, 1, rs, o0, rt2, rn, rt);
    Ok(word)
}

/// Ordered (LDAR/STLR/LDLAR/STLLR): `Code -> (sz, l, o0)`.
fn ordered_fields(code: Code) -> Option<(u32, u32, u32)> {
    use Code::*;
    Some(match code {
        Ldarb => (0, 1, 1),
        Stlrb => (0, 0, 1),
        Ldlarb => (0, 1, 0),
        Stllrb => (0, 0, 0),
        Ldarh => (1, 1, 1),
        Stlrh => (1, 0, 1),
        Ldlarh => (1, 1, 0),
        Stllrh => (1, 0, 0),
        Ldar32 => (2, 1, 1),
        Stlr32 => (2, 0, 1),
        Ldlar32 => (2, 1, 0),
        Stllr32 => (2, 0, 0),
        Ldar64 => (3, 1, 1),
        Stlr64 => (3, 0, 1),
        Ldlar64 => (3, 1, 0),
        Stllr64 => (3, 0, 0),
        _ => return None,
    })
}

/// `sz 001000 o2=1 L o1=0 Rs(11111) o0 Rt2(11111) Rn Rt`.
fn enc_ordered(insn: &Instruction) -> R {
    let (sz, l, o0) = ordered_fields(insn.code()).ok_or(EncodeError::Unsupported)?;
    let rt = reg_num(insn, 0)?;
    let (rn, _, _) = mem_imm(insn, 1)?;
    let word = excl_word(sz, 1, l, 0, 0b11111, o0, 0b11111, rn, rt);
    Ok(word)
}

/// Assemble an exclusive/ordered/CAS-class word from its raw fields:
/// `sz 001000 o2 L o1 Rs o0 Rt2 Rn Rt`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn excl_word(
    sz: u32,
    o2: u32,
    l: u32,
    o1: u32,
    rs: u32,
    o0: u32,
    rt2: u32,
    rn: u32,
    rt: u32,
) -> u32 {
    (sz << 30)
        | (0b001000 << 24)
        | (o2 << 23)
        | (l << 22)
        | (o1 << 21)
        | (rs << 16)
        | (o0 << 15)
        | (rt2 << 10)
        | (rn << 5)
        | rt
}

// ---------------------------------------------------------------------------
// CAS / CASP (LSE).
// ---------------------------------------------------------------------------

/// CAS: `Code -> (sz, l, o0)`.
fn cas_fields(code: Code) -> Option<(u32, u32, u32)> {
    use Code::*;
    Some(match code {
        Casb => (0, 0, 0),
        Casab => (0, 1, 0),
        Caslb => (0, 0, 1),
        Casalb => (0, 1, 1),
        Cash => (1, 0, 0),
        Casah => (1, 1, 0),
        Caslh => (1, 0, 1),
        Casalh => (1, 1, 1),
        Cas32 => (2, 0, 0),
        Casa32 => (2, 1, 0),
        Casl32 => (2, 0, 1),
        Casal32 => (2, 1, 1),
        Cas64 => (3, 0, 0),
        Casa64 => (3, 1, 0),
        Casl64 => (3, 0, 1),
        Casal64 => (3, 1, 1),
        _ => return None,
    })
}

/// `sz 001000 o2=1 L o1=1 Rs o0 Rt2(11111) Rn Rt`.
fn enc_cas(insn: &Instruction) -> R {
    let (sz, l, o0) = cas_fields(insn.code()).ok_or(EncodeError::Unsupported)?;
    let rs = reg_num(insn, 0)?;
    let rt = reg_num(insn, 1)?;
    let (rn, _, _) = mem_imm(insn, 2)?;
    let word = excl_word(sz, 1, l, 1, rs, o0, 0b11111, rn, rt);
    Ok(word)
}

/// CASP: `Code -> (x, l, o0)` where `x` selects 64-bit (sz field high bit = 0).
fn casp_fields(code: Code) -> Option<(u32, u32, u32)> {
    use Code::*;
    Some(match code {
        Casp32 => (0, 0, 0),
        Caspa32 => (0, 1, 0),
        Caspl32 => (0, 0, 1),
        Caspal32 => (0, 1, 1),
        Casp64 => (1, 0, 0),
        Caspa64 => (1, 1, 0),
        Caspl64 => (1, 0, 1),
        Caspal64 => (1, 1, 1),
        _ => return None,
    })
}

/// CASP `sz(=0:x) 001000 o2=0 L o1=1 Rs o0 Rt2(11111) Rn Rt`. The `sz` field is
/// `0:x` (top bit 0, low bit selects 32/64).
fn enc_casp(insn: &Instruction) -> R {
    let (x, l, o0) = casp_fields(insn.code()).ok_or(EncodeError::Unsupported)?;
    // Pair: Rs, Rs+1, Rt, Rt+1, [Xn]. The even-register pairs are implied; we
    // emit Rs and Rt (the +1 registers must be the consecutive ones).
    let rs = reg_num(insn, 0)?;
    let rs1 = reg_num(insn, 1)?;
    let rt = reg_num(insn, 2)?;
    let rt1 = reg_num(insn, 3)?;
    let (rn, _, _) = mem_imm(insn, 4)?;
    // Validate the pair consecutiveness (Rs even, Rs+1; Rt even, Rt+1). The base
    // registers must be even — odd Rs/Rt is UNDEFINED (ARM ARM).
    if (rs & 1) != 0 || (rt & 1) != 0 {
        return Err(EncodeError::InvalidOperand);
    }
    if rs1 != (rs.wrapping_add(1) & 0x1f) || rt1 != (rt.wrapping_add(1) & 0x1f) {
        return Err(EncodeError::InvalidOperand);
    }
    let sz = x; // sz field = 0:x -> value 0 or 1.
    let word = excl_word(sz, 0, l, 1, rs, o0, 0b11111, rn, rt);
    Ok(word)
}

// ---------------------------------------------------------------------------
// FEAT_LSUI unprivileged atomics.
// ---------------------------------------------------------------------------

/// LSUI single-register exclusive: `Code -> (sz, l, o0)`.
fn lsui_excl_single_fields(code: Code) -> Option<(u32, u32, u32)> {
    use Code::*;
    Some(match code {
        Ldtxr32 => (2, 1, 0),
        Ldatxr32 => (2, 1, 1),
        Sttxr32 => (2, 0, 0),
        Stltxr32 => (2, 0, 1),
        Ldtxr64 => (3, 1, 0),
        Ldatxr64 => (3, 1, 1),
        Sttxr64 => (3, 0, 0),
        Stltxr64 => (3, 0, 1),
        _ => return None,
    })
}

/// `sz 001001 o2=0 L o1=0 Rs o0 Rt2(11111) Rn Rt` — the exclusive layout with the
/// LSUI group bit (`word<24>`) set.
fn enc_lsui_excl_single(insn: &Instruction) -> R {
    let (sz, l, o0) = lsui_excl_single_fields(insn.code()).ok_or(EncodeError::Unsupported)?;
    let load = l == 1;
    let (rs, rt, rn) = if load {
        // LDTXR* Rt, [Xn]. Rs field is 0b11111.
        let rt = reg_num(insn, 0)?;
        let (rn, _, _) = mem_imm(insn, 1)?;
        (0b11111u32, rt, rn)
    } else {
        // STTXR* Ws, Rt, [Xn].
        let rs = reg_num(insn, 0)?;
        let rt = reg_num(insn, 1)?;
        let (rn, _, _) = mem_imm(insn, 2)?;
        (rs, rt, rn)
    };
    let word = excl_word(sz, 0, l, 0, rs, o0, 0b11111, rn, rt) | (1u32 << 24);
    Ok(word)
}

/// LSUI single compare-and-swap (64-bit only): `Code -> (l, o0)`.
fn lsui_cas_fields(code: Code) -> Option<(u32, u32)> {
    use Code::*;
    Some(match code {
        Cast64 => (0, 0),
        Casat64 => (1, 0),
        Caslt64 => (0, 1),
        Casalt64 => (1, 1),
        _ => return None,
    })
}

/// `sz=11 001001 o2=1 L o1=0 Rs o0 Rt2(11111) Rn Rt`.
fn enc_lsui_cas(insn: &Instruction) -> R {
    let (l, o0) = lsui_cas_fields(insn.code()).ok_or(EncodeError::Unsupported)?;
    let rs = reg_num(insn, 0)?;
    let rt = reg_num(insn, 1)?;
    let (rn, _, _) = mem_imm(insn, 2)?;
    let word = excl_word(3, 1, l, 0, rs, o0, 0b11111, rn, rt) | (1u32 << 24);
    Ok(word)
}

/// LSUI compare-and-swap pair (64-bit only): `Code -> (l, o0)`.
fn lsui_casp_fields(code: Code) -> Option<(u32, u32)> {
    use Code::*;
    Some(match code {
        Caspt64 => (0, 0),
        Caspat64 => (1, 0),
        Casplt64 => (0, 1),
        Caspalt64 => (1, 1),
        _ => return None,
    })
}

/// `sz=01 001001 o2=1 L o1=0 Rs o0 Rt2(11111) Rn Rt`. The `sz` field is `0:1`
/// (64-bit pair). `Rs`/`Rt` must be even, with the consecutive odd register.
fn enc_lsui_casp(insn: &Instruction) -> R {
    let (l, o0) = lsui_casp_fields(insn.code()).ok_or(EncodeError::Unsupported)?;
    let rs = reg_num(insn, 0)?;
    let rs1 = reg_num(insn, 1)?;
    let rt = reg_num(insn, 2)?;
    let rt1 = reg_num(insn, 3)?;
    let (rn, _, _) = mem_imm(insn, 4)?;
    if (rs & 1) != 0 || (rt & 1) != 0 {
        return Err(EncodeError::InvalidOperand);
    }
    if rs1 != (rs.wrapping_add(1) & 0x1f) || rt1 != (rt.wrapping_add(1) & 0x1f) {
        return Err(EncodeError::InvalidOperand);
    }
    let word = excl_word(1, 1, l, 0, rs, o0, 0b11111, rn, rt) | (1u32 << 24);
    Ok(word)
}

// ---------------------------------------------------------------------------
// LSE atomics: SWP and LD<op>/ST<op>.
// ---------------------------------------------------------------------------

/// SWP: `Code -> (size, a, r)`.
fn swp_fields(code: Code) -> Option<(u32, u32, u32)> {
    use Code::*;
    Some(match code {
        Swpb => (0, 0, 0),
        Swplb => (0, 0, 1),
        Swpab => (0, 1, 0),
        Swpalb => (0, 1, 1),
        Swph => (1, 0, 0),
        Swplh => (1, 0, 1),
        Swpah => (1, 1, 0),
        Swpalh => (1, 1, 1),
        Swp32 => (2, 0, 0),
        Swpl32 => (2, 0, 1),
        Swpa32 => (2, 1, 0),
        Swpal32 => (2, 1, 1),
        Swp64 => (3, 0, 0),
        Swpl64 => (3, 0, 1),
        Swpa64 => (3, 1, 0),
        Swpal64 => (3, 1, 1),
        _ => return None,
    })
}

/// `SWP{A}{L}{B|H}`: `size 111 0 00 A R 1 Rs o3=1 opc=000 00 Rn Rt`.
fn enc_swp(insn: &Instruction) -> R {
    let (size, a, r) = swp_fields(insn.code()).ok_or(EncodeError::Unsupported)?;
    let rs = reg_num(insn, 0)?;
    let rt = reg_num(insn, 1)?;
    let (rn, _, _) = mem_imm(insn, 2)?;
    let word = atomic_word(size, a, r, rs, 1, 0b000, rn, rt);
    Ok(word)
}

/// FEAT_THE single-register RCW RMW ops (`RCWCLR`/`RCWSWP`/`RCWSET` and the
/// `RCWS*` variants) live in the LSE atomic major (`size 111 0 00 A R 1 Rs o3=1
/// opc 00 Rn Rt`). Recover `(size, a, r, opc)`; `size` is 0 for `RCW*`, 1 for
/// `RCWS*`. The operand shape is always `Rs, Rt, [Xn|SP]` (64-bit).
fn rcw_single_fields(code: Code) -> Option<(u32, u32, u32, u32)> {
    use Code::*;
    // ordering (a, r): plain/L/A/AL.
    macro_rules! q {
        ($size:expr, $opc:expr, $b:ident, $l:ident, $a:ident, $al:ident) => {
            match code {
                $b => return Some(($size, 0, 0, $opc)),
                $l => return Some(($size, 0, 1, $opc)),
                $a => return Some(($size, 1, 0, $opc)),
                $al => return Some(($size, 1, 1, $opc)),
                _ => {}
            }
        };
    }
    q!(0, 0b001, Rcwclr, Rcwclrl, Rcwclra, Rcwclral);
    q!(0, 0b010, Rcwswp, Rcwswpl, Rcwswpa, Rcwswpal);
    q!(0, 0b011, Rcwset, Rcwsetl, Rcwseta, Rcwsetal);
    q!(1, 0b001, Rcwsclr, Rcwsclrl, Rcwsclra, Rcwsclral);
    q!(1, 0b010, Rcwsswp, Rcwsswpl, Rcwsswpa, Rcwsswpal);
    q!(1, 0b011, Rcwsset, Rcwssetl, Rcwsseta, Rcwssetal);
    None
}

/// Encode a single-register RCW RMW: `size 111 0 00 A R 1 Rs o3=1 opc 00 Rn Rt`.
fn enc_rcw_single(insn: &Instruction) -> R {
    let (size, a, r, opc) = rcw_single_fields(insn.code()).ok_or(EncodeError::Unsupported)?;
    let rs = reg_num(insn, 0)?;
    let rt = reg_num(insn, 1)?;
    let (rn, _, _) = mem_imm(insn, 2)?;
    Ok(atomic_word(size, a, r, rs, 1, opc, rn, rt))
}

// ---------------------------------------------------------------------------
// FEAT_THE / FEAT_LSE128 atomics (LDTADD/SWPT, RCW*, RCWS*, LDCLRP/LDSETP/SWPP).
// ---------------------------------------------------------------------------

/// The operand shape of a FEAT_THE / FEAT_LSE128 atomic.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TheShape {
    /// `Rs, Rt, [Xn|SP]` (LDT<op>/SWPT in 32-bit `W` view, or RCWCAS in 64-bit).
    SingleW,
    /// `Rs, Rt, [Xn|SP]`, 64-bit `X` view (LDT<op>/SWPT 64-bit, RCWCAS).
    SingleX,
    /// `Rt, Rs, [Xn|SP]`, 64-bit pair op (LDCLRP/LDSETP/SWPP, RCW*P/RCWS*P).
    Pair,
    /// `Rs, Rs+1, Rt, Rt+1, [Xn|SP]` (RCWCASP / RCWSCASP), even-register pairs.
    CasPair,
}

/// Recover the encoding fields `(sz, a, r, o3, opc, op2, shape)` of a FEAT_THE /
/// FEAT_LSE128 atomic [`Code`]. The word is assembled by [`the_atomic_word`].
fn the_atomic_fields(code: Code) -> Option<(u32, u32, u32, u32, u32, u32, TheShape)> {
    use Code::*;
    use TheShape::*;
    // Ordering tuples (a, r): plain/L/A/AL.
    macro_rules! ord {
        ($base:ident, $l:ident, $a:ident, $al:ident, $body:expr) => {
            match code {
                $base => Some((0u32, 0u32, $body)),
                $l => Some((0, 1, $body)),
                $a => Some((1, 0, $body)),
                $al => Some((1, 1, $body)),
                _ => None,
            }
        };
    }
    // LDTADD/LDTCLR/LDTSET and SWPT: op2=01, single Rs,Rt; sz selects W/X.
    // Each width has its own ordering quartet of codes.
    macro_rules! ldt {
        ($b32:ident,$l32:ident,$a32:ident,$al32:ident,$b64:ident,$l64:ident,$a64:ident,$al64:ident,$o3:expr,$opc:expr) => {
            if let Some((a, r, _)) = ord!($b32, $l32, $a32, $al32, ()) {
                return Some((0, a, r, $o3, $opc, 0b01, SingleW));
            }
            if let Some((a, r, _)) = ord!($b64, $l64, $a64, $al64, ()) {
                return Some((1, a, r, $o3, $opc, 0b01, SingleX));
            }
        };
    }
    ldt!(Ldtadd32, Ldtaddl32, Ldtadda32, Ldtaddal32, Ldtadd64, Ldtaddl64, Ldtadda64, Ldtaddal64, 0, 0b000);
    ldt!(Ldtclr32, Ldtclrl32, Ldtclra32, Ldtclral32, Ldtclr64, Ldtclrl64, Ldtclra64, Ldtclral64, 0, 0b001);
    ldt!(Ldtset32, Ldtsetl32, Ldtseta32, Ldtsetal32, Ldtset64, Ldtsetl64, Ldtseta64, Ldtsetal64, 0, 0b011);
    ldt!(Swpt32, Swptl32, Swpta32, Swptal32, Swpt64, Swptl64, Swpta64, Swptal64, 1, 0b000);

    // RCWCAS (sz 0) / RCWSCAS (sz 1): op2=10, o3=0, opc=0, single Rs,Rt (X).
    if let Some((a, r, _)) = ord!(Rcwcas, Rcwcasl, Rcwcasa, Rcwcasal, ()) {
        return Some((0, a, r, 0, 0b000, 0b10, SingleX));
    }
    if let Some((a, r, _)) = ord!(Rcwscas, Rcwscasl, Rcwscasa, Rcwscasal, ()) {
        return Some((1, a, r, 0, 0b000, 0b10, SingleX));
    }
    // RCWCASP (sz 0) / RCWSCASP (sz 1): op2=11, o3=0, opc=0, even-register pairs.
    if let Some((a, r, _)) = ord!(Rcwcasp, Rcwcaspl, Rcwcaspa, Rcwcaspal, ()) {
        return Some((0, a, r, 0, 0b000, 0b11, CasPair));
    }
    if let Some((a, r, _)) = ord!(Rcwscasp, Rcwscaspl, Rcwscaspa, Rcwscaspal, ()) {
        return Some((1, a, r, 0, 0b000, 0b11, CasPair));
    }
    // op2=00 pair load-op ops (Rt, Rs, [Xn]).
    //   LSE128: LDCLRP (o3=0 opc=1), LDSETP (o3=0 opc=3), SWPP (o3=1 opc=0); sz 0.
    //   THE RCW: RCWCLRP/RCWSWPP/RCWSETP (sz 0), RCWSCLRP/RCWSSWPP/RCWSSETP (sz 1).
    if let Some((a, r, _)) = ord!(Ldclrp, Ldclrpl, Ldclrpa, Ldclrpal, ()) {
        return Some((0, a, r, 0, 0b001, 0b00, Pair));
    }
    if let Some((a, r, _)) = ord!(Ldsetp, Ldsetpl, Ldsetpa, Ldsetpal, ()) {
        return Some((0, a, r, 0, 0b011, 0b00, Pair));
    }
    if let Some((a, r, _)) = ord!(Swpp, Swppl, Swppa, Swppal, ()) {
        return Some((0, a, r, 1, 0b000, 0b00, Pair));
    }
    if let Some((a, r, _)) = ord!(Rcwclrp, Rcwclrpl, Rcwclrpa, Rcwclrpal, ()) {
        return Some((0, a, r, 1, 0b001, 0b00, Pair));
    }
    if let Some((a, r, _)) = ord!(Rcwswpp, Rcwswppl, Rcwswppa, Rcwswppal, ()) {
        return Some((0, a, r, 1, 0b010, 0b00, Pair));
    }
    if let Some((a, r, _)) = ord!(Rcwsetp, Rcwsetpl, Rcwsetpa, Rcwsetpal, ()) {
        return Some((0, a, r, 1, 0b011, 0b00, Pair));
    }
    if let Some((a, r, _)) = ord!(Rcwsclrp, Rcwsclrpl, Rcwsclrpa, Rcwsclrpal, ()) {
        return Some((1, a, r, 1, 0b001, 0b00, Pair));
    }
    if let Some((a, r, _)) = ord!(Rcwsswpp, Rcwsswppl, Rcwsswppa, Rcwsswppal, ()) {
        return Some((1, a, r, 1, 0b010, 0b00, Pair));
    }
    if let Some((a, r, _)) = ord!(Rcwssetp, Rcwssetpl, Rcwssetpa, Rcwssetpal, ()) {
        return Some((1, a, r, 1, 0b011, 0b00, Pair));
    }
    None
}

/// Assemble a FEAT_THE / FEAT_LSE128 atomic word:
/// `size 011 0 01 A R 1 Rs o3 opc(14:12) op2(11:10) Rn Rt`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn the_atomic_word(sz: u32, a: u32, r: u32, rs: u32, o3: u32, opc: u32, op2: u32, rn: u32, rt: u32) -> u32 {
    (sz << 30)
        | (0b011 << 27)
        | (0b01 << 24)
        | (a << 23)
        | (r << 22)
        | (1 << 21)
        | (rs << 16)
        | (o3 << 15)
        | (opc << 12)
        | (op2 << 10)
        | (rn << 5)
        | rt
}

/// Encode a FEAT_THE / FEAT_LSE128 atomic from its [`Code`] and operands.
fn enc_the_atomic(insn: &Instruction) -> R {
    let (sz, a, r, o3, opc, op2, shape) =
        the_atomic_fields(insn.code()).ok_or(EncodeError::Unsupported)?;
    let (rs, rt, rn) = match shape {
        TheShape::SingleW | TheShape::SingleX => {
            // Rs, Rt, [Xn|SP].
            let rs = reg_num(insn, 0)?;
            let rt = reg_num(insn, 1)?;
            let (rn, _, _) = mem_imm(insn, 2)?;
            (rs, rt, rn)
        }
        TheShape::Pair => {
            // Rt, Rs, [Xn|SP] (Rt printed first). Register 31 is reserved in both
            // the Rs and Rt fields for the load-op-pair forms.
            let rt = reg_num(insn, 0)?;
            let rs = reg_num(insn, 1)?;
            let (rn, _, _) = mem_imm(insn, 2)?;
            if rs == 31 || rt == 31 {
                return Err(EncodeError::InvalidOperand);
            }
            (rs, rt, rn)
        }
        TheShape::CasPair => {
            // Rs, Rs+1, Rt, Rt+1, [Xn|SP] — even-register consecutive pairs.
            let rs = reg_num(insn, 0)?;
            let rs1 = reg_num(insn, 1)?;
            let rt = reg_num(insn, 2)?;
            let rt1 = reg_num(insn, 3)?;
            let (rn, _, _) = mem_imm(insn, 4)?;
            if (rs & 1) != 0 || (rt & 1) != 0 {
                return Err(EncodeError::InvalidOperand);
            }
            if rs1 != (rs.wrapping_add(1) & 0x1f) || rt1 != (rt.wrapping_add(1) & 0x1f) {
                return Err(EncodeError::InvalidOperand);
            }
            (rs, rt, rn)
        }
    };
    Ok(the_atomic_word(sz, a, r, rs, o3, opc, op2, rn, rt))
}

/// The atomic RMW family identity recovered from a [`Code`]: `(size, a, r, opc)`
/// where `opc` is the 3-bit operation selector. The byte/half (`*b`/`*h`) and
/// the base 32/64 width codes fold the A/L ordering onto the mnemonic, so we
/// also need the mnemonic to recover `a`/`r` for those.
fn atomic_fields(code: Code) -> Option<u32> {
    // Returns the 3-bit opc; size/ordering come from `atomic_size_ord`.
    atomic_opc(code)
}

/// The 3-bit `opc` for an LD<op> atomic code, or `None`.
#[allow(clippy::too_many_lines)]
fn atomic_opc(code: Code) -> Option<u32> {
    use Code::*;
    Some(match code {
        Ldadd32 | Ldaddl32 | Ldadda32 | Ldaddal32 | Ldadd64 | Ldaddl64 | Ldadda64 | Ldaddal64
        | Ldaddb | Ldaddh => 0b000,
        Ldclr32 | Ldclrl32 | Ldclra32 | Ldclral32 | Ldclr64 | Ldclrl64 | Ldclra64 | Ldclral64
        | Ldclrb | Ldclrh => 0b001,
        Ldeor32 | Ldeorl32 | Ldeora32 | Ldeoral32 | Ldeor64 | Ldeorl64 | Ldeora64 | Ldeoral64
        | Ldeorb | Ldeorh => 0b010,
        Ldset32 | Ldsetl32 | Ldseta32 | Ldsetal32 | Ldset64 | Ldsetl64 | Ldseta64 | Ldsetal64
        | Ldsetb | Ldseth => 0b011,
        Ldsmax32 | Ldsmax64 | Ldsmaxb | Ldsmaxh => 0b100,
        Ldsmin32 | Ldsmin64 | Ldsminb | Ldsminh => 0b101,
        Ldumax32 | Ldumax64 | Ldumaxb | Ldumaxh => 0b110,
        Ldumin32 | Ldumin64 | Lduminb | Lduminh => 0b111,
        _ => return None,
    })
}

/// The `(size, a, r)` for an atomic code, combining the code's width with the
/// (possibly suffix-bearing) mnemonic's ordering for the folded byte/half/base
/// codes.
fn atomic_size_ord(code: Code, mnem: Mnemonic) -> Option<(u32, u32, u32)> {
    use Code::*;
    // First, codes that carry the ordering directly in the *32-bit* name.
    let direct = |size: u32, a: u32, r: u32| Some((size, a, r));
    match code {
        Ldadd32 | Ldclr32 | Ldeor32 | Ldset32 => return direct(2, 0, 0),
        Ldaddl32 | Ldclrl32 | Ldeorl32 | Ldsetl32 => return direct(2, 0, 1),
        Ldadda32 | Ldclra32 | Ldeora32 | Ldseta32 => return direct(2, 1, 0),
        Ldaddal32 | Ldclral32 | Ldeoral32 | Ldsetal32 => return direct(2, 1, 1),
        Ldadd64 | Ldclr64 | Ldeor64 | Ldset64 => return direct(3, 0, 0),
        Ldaddl64 | Ldclrl64 | Ldeorl64 | Ldsetl64 => return direct(3, 0, 1),
        Ldadda64 | Ldclra64 | Ldeora64 | Ldseta64 => return direct(3, 1, 0),
        Ldaddal64 | Ldclral64 | Ldeoral64 | Ldsetal64 => return direct(3, 1, 1),
        // SMAX/SMIN/UMAX/UMIN: base 32/64 width, ordering folded onto mnemonic.
        Ldsmax32 | Ldsmin32 | Ldumax32 | Ldumin32 => return ord_from_mnem(2, mnem),
        Ldsmax64 | Ldsmin64 | Ldumax64 | Ldumin64 => return ord_from_mnem(3, mnem),
        // Byte/half width, ordering folded onto mnemonic.
        Ldaddb | Ldclrb | Ldeorb | Ldsetb | Ldsmaxb | Ldsminb | Ldumaxb | Lduminb => {
            return ord_from_mnem(0, mnem)
        }
        Ldaddh | Ldclrh | Ldeorh | Ldseth | Ldsmaxh | Ldsminh | Ldumaxh | Lduminh => {
            return ord_from_mnem(1, mnem)
        }
        _ => {}
    }
    None
}

/// Recover the `(a, r)` ordering from a folded atomic mnemonic by inspecting its
/// trailing `a`/`l` suffix letters (before any `b`/`h`). Returns
/// `Some((size, a, r))`.
fn ord_from_mnem(size: u32, mnem: Mnemonic) -> Option<(u32, u32, u32)> {
    let name = mnem.name();
    // Strip a trailing 'b'/'h' size letter.
    let core = name
        .strip_suffix('b')
        .or_else(|| name.strip_suffix('h'))
        .unwrap_or(name);
    // The ordering suffix is the trailing 'a'?, 'l'? after the op stem. Detect
    // by checking the last one/two chars.
    let a = u32::from(core.ends_with('a') || core.ends_with("al"));
    let r = u32::from(core.ends_with('l') || core.ends_with("al"));
    Some((size, a, r))
}

/// Encode an LD<op>/ST<op> atomic. The ST<op> alias (Rt==31, A==0) takes
/// `<Rs>, [Xn]`; the LD form takes `<Rs>, <Rt>, [Xn]`.
fn enc_atomic(insn: &Instruction) -> R {
    let code = insn.code();
    let opc = atomic_opc(code).ok_or(EncodeError::Unsupported)?;
    let (size, a, r) = atomic_size_ord(code, insn.mnemonic()).ok_or(EncodeError::Unsupported)?;

    // Distinguish ST<op> alias (mnemonic starts with "st") from LD<op>.
    let is_st = insn.mnemonic().name().starts_with("st");
    let (rs, rt, rn) = if is_st {
        let rs = reg_num(insn, 0)?;
        let (rn, _, _) = mem_imm(insn, 1)?;
        (rs, 0b11111u32, rn)
    } else {
        let rs = reg_num(insn, 0)?;
        let rt = reg_num(insn, 1)?;
        let (rn, _, _) = mem_imm(insn, 2)?;
        (rs, rt, rn)
    };
    let word = atomic_word(size, a, r, rs, 0, opc, rn, rt);
    Ok(word)
}

/// Assemble an LSE atomic word: `size 111 0 00 A R 1 Rs o3 opc 00 Rn Rt`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn atomic_word(size: u32, a: u32, r: u32, rs: u32, o3: u32, opc: u32, rn: u32, rt: u32) -> u32 {
    (size << 30)
        | (0b111 << 27)
        | (a << 23)
        | (r << 22)
        | (1 << 21)
        | (rs << 16)
        | (o3 << 15)
        | (opc << 12)
        | (rn << 5)
        | rt
}

// ---------------------------------------------------------------------------
// FEAT_LSFE atomic floating-point in-memory ops (LDF*/STF* + BF16 LDBF*/STBF*).
// ---------------------------------------------------------------------------

/// Recover the LSFE encoding fields from a [`Code`]:
/// `(o3, a, r, opc, is_bf, is_store)`. The data size (H/S/D) is read from the
/// register operand width; `is_bf` forces `size==00` (BF16). The word is built by
/// [`lsfe_word`].
fn lsfe_fields(code: Code) -> Option<(u32, u32, u32, u32, bool, bool)> {
    use Code::*;
    Some(match code {
        Ldfadd => (0, 0, 0, 0b000, false, false),
        Ldfadda => (0, 1, 0, 0b000, false, false),
        Ldfaddl => (0, 0, 1, 0b000, false, false),
        Ldfaddal => (0, 1, 1, 0b000, false, false),
        Ldfmax => (0, 0, 0, 0b100, false, false),
        Ldfmaxa => (0, 1, 0, 0b100, false, false),
        Ldfmaxl => (0, 0, 1, 0b100, false, false),
        Ldfmaxal => (0, 1, 1, 0b100, false, false),
        Ldfmin => (0, 0, 0, 0b101, false, false),
        Ldfmina => (0, 1, 0, 0b101, false, false),
        Ldfminl => (0, 0, 1, 0b101, false, false),
        Ldfminal => (0, 1, 1, 0b101, false, false),
        Ldfmaxnm => (0, 0, 0, 0b110, false, false),
        Ldfmaxnma => (0, 1, 0, 0b110, false, false),
        Ldfmaxnml => (0, 0, 1, 0b110, false, false),
        Ldfmaxnmal => (0, 1, 1, 0b110, false, false),
        Ldfminnm => (0, 0, 0, 0b111, false, false),
        Ldfminnma => (0, 1, 0, 0b111, false, false),
        Ldfminnml => (0, 0, 1, 0b111, false, false),
        Ldfminnmal => (0, 1, 1, 0b111, false, false),
        Stfadd => (1, 0, 0, 0b000, false, true),
        Stfaddl => (1, 0, 1, 0b000, false, true),
        Stfmax => (1, 0, 0, 0b100, false, true),
        Stfmaxl => (1, 0, 1, 0b100, false, true),
        Stfmin => (1, 0, 0, 0b101, false, true),
        Stfminl => (1, 0, 1, 0b101, false, true),
        Stfmaxnm => (1, 0, 0, 0b110, false, true),
        Stfmaxnml => (1, 0, 1, 0b110, false, true),
        Stfminnm => (1, 0, 0, 0b111, false, true),
        Stfminnml => (1, 0, 1, 0b111, false, true),
        Ldbfadd => (0, 0, 0, 0b000, true, false),
        Ldbfadda => (0, 1, 0, 0b000, true, false),
        Ldbfaddl => (0, 0, 1, 0b000, true, false),
        Ldbfaddal => (0, 1, 1, 0b000, true, false),
        Ldbfmax => (0, 0, 0, 0b100, true, false),
        Ldbfmaxa => (0, 1, 0, 0b100, true, false),
        Ldbfmaxl => (0, 0, 1, 0b100, true, false),
        Ldbfmaxal => (0, 1, 1, 0b100, true, false),
        Ldbfmin => (0, 0, 0, 0b101, true, false),
        Ldbfmina => (0, 1, 0, 0b101, true, false),
        Ldbfminl => (0, 0, 1, 0b101, true, false),
        Ldbfminal => (0, 1, 1, 0b101, true, false),
        Ldbfmaxnm => (0, 0, 0, 0b110, true, false),
        Ldbfmaxnma => (0, 1, 0, 0b110, true, false),
        Ldbfmaxnml => (0, 0, 1, 0b110, true, false),
        Ldbfmaxnmal => (0, 1, 1, 0b110, true, false),
        Ldbfminnm => (0, 0, 0, 0b111, true, false),
        Ldbfminnma => (0, 1, 0, 0b111, true, false),
        Ldbfminnml => (0, 0, 1, 0b111, true, false),
        Ldbfminnmal => (0, 1, 1, 0b111, true, false),
        Stbfadd => (1, 0, 0, 0b000, true, true),
        Stbfaddl => (1, 0, 1, 0b000, true, true),
        Stbfmax => (1, 0, 0, 0b100, true, true),
        Stbfmaxl => (1, 0, 1, 0b100, true, true),
        Stbfmin => (1, 0, 0, 0b101, true, true),
        Stbfminl => (1, 0, 1, 0b101, true, true),
        Stbfmaxnm => (1, 0, 0, 0b110, true, true),
        Stbfmaxnml => (1, 0, 1, 0b110, true, true),
        Stbfminnm => (1, 0, 0, 0b111, true, true),
        Stbfminnml => (1, 0, 1, 0b111, true, true),
        _ => return None,
    })
}

/// Encode a FEAT_LSFE atomic float op:
/// `size 111 1 00 A R 1 Rs o3 opc 00 Rn Rt`. Load form is `<V>s, <V>t, [Xn|SP]`;
/// store form is `<V>s, [Xn|SP]` with `Rt` forced to 31.
fn enc_lsfe(insn: &Instruction) -> R {
    let (o3, a, r, opc, is_bf, is_store) =
        lsfe_fields(insn.code()).ok_or(EncodeError::Unsupported)?;
    // Size from the source register's width: BF16/H -> per code, S=32, D=64.
    let size = if is_bf {
        0
    } else {
        match insn.op(0) {
            Operand::Reg { reg, .. } => match reg.width_bits() {
                16 => 1,
                32 => 2,
                64 => 3,
                _ => return Err(EncodeError::InvalidOperand),
            },
            _ => return Err(EncodeError::InvalidOperand),
        }
    };
    let rs = reg_num(insn, 0)?;
    let (rt, rn) = if is_store {
        let (rn, _, _) = mem_imm(insn, 1)?;
        (0b11111u32, rn)
    } else {
        let rt = reg_num(insn, 1)?;
        let (rn, _, _) = mem_imm(insn, 2)?;
        (rt, rn)
    };
    Ok(lsfe_word(size, a, r, rs, o3, opc, rn, rt))
}

/// Assemble an LSFE atomic word: `size 111 1 00 A R 1 Rs o3 opc 00 Rn Rt`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn lsfe_word(size: u32, a: u32, r: u32, rs: u32, o3: u32, opc: u32, rn: u32, rt: u32) -> u32 {
    (size << 30)
        | (0b111 << 27)
        | (1 << 26)
        | (a << 23)
        | (r << 22)
        | (1 << 21)
        | (rs << 16)
        | (o3 << 15)
        | (opc << 12)
        | (rn << 5)
        | rt
}

/// `LDAPR`/`LDAPRB`/`LDAPRH`: `size 111 0 00 A=1 R=0 1 Rs=11111 o3=1 opc=100 00
/// Rn Rt`.
fn enc_ldapr(insn: &Instruction) -> R {
    use Code::*;
    let size = match insn.code() {
        Ldaprb => 0u32,
        Ldaprh => 1,
        Ldapr32 => 2,
        _ => 3, // Ldapr64
    };
    let rt = reg_num(insn, 0)?;
    let (rn, _, _) = mem_imm(insn, 1)?;
    let word = atomic_word(size, 1, 0, 0b11111, 1, 0b100, rn, rt);
    Ok(word)
}

/// FEAT_LS64 64-byte ops, all `size=11 A=0 R=0 o3=1 opc=... 00`:
///   LD64B   opc=101 Rs=11111 -> `<Xt>, [Xn]`
///   ST64B   opc=001 Rs=11111 -> `<Xt>, [Xn]`
///   ST64BV  opc=011 Rs=<Xs>  -> `<Xs>, <Xt>, [Xn]`
///   ST64BV0 opc=010 Rs=<Xs>  -> `<Xs>, <Xt>, [Xn]`
fn enc_ls64(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();
    let (rs, rt, rn) = match code {
        Ld64b | St64b => {
            let rt = reg_num(insn, 0)?;
            let (rn, _, _) = mem_imm(insn, 1)?;
            (0b11111u32, rt, rn)
        }
        // St64bv / St64bv0
        _ => {
            let rs = reg_num(insn, 0)?;
            let rt = reg_num(insn, 1)?;
            let (rn, _, _) = mem_imm(insn, 2)?;
            (rs, rt, rn)
        }
    };
    let opc = match code {
        Ld64b => 0b101,
        St64b => 0b001,
        St64bv => 0b011,
        _ => 0b010, // St64bv0
    };
    Ok(atomic_word(3, 0, 0, rs, 1, opc, rn, rt))
}

/// `GCSSTR`/`GCSSTTR <Xt>, [<Xn|SP>]` (FEAT_GCS) — the inverse of
/// `decode_gcsstr`. Layout: `11 011001 0 00 11111 000 o(12) 11 Rn Rt`, `word<12>`
/// selecting `GCSSTR`(0)/`GCSSTTR`(1).
fn enc_gcsstr(insn: &Instruction) -> R {
    let o = u32::from(insn.code() == Code::Gcssttr);
    let rt = reg_num(insn, 0)?;
    let (rn, imm, _) = mem_imm(insn, 1)?;
    if imm != 0 {
        return Err(EncodeError::InvalidImmediate);
    }
    // Fixed base `11 011001 0 00 11111 000 _ 11 _____ _____` (word<12> = o).
    let word = 0xd91f_0000u32 | (o << 12) | (0b11 << 10) | (rn << 5) | rt;
    Ok(word)
}

// ---------------------------------------------------------------------------
// Pointer-authenticated loads (LDRAA / LDRAB).
// ---------------------------------------------------------------------------

/// `LDRAA`/`LDRAB`: `11 111 0 00 M S 1 imm9 W 1 Rn Rt`. Offset = SignExtend(S:imm9,
/// 10) << 3.
fn enc_pac(insn: &Instruction) -> R {
    use Code::*;
    let (m, wbit) = match insn.code() {
        LdraaOff => (0u32, 0u32),
        LdraaPre => (0, 1),
        LdrabOff => (1, 0),
        _ => (1, 1), // LdrabPre
    };
    let rt = reg_num(insn, 0)?;
    let (rn, imm, mode) = mem_imm(insn, 1)?;
    let want = if wbit == 1 {
        MemIndexMode::PreIndex
    } else {
        MemIndexMode::Offset
    };
    if mode != want {
        return Err(EncodeError::InvalidOperand);
    }
    // imm = SignExtend(off10,10) << 3 -> off10 = imm >> 3 (10-bit signed).
    if imm & 0b111 != 0 {
        return Err(EncodeError::InvalidImmediate);
    }
    let off10 = imm >> 3;
    if !(-512..=511).contains(&off10) {
        return Err(EncodeError::InvalidImmediate);
    }
    let off10 = (off10 as u32) & 0x3ff;
    let s = (off10 >> 9) & 1;
    let imm9 = off10 & 0x1ff;
    let word = (0b11 << 30)
        | (0b111 << 27)
        | (m << 23)
        | (s << 22)
        | (1 << 21)
        | (imm9 << 12)
        | (wbit << 11)
        | (1 << 10)
        | (rn << 5)
        | rt;
    Ok(word)
}

// ---------------------------------------------------------------------------
// LDAPUR/STLUR (RCpc unscaled).
// ---------------------------------------------------------------------------

/// RCpc-unscaled forms: `Code -> (size, opc)`.
fn ldapstl_fields(code: Code) -> Option<(u32, u32)> {
    use Code::*;
    Some(match code {
        Stlurb => (0, 0b00),
        Ldapurb => (0, 0b01),
        Ldapursb64 => (0, 0b10),
        Ldapursb32 => (0, 0b11),
        Stlurh => (1, 0b00),
        Ldapurh => (1, 0b01),
        Ldapursh64 => (1, 0b10),
        Ldapursh32 => (1, 0b11),
        Stlur32 => (2, 0b00),
        Ldapur32 => (2, 0b01),
        Ldapursw => (2, 0b10),
        Stlur64 => (3, 0b00),
        Ldapur64 => (3, 0b01),
        _ => return None,
    })
}

/// `LDAPUR`/`STLUR`: `size 011001 opc 0 imm9 00 Rn Rt`.
fn enc_ldapstl(insn: &Instruction) -> R {
    let (size, opc) = ldapstl_fields(insn.code()).ok_or(EncodeError::Unsupported)?;
    let rt = reg_num(insn, 0)?;
    let (rn, imm, mode) = mem_imm(insn, 1)?;
    if mode != MemIndexMode::Offset {
        return Err(EncodeError::InvalidOperand);
    }
    let imm9 = encode_imm9(imm)?;
    let word = (size << 30)
        | (0b011001 << 24)
        | (opc << 22)
        | (imm9 << 12)
        | (rn << 5)
        | rt;
    Ok(word)
}

// ---------------------------------------------------------------------------
// FEAT_LRCPC3 SIMD&FP LDAPUR/STLUR and LDIAPP/STILP.
// ---------------------------------------------------------------------------

/// SIMD&FP `LDAPUR`/`STLUR` (FEAT_LRCPC3). Encoding:
/// `size 011 1 01 opc 0 imm9 10 Rn Rt` (`V=1`). The access size is recovered from
/// the data register width: `(size, opc<1>)` is `(acc, 0)` for B/H/S/D and
/// `(00, 1)` for the `Q` view; `opc<0>` is the load bit.
fn enc_fp_ldapstl(insn: &Instruction) -> R {
    use Code::*;
    let load = matches!(
        insn.code(),
        LdapurFp8 | LdapurFp16 | LdapurFp32 | LdapurFp64 | LdapurFp128
    );
    let acc = fp_acc_of(insn.op_register(0)).ok_or(EncodeError::InvalidOperand)?;
    // (size, opc<1>): Q -> (00, 1); B/H/S/D -> (acc, 0).
    let (size, opc_hi) = if acc == 4 { (0u32, 1u32) } else { (acc, 0u32) };
    let opc = (opc_hi << 1) | (load as u32);
    let rt = reg_num(insn, 0)?;
    let (rn, imm, mode) = mem_imm(insn, 1)?;
    if mode != MemIndexMode::Offset {
        return Err(EncodeError::InvalidOperand);
    }
    let imm9 = encode_imm9(imm)?;
    let word = (size << 30)
        | (0b011 << 27)
        | (1 << 26)
        | (0b01 << 24)
        | (opc << 22)
        | (imm9 << 12)
        | (0b10 << 10)
        | (rn << 5)
        | rt;
    Ok(word)
}

/// `LDIAPP`/`STILP` (FEAT_LRCPC3 ordered load/store pair). Encoding:
/// `sz 011001 0 L 0 Rt2 0 0 0 o(12) 1 0 Rn Rt`, `sz` is `10`(W)/`11`(X), `L` the
/// load bit, `o`(bit12)=1 the offset form. The indexed forms (`o==0`) are
/// post-index for the load and pre-index for the store with an implicit
/// datasize displacement.
fn enc_ldiapp_stilp(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();
    let load = matches!(code, LdiappOff | LdiappPost);
    let offset_form = matches!(code, LdiappOff | StilpOff);

    // Both transfer registers carry the W/X width; derive `sz` from it and
    // require Rt2 to match.
    let sz = match insn.op_register(0).width_bits() {
        32 => 0b10u32,
        64 => 0b11,
        _ => return Err(EncodeError::InvalidOperand),
    };
    let rt = reg_num(insn, 0)?;
    let rt2 = reg_num(insn, 1)?;
    let (rn, imm, mode) = mem_imm(insn, 2)?;

    let want_mode = if offset_form {
        MemIndexMode::Offset
    } else if load {
        MemIndexMode::PostImm
    } else {
        MemIndexMode::PreIndex
    };
    if mode != want_mode {
        return Err(EncodeError::InvalidOperand);
    }
    // Validate the implicit displacement of the indexed/offset forms.
    let bytes: i64 = if sz == 0b11 { 16 } else { 8 };
    let want_imm = if offset_form {
        0
    } else if load {
        bytes
    } else {
        -bytes
    };
    if imm != want_imm {
        return Err(EncodeError::InvalidOperand);
    }

    let l = load as u32;
    let o = offset_form as u32;
    let word = (sz << 30)
        | (0b011001 << 24)
        | (l << 22)
        | (rt2 << 16)
        | (o << 12)
        | (1 << 11)
        | (rn << 5)
        | rt;
    Ok(word)
}

/// FEAT_LRCPC3 ordered load/store pair `LDAPP`/`LDAP`/`STLP` (X-only, no offset).
/// Encoding: `11 011001 0 L 0 Rt2 opc2 10 Rn Rt`, with `opc2`(bits<15:12>) =
/// `0101` for STLP(L=0)/LDAP(L=1) and `0111` for LDAPP(L=1). All forms are
/// 64-bit-only with a plain `[<Xn|SP>]` base and no writeback.
fn enc_ldapp_stlp(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();
    let (l, opc2) = match code {
        StlpPair => (0u32, 0b0101u32),
        LdapPair => (1, 0b0101),
        LdappPair => (1, 0b0111),
        _ => return Err(EncodeError::InvalidOperand),
    };
    // Both transfer registers must be 64-bit X.
    if insn.op_register(0).width_bits() != 64 || insn.op_register(1).width_bits() != 64 {
        return Err(EncodeError::InvalidOperand);
    }
    let rt = reg_num(insn, 0)?;
    let rt2 = reg_num(insn, 1)?;
    let (rn, imm, mode) = mem_imm(insn, 2)?;
    if mode != MemIndexMode::Offset || imm != 0 {
        return Err(EncodeError::InvalidOperand);
    }
    let word = (0b11u32 << 30)
        | (0b011001 << 24)
        | (l << 22)
        | (rt2 << 16)
        | (opc2 << 12)
        | (0b10 << 10)
        | (rn << 5)
        | rt;
    Ok(word)
}

/// FEAT_LRCPC3 writeback `STLR` (pre-index) / `LDAPR` (post-index). Encoding:
/// `sz 011001 1 L 0 000000000 10 Rn Rt`. The displacement is the implicit access
/// size and must match (`-dsz` for the pre-index store, `+dsz` for the post-index
/// load); the `imm9` field is always zero.
fn enc_stlr_ldapr_wb(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();
    let (sz, load) = match code {
        StlrPre32 => (0b10u32, false),
        StlrPre64 => (0b11, false),
        LdaprPost32 => (0b10, true),
        LdaprPost64 => (0b11, true),
        _ => return Err(EncodeError::Unsupported),
    };
    let rt = reg_num(insn, 0)?;
    let (rn, imm, mode) = mem_imm(insn, 1)?;
    let bytes: i64 = if sz == 0b11 { 8 } else { 4 };
    let (want_mode, want_imm) = if load {
        (MemIndexMode::PostImm, bytes)
    } else {
        (MemIndexMode::PreIndex, -bytes)
    };
    if mode != want_mode || imm != want_imm {
        return Err(EncodeError::InvalidOperand);
    }
    let l = load as u32;
    let word = (sz << 30)
        | (0b011001 << 24)
        | (1 << 23)
        | (l << 22)
        | (0b10 << 10)
        | (rn << 5)
        | rt;
    Ok(word)
}

// ---------------------------------------------------------------------------
// Memory tagging (STG/STZG/ST2G/STZ2G/LDG/LDGM/STGM/STZGM).
// ---------------------------------------------------------------------------

/// `true` if `code` is one of the MTE tag forms handled here.
fn tag_is(code: Code) -> bool {
    use Code::*;
    matches!(
        code,
        StgPost
            | StgOff
            | StgPre
            | StzgPost
            | StzgOff
            | StzgPre
            | St2gPost
            | St2gOff
            | St2gPre
            | Stz2gPost
            | Stz2gOff
            | Stz2gPre
            | Stzgm
            | LdgOff
            | Stgm
            | Ldgm
    )
}

/// Memory-tagging stores/loads. Encoding:
/// `1101 1001 opc(23:22) 1 imm9 op2(11:10) Rn Rt`.
fn enc_tags(insn: &Instruction) -> R {
    use Code::*;
    let code = insn.code();

    // The op2!=00 family (STG/STZG/ST2G/STZ2G): opc selects the family, op2 the
    // addressing mode; Rt is Xt|SP.
    let stg = |opc: u32, op2: u32| Some((opc, op2));
    if let Some((opc, op2)) = match code {
        StgPost => stg(0b00, 0b01),
        StgOff => stg(0b00, 0b10),
        StgPre => stg(0b00, 0b11),
        StzgPost => stg(0b01, 0b01),
        StzgOff => stg(0b01, 0b10),
        StzgPre => stg(0b01, 0b11),
        St2gPost => stg(0b10, 0b01),
        St2gOff => stg(0b10, 0b10),
        St2gPre => stg(0b10, 0b11),
        Stz2gPost => stg(0b11, 0b01),
        Stz2gOff => stg(0b11, 0b10),
        Stz2gPre => stg(0b11, 0b11),
        _ => None,
    } {
        let rt = reg_num(insn, 0)?;
        let (rn, imm, mode) = mem_imm(insn, 1)?;
        let want = match op2 {
            0b01 => MemIndexMode::PostImm,
            0b10 => MemIndexMode::Offset,
            _ => MemIndexMode::PreIndex,
        };
        if mode != want {
            return Err(EncodeError::InvalidOperand);
        }
        let imm9 = encode_tag_imm9(imm)?;
        return Ok(tag_word(opc, imm9, op2, rn, rt));
    }

    // The op2==00 family.
    let (opc, with_off) = match code {
        Stzgm => (0b00u32, false),
        LdgOff => (0b01, true),
        Stgm => (0b10, false),
        Ldgm => (0b11, false),
        _ => return Err(EncodeError::Unsupported),
    };
    let rt = reg_num(insn, 0)?;
    let (rn, imm, mode) = mem_imm(insn, 1)?;
    if mode != MemIndexMode::Offset {
        return Err(EncodeError::InvalidOperand);
    }
    let imm9 = if with_off {
        encode_tag_imm9(imm)?
    } else {
        if imm != 0 {
            return Err(EncodeError::InvalidImmediate);
        }
        0
    };
    Ok(tag_word(opc, imm9, 0b00, rn, rt))
}

/// Scale + sign-encode a tag immediate (granule = 16 bytes -> shift 4).
fn encode_tag_imm9(imm: i64) -> Result<u32, EncodeError> {
    if imm & 0xf != 0 {
        return Err(EncodeError::InvalidImmediate);
    }
    let v = imm >> 4;
    if !(-256..=255).contains(&v) {
        return Err(EncodeError::InvalidImmediate);
    }
    Ok((v as u32) & 0x1ff)
}

/// Assemble a memory-tagging word: `11011001 opc 1 imm9 op2 Rn Rt`.
#[inline]
fn tag_word(opc: u32, imm9: u32, op2: u32, rn: u32, rt: u32) -> u32 {
    (0b11011001 << 24) | (opc << 22) | (1 << 21) | (imm9 << 12) | (op2 << 10) | (rn << 5) | rt
}

#[cfg(test)]
mod tests {
    use crate::features::FeatureSet;
    use crate::instruction::Instruction;

    /// Decode a word, re-encode it, and require a **semantic** round-trip: the
    /// re-encoded word must decode back to the identical instruction. Exact-word
    /// equality is the common case, but some encodings carry bits the decoder
    /// discards (e.g. the unused `Rs`/`Rt2` fields of a load-exclusive), so the
    /// authoritative criterion is semantic equality.
    #[track_caller]
    fn rt(word: u32) {
        let mut insn = Instruction::default();
        crate::decode::decode_into(word, 0x8000_0000_0000_0004, FeatureSet::ALL, &mut insn);
        assert!(!insn.is_invalid(), "word {word:#010x} failed to decode");
        let got = insn
            .encode()
            .unwrap_or_else(|e| panic!("encode of {word:#010x} ({:?}) failed: {e:?}", insn.code()));
        if got == word {
            return;
        }
        // Not exact: accept iff it re-decodes to a semantically-equal instruction.
        let mut re = Instruction::default();
        crate::decode::decode_into(got, 0x8000_0000_0000_0004, FeatureSet::ALL, &mut re);
        let same = !re.is_invalid()
            && re.code() == insn.code()
            && re.mnemonic() == insn.mnemonic()
            && re.op_count() == insn.op_count()
            && (0..insn.op_count()).all(|i| re.op(i) == insn.op(i));
        assert!(
            same,
            "round-trip mismatch for {word:#010x}: re-encoded {got:#010x} (code={:?}, mnem={:?})",
            insn.code(),
            insn.mnemonic()
        );
    }

    #[test]
    fn ldst_known_words() {
        // Unsigned offset.
        rt(0xF96ECD59); // ldr x25, [x10, #0x5d98]
        rt(0xB9205D5A); // str w26, [x10, #0x205c]
        rt(0x394FEDDF); // ldrb wzr, [x14, #0x3fb]
        rt(0x7983E17B); // ldrsh x27, [x11, #0x1f0]
        rt(0x3D4666EE); // ldr b14, [x23, #0x199]
        rt(0x3DDEABF2); // ldr q18, [sp, #0x7aa0]
        // Register offset.
        rt(0xF87B5ADC); // ldr x28, [x22, w27, uxtw #0x3]
        rt(0xF86D7B64); // ldr x4, [x27, x13, lsl #0x3]
        rt(0xBC7F6A15); // ldr s21, [x16, xzr]
        rt(0x3863E8A0); // ldrb w0, [x5, x3, sxtx]
        rt(0x3862F9C1); // ldrb w1, [x14, x2, sxtx #0x0]
        rt(0x387E7B04); // ldrb w4, [x24, x30, lsl #0x0]
        // Unscaled / pre / post.
        rt(0xF843F3F3); // ldur x19, [sp, #0x3f]
        rt(0xF845EC10); // ldr x16, [x0, #0x5e]!
        rt(0xF85FA74E); // ldr x14, [x26], #-0x6
        rt(0xF85E09FB); // ldtr x27, [x15, #-0x20]
        // Literal + prefetch.
        rt(0x58564E32); // ldr x18, <lit>
        rt(0x98993236); // ldrsw x22, <lit>
        rt(0xD80580C5); // prfm pldl3strm, <lit>
        rt(0xF9818708); // prfm plil1keep, [x24, #0x308]
        rt(0xF89961AF); // prfum #0xf, [x13, #-0x6a]
        // Pair.
        rt(0xA943A9CC); // ldp x12, x10, [x14, #0x38]
        rt(0xA8EA2124); // ldp x4, x8, [x9], #-0x160
        rt(0xA9EACE40); // ldp x0, x19, [x18, #-0x158]!
        rt(0x69723695); // ldpsw x21, x13, [x20, #-0x70]
        rt(0xA86520FA); // ldnp x26, x8, [x7, #-0x1b0]
        rt(0xAD576824); // ldp q4, q26, [x1, #0x2e0]
        // Exclusive / ordered.
        rt(0xC8572629); // ldxr x9, [x17]
        rt(0xC806455E); // stxr w6, x30, [x10]
        rt(0xC8677530); // ldxp x16, x29, [x9]
        rt(0xC82E295F); // stxp w14, xzr, x10, [x10]
        rt(0xC8DFFD72); // ldar x18, [x11]
        rt(0xC88DD332); // stlr x18, [x25]
        rt(0xC8C124B7); // ldlar x23, [x5]
        // LSE atomics + CAS.
        rt(0xC8BB7E15); // cas x27, x21, [x16]
        rt(0x48227F0A); // casp x2, x3, x10, x11, [x24]
        rt(0x88EDFC7E); // casal w13, w30, [x3]
        rt(0xF82400B8); // ldadd x4, x24, [x5]
        rt(0xF8E00025); // ldaddal x0, x5, [x1]
        rt(0xF82183F0); // swp x1, x16, [sp]
        rt(0xF83F021F); // stadd xzr, [x16]
        rt(0xF8BFC058); // ldapr x24, [x2]
        // PAC, RCpc unscaled, tags.
        rt(0xF8246596); // ldraa x22, [x12, #0x230]
        rt(0xF8AC1C5A); // ldrab x26, [x2, #0x608]!
        rt(0xD94CC1FF); // ldapur xzr, [x15, #0xcc]
        rt(0x999D5150); // ldapursw x16, [x10, #-0x2b]
        rt(0xD9A55BC6); // st2g x6, [x30, #0x550]
        rt(0xD938D482); // stg x2, [x4], #-0x730
        rt(0xD96583D1); // ldg x17, [x30, #0x580]
        rt(0xD92003C5); // stzgm x5, [x30]
        rt(0x691FC8DC); // stgp x28, x18, [x6, #0x3f0]
    }
}
