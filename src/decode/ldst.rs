//! Loads and Stores (ARM ARM C4.1.96).
//!
//! Hand-written decoder dispatched here from [`crate::decode::decode_into`] when
//! `op0 = word<28:25>` selects the load/store group (`x1x0`). This is the widest
//! A64 group; it sub-classes on `word<31:21>` and covers:
//!
//! * Load register (literal / PC-relative).
//! * Load/store register (unsigned immediate offset).
//! * Load/store register (immediate pre/post-index, unscaled `*UR`, unprivileged
//!   `*TR`).
//! * Load/store register (register offset, with extend/scale).
//! * Load/store pair (no-alloc, signed offset, pre/post-index) + SIMD&FP pairs.
//! * Load/store exclusive and the acquire/release ordered forms.
//! * Compare-and-swap (`CAS`/`CASP`) and atomic memory operations (LSE) — gated
//!   on [`Feature::Lse`].
//! * Pointer-authenticated `LDRAA`/`LDRAB` — gated on [`Feature::PAuth`].
//! * `LDAPUR`/`STLUR` (RCpc unscaled) and the memory-tagging
//!   `STG`/`STZG`/`ST2G`/`STZ2G`/`STGP`/`LDG`/`LDGM`/`STGM`/`STZGM` — tag forms
//!   gated on [`Feature::Mte`].
//!
//! The Advanced-SIMD *structure* load/stores (`LD1`..`LD4`, `ST1`..`ST4`, single
//! and multiple structures, `LD1R` ...) are intentionally left to the SIMD group
//! and are **not** decoded here.

use crate::decode::bits::{bit, bits, sign_extend};
use crate::decode::ldst_simd;
use crate::enums::ExtendType;
use crate::features::{Feature, FeatureSet};
use crate::instruction::Instruction;
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::{MemIndexMode, Operand};
use crate::register::{gp_register, RegWidth, Register};
use crate::sysop::SysToken;

// ---------------------------------------------------------------------------
// Register builders.
// ---------------------------------------------------------------------------

/// Build a plain GP register operand (`use_sp` selects SP vs ZR for reg-31).
#[inline]
fn gp(use_sp: bool, w: RegWidth, n: u32) -> Operand {
    Operand::Reg {
        reg: gp_register(use_sp, w, (n & 0x1f) as u8),
        arr: None,
        lane: None,
        shift: None,
        extend: None,
        pred: None,
    }
}

/// The GP width for a 1-bit size selector (`1` => X/64, `0` => W/32).
#[inline]
fn w_of(x: u32) -> RegWidth {
    if x & 1 == 1 {
        RegWidth::X64
    } else {
        RegWidth::W32
    }
}

/// Wrap a scalar-FP/SIMD register in a bare operand.
#[inline]
fn simd_op(r: Register) -> Operand {
    Operand::Reg {
        reg: r,
        arr: None,
        lane: None,
        shift: None,
        extend: None,
        pred: None,
    }
}

/// The scalar-FP register `<class><n>` for a 0..=4 size code
/// (`0`=B, `1`=H, `2`=S, `3`=D, `4`=Q). Out-of-range codes fall back to `Q`.
#[inline]
fn fp_reg(size_code: u32, n: u32) -> Register {
    let n = (n & 0x1f) as usize;
    match size_code {
        0 => B[n],
        1 => H[n],
        2 => S[n],
        3 => D[n],
        _ => Q[n],
    }
}

// Numbered scalar-FP register tables (the enum lays these out contiguously, but
// we list them so no discriminant arithmetic / transmute is needed).
#[rustfmt::skip]
const B: [Register; 32] = [
    Register::B0, Register::B1, Register::B2, Register::B3, Register::B4, Register::B5, Register::B6, Register::B7,
    Register::B8, Register::B9, Register::B10, Register::B11, Register::B12, Register::B13, Register::B14, Register::B15,
    Register::B16, Register::B17, Register::B18, Register::B19, Register::B20, Register::B21, Register::B22, Register::B23,
    Register::B24, Register::B25, Register::B26, Register::B27, Register::B28, Register::B29, Register::B30, Register::B31,
];
#[rustfmt::skip]
const H: [Register; 32] = [
    Register::H0, Register::H1, Register::H2, Register::H3, Register::H4, Register::H5, Register::H6, Register::H7,
    Register::H8, Register::H9, Register::H10, Register::H11, Register::H12, Register::H13, Register::H14, Register::H15,
    Register::H16, Register::H17, Register::H18, Register::H19, Register::H20, Register::H21, Register::H22, Register::H23,
    Register::H24, Register::H25, Register::H26, Register::H27, Register::H28, Register::H29, Register::H30, Register::H31,
];
#[rustfmt::skip]
const S: [Register; 32] = [
    Register::S0, Register::S1, Register::S2, Register::S3, Register::S4, Register::S5, Register::S6, Register::S7,
    Register::S8, Register::S9, Register::S10, Register::S11, Register::S12, Register::S13, Register::S14, Register::S15,
    Register::S16, Register::S17, Register::S18, Register::S19, Register::S20, Register::S21, Register::S22, Register::S23,
    Register::S24, Register::S25, Register::S26, Register::S27, Register::S28, Register::S29, Register::S30, Register::S31,
];
#[rustfmt::skip]
const D: [Register; 32] = [
    Register::D0, Register::D1, Register::D2, Register::D3, Register::D4, Register::D5, Register::D6, Register::D7,
    Register::D8, Register::D9, Register::D10, Register::D11, Register::D12, Register::D13, Register::D14, Register::D15,
    Register::D16, Register::D17, Register::D18, Register::D19, Register::D20, Register::D21, Register::D22, Register::D23,
    Register::D24, Register::D25, Register::D26, Register::D27, Register::D28, Register::D29, Register::D30, Register::D31,
];
#[rustfmt::skip]
const Q: [Register; 32] = [
    Register::Q0, Register::Q1, Register::Q2, Register::Q3, Register::Q4, Register::Q5, Register::Q6, Register::Q7,
    Register::Q8, Register::Q9, Register::Q10, Register::Q11, Register::Q12, Register::Q13, Register::Q14, Register::Q15,
    Register::Q16, Register::Q17, Register::Q18, Register::Q19, Register::Q20, Register::Q21, Register::Q22, Register::Q23,
    Register::Q24, Register::Q25, Register::Q26, Register::Q27, Register::Q28, Register::Q29, Register::Q30, Register::Q31,
];

// ---------------------------------------------------------------------------
// Memory-operand helpers.
// ---------------------------------------------------------------------------

/// `[Xn|SP, #imm]` (or `[Xn|SP]` when imm==0). Base is always SP-capable.
#[inline]
fn mem_off(rn: u32, imm: i64) -> Operand {
    Operand::MemImm {
        base: gp_register(true, RegWidth::X64, (rn & 0x1f) as u8),
        imm,
        mode: MemIndexMode::Offset,
    }
}

/// `[Xn|SP], #imm` (post-index by immediate).
#[inline]
fn mem_post(rn: u32, imm: i64) -> Operand {
    Operand::MemImm {
        base: gp_register(true, RegWidth::X64, (rn & 0x1f) as u8),
        imm,
        mode: MemIndexMode::PostImm,
    }
}

/// `[Xn|SP, #imm]!` (pre-index by immediate).
#[inline]
fn mem_pre(rn: u32, imm: i64) -> Operand {
    Operand::MemImm {
        base: gp_register(true, RegWidth::X64, (rn & 0x1f) as u8),
        imm,
        mode: MemIndexMode::PreIndex,
    }
}

/// The prefetch operand for a 5-bit `Rt`. Returns the named `<type>l<n><policy>`
/// keyword when the type (`pld`/`pli`/`pst`) and target (`l1`/`l2`/`l3`) are
/// allocated, else a raw `#imm` (matching Binary Ninja's fall-back).
#[inline]
fn prefetch_op(rt: u32) -> Operand {
    let ty = (rt >> 3) & 3; // 00 pld, 01 pli, 10 pst, 11 reserved
    let target = (rt >> 1) & 3; // 00 l1, 01 l2, 10 l3, 11 reserved
    let policy = rt & 1; // 0 keep, 1 strm
    if ty == 3 || target == 3 {
        return Operand::ImmUnsigned(rt as u64);
    }
    let name = match (ty, target, policy) {
        (0, 0, 0) => "pldl1keep",
        (0, 0, 1) => "pldl1strm",
        (0, 1, 0) => "pldl2keep",
        (0, 1, 1) => "pldl2strm",
        (0, 2, 0) => "pldl3keep",
        (0, 2, 1) => "pldl3strm",
        (1, 0, 0) => "plil1keep",
        (1, 0, 1) => "plil1strm",
        (1, 1, 0) => "plil2keep",
        (1, 1, 1) => "plil2strm",
        (1, 2, 0) => "plil3keep",
        (1, 2, 1) => "plil3strm",
        (2, 0, 0) => "pstl1keep",
        (2, 0, 1) => "pstl1strm",
        (2, 1, 0) => "pstl2keep",
        (2, 1, 1) => "pstl2strm",
        (2, 2, 0) => "pstl3keep",
        _ => "pstl3strm",
    };
    Operand::SysOp(SysToken::of(name))
}

/// The range-prefetch operand for a 6-bit `rprfop` field (`RPRFM`). Only four
/// values are named: `pldkeep`(0)/`pstkeep`(1)/`pldstrm`(4)/`pststrm`(5) — i.e.
/// `imm6<5:3>==000 && imm6<1>==0`, with `imm6<0>` selecting `pld`/`pst` and
/// `imm6<2>` the `keep`/`strm` policy. Everything else renders as a raw `#imm`,
/// matching LLVM/Binary Ninja.
#[inline]
fn rprfop_op(imm6: u32) -> Operand {
    let imm6 = imm6 & 0x3f;
    // Named only when bits<5:3>==0 and bit1==0 (no level component).
    if (imm6 & 0b111010) == 0 {
        let name = match (imm6 & 0b100, imm6 & 1) {
            (0, 0) => "pldkeep",
            (0, _) => "pstkeep",
            (_, 0) => "pldstrm",
            (_, _) => "pststrm",
        };
        return Operand::SysOp(SysToken::of(name));
    }
    Operand::ImmUnsigned(imm6 as u64)
}

/// The SVE prefetch operand for a 4-bit `prfop` field (`PRFB`/`PRFH`/`PRFW`/
/// `PRFD`). SVE prefetch has no `pli` (instruction-prefetch) type: bit 3 selects
/// `pld`(0)/`pst`(1), bits<2:1> the target (`l1`/`l2`/`l3`/reserved) and bit 0
/// the policy. The reserved target (`11`) renders as a raw `#imm`, matching the
/// corpus (`#0x6`, `#0x7`, `#0xe`, `#0xf`).
///
/// Only referenced by the SVE prefetch decoders, so it is gated on the `sve`
/// feature to keep the default build warning-free.
#[cfg(feature = "sve")]
#[inline]
pub(crate) fn prefetch_op_sve(prfop: u32) -> Operand {
    let prfop = prfop & 0xf;
    let target = (prfop >> 1) & 3; // 00 l1, 01 l2, 10 l3, 11 reserved
    if target == 3 {
        return Operand::ImmUnsigned(prfop as u64);
    }
    let name = match prfop {
        0b0000 => "pldl1keep",
        0b0001 => "pldl1strm",
        0b0010 => "pldl2keep",
        0b0011 => "pldl2strm",
        0b0100 => "pldl3keep",
        0b0101 => "pldl3strm",
        0b1000 => "pstl1keep",
        0b1001 => "pstl1strm",
        0b1010 => "pstl2keep",
        0b1011 => "pstl2strm",
        0b1100 => "pstl3keep",
        _ => "pstl3strm",
    };
    Operand::SysOp(SysToken::of(name))
}

// ---------------------------------------------------------------------------
// Top-level dispatch.
// ---------------------------------------------------------------------------

/// Decode a Loads-and-Stores instruction into `out`.
///
/// `ip` is required for the literal (PC-relative) load forms. LSE atomics,
/// pointer-authenticated and memory-tagging forms are only decoded when their
/// [`Feature`] is accepted by `features`; otherwise the encoding is left invalid.
#[inline]
pub fn decode(word: u32, ip: u64, features: FeatureSet, out: &mut Instruction) {
    // The load/store group is partitioned on word<31:28>:op0, word<26>:op1(V),
    // word<24:23>:op2, word<21>:op3 and word<11:10>:op4 (ARM ARM C4.1.96 table).
    // We dispatch on a few discriminating bits and let each sub-decoder do the
    // rest of the field checks. (`word<31:28>`=op0 and `word<28>`=op1 from the
    // ARM ARM table are not needed directly; the sub-class selector below uses
    // the discriminating high bits instead.)

    // Use the canonical sub-class selector: bits<29:28> = op0<1:0>, and
    // bits<25:24>, bit<21>, bits<11:10>. Concretely:
    //   word<29:28>:
    //     For most ld/st-register forms word<27:24> == 0b11x0/0b11x1.
    // Rather than reverse-engineer the whole tree, branch on stable high bits.

    let b2725 = bits(word, 25, 3); // word<27:25>
    let b28 = bit(word, 28);

    // Load/store register (all single-register forms) and pair forms share
    // word<27:25> == 0b111 (with op0 high bits). Exclusive/ordered/atomic share
    // word<27:24> == 0b0010. Tags share word<29:24> == 0b011001.
    // Dispatch precisely on the documented field combos.

    // 1) Load register (literal): op0<1:0>=0bx1? -> word<29:27> == 0b011, V=word<26>.
    //    Encoding: opc(31:30) 011 V 00 imm19 Rt.
    if bits(word, 27, 3) == 0b011 && bits(word, 24, 2) == 0b00 {
        decode_literal(word, ip, out);
        return;
    }

    // 2) Load/store exclusive: word<29:24> == 0b001000.
    if bits(word, 24, 6) == 0b001000 {
        decode_exclusive(word, features, out);
        return;
    }

    // 2b) FEAT_LSUI unprivileged atomics: word<29:24> == 0b001001. The same
    //     exclusive/CAS layout but with the group's low bit (word<24>) set —
    //     unprivileged load/store-exclusive (LDTXR/LDATXR/STTXR/STLTXR) and
    //     unprivileged compare-and-swap (CAST/.../CASPT/...).
    if bits(word, 24, 6) == 0b001001 {
        decode_lsui(word, features, out);
        return;
    }

    // 3) Load/store ordered (LDAR/STLR/...): word<29:24> == 0b001000 already
    //    handled above (ordered shares the exclusive major class). Nothing here.

    // 3b) Advanced SIMD load/store structures (LD1..LD4 / ST1..ST4, multiple and
    //     single, plus LD1R..LD4R). word<29:24> == 0b001100 (multiple) or
    //     0b001101 (single/replicate), with word<31> == 0. Must precede the pair
    //     check below, since the single form's word<27:25> aliases the pair's.
    if bit(word, 31) == 0 {
        match bits(word, 24, 6) {
            0b001100 | 0b001101 => {
                ldst_simd::decode(word, out);
                return;
            }
            _ => {}
        }
    }

    // 4a) FEAT_MOPS Memory Copy / Memory Set. These live in the loads/stores
    //     region with the fixed signature `word<31:30> == 0b00`,
    //     `word<29:27> == 0b011`, `word<25:24> == 0b01`, `word<21> == 0` and
    //     `word<11:10> == 0b01` (the family-class bit `word<26>` selects
    //     `CPYF*`/`SET*` vs `CPY*`/`SETG*`). Match the precise signature so we do
    //     not shadow the neighbouring RCpc-unscaled / tagging / RCPC3 forms.
    if bits(word, 30, 2) == 0b00
        && bits(word, 27, 3) == 0b011
        && bits(word, 24, 2) == 0b01
        && bit(word, 21) == 0
        && bits(word, 10, 2) == 0b01
    {
        if features.has(Feature::Mops) {
            crate::decode::mops::decode(word, out);
        }
        return;
    }

    // 4b) Memory tagging + GP LDAPUR/STLUR + RCPC3 LDIAPP/STILP live under
    //     word<29:24> == 0b011001.
    if bits(word, 24, 6) == 0b011001 {
        decode_ldst_tags_or_ldapstl(word, features, out);
        return;
    }

    // 4c) FEAT_LRCPC3 SIMD&FP LDAPUR/STLUR: word<31:30>=size, word<29:27>=0b011,
    //     V(26)=1, word<25:24>=0b01, word<23:22>=opc, word<21>=0, imm9 then
    //     word<11:10>=0b10. Distinct from the GP forms above (which clear V) and
    //     from the SIMD literal (word<25:24>=0b00) / unscaled-register family
    //     (word<29:27>=0b111).
    if bits(word, 27, 3) == 0b011
        && bit(word, 26) == 1
        && bits(word, 24, 2) == 0b01
        && bit(word, 21) == 0
        && bits(word, 10, 2) == 0b10
    {
        decode_fp_ldapstl(word, features, out);
        return;
    }

    // 5) Load/store register pair: word<29:25> == 0b101 0 (i.e. word<29:27>==0b101).
    if bits(word, 27, 3) == 0b101 {
        decode_pair(word, features, out);
        return;
    }

    // 6) Everything with word<29:27> == 0b111 is the load/store-register family
    //    (unsigned offset, imm pre/post, unscaled, unprivileged, register
    //    offset, atomics, PAC).
    if bits(word, 27, 3) == 0b111 {
        // word<24> == 1 -> unsigned immediate offset.
        if bit(word, 24) == 1 {
            decode_reg_unsigned(word, out);
            return;
        }
        // word<24> == 0: sub-classed on word<21> and word<11:10>.
        if bit(word, 21) == 1 {
            // word<11:10>:
            //   10 -> register offset
            //   00 -> atomic memory op (LSE) [op3=1, op4=00]
            //   01 -> LDAPR/STLR-ish (RCpc register-... actually LDAPR) -> handled?
            match bits(word, 10, 2) {
                0b10 => decode_reg_offset(word, features, out),
                0b00 => decode_atomic(word, features, out),
                0b01 => decode_pac(word, features, out),
                _ => decode_pac(word, features, out),
            }
            return;
        } else {
            // word<21> == 0: immediate forms keyed on word<11:10>.
            match bits(word, 10, 2) {
                0b00 => decode_reg_unscaled(word, out), // LDUR/STUR
                0b01 => decode_reg_immpost(word, out),  // post-index
                0b10 => decode_reg_unpriv(word, out),   // LDTR/STTR
                _ => decode_reg_immpre(word, out),      // pre-index (0b11)
            }
            return;
        }
    }

    let _ = (b2725, b28);
    // Anything else: leave invalid.
}

// ---------------------------------------------------------------------------
// (size, V, opc) -> (mnemonic, register-class, scale) classification.
// ---------------------------------------------------------------------------

/// The decoded identity of a single load/store register form.
struct LdStForm {
    code: Code,
    /// `true` if the data register is scalar-FP/SIMD (`reg_size` is then the
    /// B/H/S/D/Q code 0..4); otherwise it is a GP register and `gp_x` selects X.
    is_fp: bool,
    /// For FP forms: B/H/S/D/Q code (0..4). Unused for GP.
    fp_code: u32,
    /// For GP forms: `true` => X (64-bit) register, `false` => W (32-bit).
    gp_x: bool,
    /// Log2 of the access size (the immediate scale for the scaled forms).
    scale: u32,
    /// `true` if this is a `PRFM`-class encoding (the data slot is a prefetch op).
    is_prfm: bool,
}

/// Classify a load/store register form from `(size, V, opc)`, returning the
/// `imm_unsigned`-style identity (the same selector drives every register
/// addressing mode; only the immediate scaling differs). `unsigned` selects the
/// unsigned-offset/regoff codes vs the unscaled/imm/unpriv (`*UR`/`*TR`) codes.
fn classify_reg(size: u32, v: u32, opc: u32, variant: RegVariant) -> Option<LdStForm> {
    // Helper to make an FP form with explicit scale.
    let fpf = |code: Code, fp_code: u32, scale: u32| LdStForm {
        code,
        is_fp: true,
        fp_code,
        gp_x: false,
        scale,
        is_prfm: false,
    };

    if v == 1 {
        // SIMD&FP register forms. Access size log2 = (opc<1> << 2) | size.
        let acc = ((opc >> 1) << 2) | size;
        // Only B/H/S/D/Q (acc 0..4) are valid.
        if acc > 4 {
            return None;
        }
        let load = (opc & 1) == 1;
        let code = match variant {
            RegVariant::Unsigned => {
                if load {
                    [
                        Code::LdrFpImmUnsigned8,
                        Code::LdrFpImmUnsigned16,
                        Code::LdrFpImmUnsigned32,
                        Code::LdrFpImmUnsigned64,
                        Code::LdrFpImmUnsigned128,
                    ][acc as usize]
                } else {
                    [
                        Code::StrFpImmUnsigned8,
                        Code::StrFpImmUnsigned16,
                        Code::StrFpImmUnsigned32,
                        Code::StrFpImmUnsigned64,
                        Code::StrFpImmUnsigned128,
                    ][acc as usize]
                }
            }
            RegVariant::Unscaled => {
                // LDUR/STUR (SIMD&FP) share the Ldur/Stur scalar codes by view.
                if load {
                    fp_unscaled_code(acc, true)
                } else {
                    fp_unscaled_code(acc, false)
                }
            }
            RegVariant::ImmPost | RegVariant::ImmPre => {
                if load {
                    fp_immidx_code(acc, true, variant)
                } else {
                    fp_immidx_code(acc, false, variant)
                }
            }
            RegVariant::RegOff => {
                // B/H/S/D/Q register-offset forms all exist. The B and H forms
                // reuse a 32-bit FP-reg code carrier (mnemonic stays LDR/STR; the
                // register view comes from `fp_code`).
                match (acc, load) {
                    (0, true) | (1, true) | (2, true) => Code::LdrFpReg32,
                    (3, true) => Code::LdrFpReg64,
                    (4, true) => Code::LdrFpReg128,
                    (0, false) | (1, false) | (2, false) => Code::StrFpReg32,
                    (3, false) => Code::StrFpReg64,
                    (4, false) => Code::StrFpReg128,
                    _ => return None,
                }
            }
        };
        return Some(fpf(code, acc, acc));
    }

    // General-purpose register forms.
    // opc selects the operation; the L/sign/width come from the table below.
    match variant {
        RegVariant::Unsigned | RegVariant::RegOff => classify_gp(size, opc, variant),
        RegVariant::Unscaled => classify_gp_unscaled(size, opc),
        RegVariant::ImmPost | RegVariant::ImmPre => classify_gp_immidx(size, opc, variant),
    }
}

/// The addressing-mode variant of a load/store-register form.
#[derive(Clone, Copy, PartialEq)]
enum RegVariant {
    Unsigned,
    Unscaled,
    ImmPost,
    ImmPre,
    RegOff,
}

/// GP classification for unsigned-offset and register-offset forms (these share
/// the same `(size, opc)` table and code set).
fn classify_gp(size: u32, opc: u32, variant: RegVariant) -> Option<LdStForm> {
    let reg = variant == RegVariant::RegOff;
    macro_rules! f {
        ($code:expr, $x:expr) => {
            Some(LdStForm {
                code: $code,
                is_fp: false,
                fp_code: 0,
                gp_x: $x,
                scale: size,
                is_prfm: false,
            })
        };
    }
    match (size, opc) {
        (0, 0b00) => f!(if reg { Code::StrbReg } else { Code::StrbImmUnsigned }, false),
        (0, 0b01) => f!(if reg { Code::LdrbReg } else { Code::LdrbImmUnsigned }, false),
        (0, 0b10) => f!(if reg { Code::LdrsbReg64 } else { Code::LdrsbImmUnsigned64 }, true),
        (0, 0b11) => f!(if reg { Code::LdrsbReg32 } else { Code::LdrsbImmUnsigned32 }, false),
        (1, 0b00) => f!(if reg { Code::StrhReg } else { Code::StrhImmUnsigned }, false),
        (1, 0b01) => f!(if reg { Code::LdrhReg } else { Code::LdrhImmUnsigned }, false),
        (1, 0b10) => f!(if reg { Code::LdrshReg64 } else { Code::LdrshImmUnsigned64 }, true),
        (1, 0b11) => f!(if reg { Code::LdrshReg32 } else { Code::LdrshImmUnsigned32 }, false),
        (2, 0b00) => f!(if reg { Code::StrReg32 } else { Code::StrImmUnsigned32 }, false),
        (2, 0b01) => f!(if reg { Code::LdrReg32 } else { Code::LdrImmUnsigned32 }, false),
        (2, 0b10) => f!(if reg { Code::LdrswReg } else { Code::LdrswImmUnsigned }, true),
        (3, 0b00) => f!(if reg { Code::StrReg64 } else { Code::StrImmUnsigned64 }, true),
        (3, 0b01) => f!(if reg { Code::LdrReg64 } else { Code::LdrImmUnsigned64 }, true),
        (3, 0b10) => {
            // PRFM.
            Some(LdStForm {
                code: if reg { Code::PrfmReg } else { Code::PrfmImmUnsigned },
                is_fp: false,
                fp_code: 0,
                gp_x: true,
                scale: 3,
                is_prfm: true,
            })
        }
        _ => None,
    }
}

/// GP classification for the unscaled (`LDUR`/`STUR`/`PRFUM`) forms.
fn classify_gp_unscaled(size: u32, opc: u32) -> Option<LdStForm> {
    macro_rules! f {
        ($code:expr, $x:expr) => {
            Some(LdStForm { code: $code, is_fp: false, fp_code: 0, gp_x: $x, scale: 0, is_prfm: false })
        };
    }
    match (size, opc) {
        (0, 0b00) => f!(Code::Sturb, false),
        (0, 0b01) => f!(Code::Ldurb, false),
        (0, 0b10) => f!(Code::Ldursb64, true),
        (0, 0b11) => f!(Code::Ldursb32, false),
        (1, 0b00) => f!(Code::Sturh, false),
        (1, 0b01) => f!(Code::Ldurh, false),
        (1, 0b10) => f!(Code::Ldursh64, true),
        (1, 0b11) => f!(Code::Ldursh32, false),
        (2, 0b00) => f!(Code::Stur32, false),
        (2, 0b01) => f!(Code::Ldur32, false),
        (2, 0b10) => f!(Code::Ldursw, true),
        (3, 0b00) => f!(Code::Stur64, true),
        (3, 0b01) => f!(Code::Ldur64, true),
        (3, 0b10) => Some(LdStForm {
            code: Code::Prfum,
            is_fp: false,
            fp_code: 0,
            gp_x: true,
            scale: 0,
            is_prfm: true,
        }),
        _ => None,
    }
}

/// GP classification for the immediate pre/post-index forms.
fn classify_gp_immidx(size: u32, opc: u32, variant: RegVariant) -> Option<LdStForm> {
    let pre = variant == RegVariant::ImmPre;
    macro_rules! f {
        ($post:expr, $prec:expr, $x:expr) => {
            Some(LdStForm {
                code: if pre { $prec } else { $post },
                is_fp: false,
                fp_code: 0,
                gp_x: $x,
                scale: 0,
                is_prfm: false,
            })
        };
    }
    match (size, opc) {
        (0, 0b00) => f!(Code::StrbImmPost, Code::StrbImmPre, false),
        (0, 0b01) => f!(Code::LdrbImmPost, Code::LdrbImmPre, false),
        (0, 0b10) => f!(Code::LdrsbImmPost64, Code::LdrsbImmPre64, true),
        (0, 0b11) => f!(Code::LdrsbImmPost32, Code::LdrsbImmPre32, false),
        (1, 0b00) => f!(Code::StrhImmPost, Code::StrhImmPre, false),
        (1, 0b01) => f!(Code::LdrhImmPost, Code::LdrhImmPre, false),
        (1, 0b10) => f!(Code::LdrshImmPost64, Code::LdrshImmPre64, true),
        (1, 0b11) => f!(Code::LdrshImmPost32, Code::LdrshImmPre32, false),
        (2, 0b00) => f!(Code::StrImmPost32, Code::StrImmPre32, false),
        (2, 0b01) => f!(Code::LdrImmPost32, Code::LdrImmPre32, false),
        (2, 0b10) => f!(Code::LdrswImmPost, Code::LdrswImmPre, true),
        (3, 0b00) => f!(Code::StrImmPost64, Code::StrImmPre64, true),
        (3, 0b01) => f!(Code::LdrImmPost64, Code::LdrImmPre64, true),
        _ => None,
    }
}

/// FP unscaled code for an access-size log2 `acc` (0..4) and load flag.
fn fp_unscaled_code(acc: u32, load: bool) -> Code {
    // There is one Ldur/Stur scalar-FP code per width; reuse the integer Ldur*
    // codes are GP-only, so SIMD&FP unscaled forms map onto the dedicated FP
    // unsigned codes is wrong — instead the corpus shows them as ldur/stur with
    // a B/H/S/D/Q register. We model them with the matching FP *immediate* code
    // but emit the LDUR/STUR mnemonic via set_mnemonic at the call site.
    match (acc, load) {
        (0, true) => Code::LdrFpImmUnsigned8,
        (1, true) => Code::LdrFpImmUnsigned16,
        (2, true) => Code::LdrFpImmUnsigned32,
        (3, true) => Code::LdrFpImmUnsigned64,
        (4, true) => Code::LdrFpImmUnsigned128,
        (0, false) => Code::StrFpImmUnsigned8,
        (1, false) => Code::StrFpImmUnsigned16,
        (2, false) => Code::StrFpImmUnsigned32,
        (3, false) => Code::StrFpImmUnsigned64,
        _ => Code::StrFpImmUnsigned128,
    }
}

/// FP immediate pre/post code (reuses the FP unsigned codes; mnemonic stays LDR/STR).
fn fp_immidx_code(acc: u32, load: bool, _variant: RegVariant) -> Code {
    fp_unscaled_code(acc, load)
}

// ---------------------------------------------------------------------------
// Load register (literal).
// ---------------------------------------------------------------------------

/// `LDR (literal)` / `LDRSW (literal)` / `PRFM (literal)` and the SIMD&FP literal
/// loads. Encoding: `opc(31:30) 011 V(26) 00 imm19 Rt`. Target is
/// `ip + SignExtend(imm19:00, 21)`.
#[inline]
fn decode_literal(word: u32, ip: u64, out: &mut Instruction) {
    let opc = bits(word, 30, 2);
    let v = bit(word, 26);
    let imm19 = bits(word, 5, 19);
    let rt = bits(word, 0, 5);
    let offset = sign_extend((imm19 as u64) << 2, 21);
    let target = ip.wrapping_add(offset as u64);

    if v == 1 {
        // SIMD&FP: opc 00->S, 01->D, 10->Q, 11 reserved.
        let (code, fp_code) = match opc {
            0b00 => (Code::LdrLitFp32, 2),
            0b01 => (Code::LdrLitFp64, 3),
            0b10 => (Code::LdrLitFp128, 4),
            _ => return,
        };
        out.set(code);
        out.push_operand(simd_op(fp_reg(fp_code, rt)));
        out.push_operand(Operand::Label(target));
        return;
    }

    match opc {
        0b00 => {
            out.set(Code::LdrLit32);
            out.push_operand(gp(false, RegWidth::W32, rt));
            out.push_operand(Operand::Label(target));
        }
        0b01 => {
            out.set(Code::LdrLit64);
            out.push_operand(gp(false, RegWidth::X64, rt));
            out.push_operand(Operand::Label(target));
        }
        0b10 => {
            out.set(Code::LdrswLit);
            out.push_operand(gp(false, RegWidth::X64, rt));
            out.push_operand(Operand::Label(target));
        }
        _ => {
            // PRFM (literal).
            out.set(Code::PrfmLit);
            out.push_operand(prefetch_op(rt));
            out.push_operand(Operand::Label(target));
        }
    }
}

// ---------------------------------------------------------------------------
// Load/store register: unsigned immediate offset.
// ---------------------------------------------------------------------------

/// `LDR/STR (immediate, unsigned offset)` and the byte/half/sign/FP variants.
/// Encoding: `size(31:30) 111 V(26) 01 opc(23:22) imm12 Rn Rt`. The 12-bit
/// immediate is zero-extended and scaled by the access size.
#[inline]
fn decode_reg_unsigned(word: u32, out: &mut Instruction) {
    let size = bits(word, 30, 2);
    let v = bit(word, 26);
    let opc = bits(word, 22, 2);
    let imm12 = bits(word, 10, 12);
    let rn = bits(word, 5, 5);
    let rt = bits(word, 0, 5);

    let form = match classify_reg(size, v, opc, RegVariant::Unsigned) {
        Some(f) => f,
        None => return,
    };
    let imm = (imm12 as i64) << form.scale;
    out.set(form.code);
    if form.is_fp {
        out.push_operand(simd_op(fp_reg(form.fp_code, rt)));
    } else {
        push_data_reg(out, &form, rt);
    }
    out.push_operand(mem_off(rn, imm));
}

// ---------------------------------------------------------------------------
// Load/store register: unscaled (LDUR/STUR/PRFUM).
// ---------------------------------------------------------------------------

/// `LDUR/STUR (unscaled)` and friends. Encoding:
/// `size 111 V 00 opc 0 imm9 00 Rn Rt`. `imm9` is signed, unscaled.
#[inline]
fn decode_reg_unscaled(word: u32, out: &mut Instruction) {
    let size = bits(word, 30, 2);
    let v = bit(word, 26);
    let opc = bits(word, 22, 2);
    let imm9 = bits(word, 12, 9);
    let rn = bits(word, 5, 5);
    let rt = bits(word, 0, 5);
    let imm = sign_extend(imm9 as u64, 9);

    let form = match classify_reg(size, v, opc, RegVariant::Unscaled) {
        Some(f) => f,
        None => return,
    };
    out.set(form.code);
    if form.is_fp {
        // SIMD&FP unscaled => LDUR/STUR mnemonic with the B/H/S/D/Q register.
        out.set_mnemonic(if (opc & 1) == 1 { Mnemonic::Ldur } else { Mnemonic::Stur });
        out.push_operand(simd_op(fp_reg(form.fp_code, rt)));
    } else {
        push_data_reg(out, &form, rt);
    }
    out.push_operand(mem_off(rn, imm));
}

// ---------------------------------------------------------------------------
// Load/store register: immediate post/pre-index.
// ---------------------------------------------------------------------------

#[inline]
fn decode_reg_immpost(word: u32, out: &mut Instruction) {
    decode_reg_immidx(word, RegVariant::ImmPost, out);
}

#[inline]
fn decode_reg_immpre(word: u32, out: &mut Instruction) {
    decode_reg_immidx(word, RegVariant::ImmPre, out);
}

/// Shared pre/post-index immediate decode. Encoding:
/// `size 111 V 00 opc 0 imm9 (01|11) Rn Rt`. `imm9` is signed, unscaled.
#[inline]
fn decode_reg_immidx(word: u32, variant: RegVariant, out: &mut Instruction) {
    let size = bits(word, 30, 2);
    let v = bit(word, 26);
    let opc = bits(word, 22, 2);
    let imm9 = bits(word, 12, 9);
    let rn = bits(word, 5, 5);
    let rt = bits(word, 0, 5);
    let imm = sign_extend(imm9 as u64, 9);

    let form = match classify_reg(size, v, opc, variant) {
        Some(f) => f,
        None => return,
    };
    out.set(form.code);
    if form.is_fp {
        let load = (opc & 1) == 1;
        // FP forms keep LDR/STR; the FP code carries the right view.
        out.set_mnemonic(if load { Mnemonic::Ldr } else { Mnemonic::Str });
        out.push_operand(simd_op(fp_reg(form.fp_code, rt)));
    } else {
        push_data_reg(out, &form, rt);
    }
    let mem = if variant == RegVariant::ImmPre {
        mem_pre(rn, imm)
    } else {
        mem_post(rn, imm)
    };
    out.push_operand(mem);
}

// ---------------------------------------------------------------------------
// Load/store register: unprivileged (LDTR/STTR).
// ---------------------------------------------------------------------------

/// `LDTR/STTR` (unprivileged). Encoding: `size 111 0 00 opc 0 imm9 10 Rn Rt`.
/// Signed, unscaled `imm9`. No SIMD&FP forms.
#[inline]
fn decode_reg_unpriv(word: u32, out: &mut Instruction) {
    if bit(word, 26) != 0 {
        return; // V must be 0
    }
    let size = bits(word, 30, 2);
    let opc = bits(word, 22, 2);
    let imm9 = bits(word, 12, 9);
    let rn = bits(word, 5, 5);
    let rt = bits(word, 0, 5);
    let imm = sign_extend(imm9 as u64, 9);

    macro_rules! emit {
        ($code:expr, $x:expr) => {{
            out.set($code);
            out.push_operand(gp(false, w_of($x as u32), rt));
            out.push_operand(mem_off(rn, imm));
            return;
        }};
    }
    match (size, opc) {
        (0, 0b00) => emit!(Code::Sttrb, 0),
        (0, 0b01) => emit!(Code::Ldtrb, 0),
        (0, 0b10) => emit!(Code::Ldtrsb64, 1),
        (0, 0b11) => emit!(Code::Ldtrsb32, 0),
        (1, 0b00) => emit!(Code::Sttrh, 0),
        (1, 0b01) => emit!(Code::Ldtrh, 0),
        (1, 0b10) => emit!(Code::Ldtrsh64, 1),
        (1, 0b11) => emit!(Code::Ldtrsh32, 0),
        (2, 0b00) => emit!(Code::Sttr32, 0),
        (2, 0b01) => emit!(Code::Ldtr32, 0),
        (2, 0b10) => emit!(Code::Ldtrsw, 1),
        (3, 0b00) => emit!(Code::Sttr64, 1),
        (3, 0b01) => emit!(Code::Ldtr64, 1),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Load/store register: register offset.
// ---------------------------------------------------------------------------

/// `LDR/STR (register)` and friends. Encoding:
/// `size 111 V 00 opc 1 Rm option(15:13) S(12) 10 Rn Rt`.
/// The `option` field is the extend type; the shift amount is `S ? scale : 0`.
#[inline]
fn decode_reg_offset(word: u32, features: FeatureSet, out: &mut Instruction) {
    let size = bits(word, 30, 2);
    let v = bit(word, 26);
    let opc = bits(word, 22, 2);
    let rm = bits(word, 16, 5);
    let option = bits(word, 13, 3);
    let s = bit(word, 12);
    let rn = bits(word, 5, 5);
    let rt = bits(word, 0, 5);

    // option<1> must be set (only uxtw/uxtx/sxtw/sxtx — i.e. 010/011/110/111).
    if (option & 0b010) == 0 {
        return;
    }

    // RPRFM (FEAT_RPRFM) carves out a sub-slot of the `PRFM (register offset)`
    // encoding (`size==11`, `V==0`, `opc==10`): it is selected when `Rt<4:3>==11`
    // (i.e. `Rt` in `24..=31`) — the `prfop` values the ordinary PRFM does not
    // allocate. Plain PRFM keeps `Rt<4:3>!=11`. The op spelling is a 6-bit
    // `rprfop` `imm6 = option<2> : option<0> : S : Rt<2:0>`; the index is `Xm`;
    // the base carries no displacement.
    if size == 0b11 && v == 0 && opc == 0b10 && (rt >> 3) == 0b11 {
        if !features.has(Feature::Rprfm) {
            return;
        }
        let imm6 = (((option >> 2) & 1) << 5) | ((option & 1) << 4) | (s << 3) | (rt & 0b111);
        out.set(Code::RprfmReg);
        out.push_operand(rprfop_op(imm6));
        out.push_operand(gp(false, RegWidth::X64, rm));
        out.push_operand(mem_off(rn, 0));
        return;
    }

    let form = match classify_reg(size, v, opc, RegVariant::RegOff) {
        Some(f) => f,
        None => return,
    };
    out.set(form.code);

    if form.is_fp {
        // Refine the B (acc==0) register-offset form, which reuses LdrFpReg32 as
        // the code carrier; emit LDR/STR with the byte register.
        let load = (opc & 1) == 1;
        out.set_mnemonic(if load { Mnemonic::Ldr } else { Mnemonic::Str });
        out.push_operand(simd_op(fp_reg(form.fp_code, rt)));
    } else if form.is_prfm {
        out.push_operand(prefetch_op(rt));
    } else {
        out.push_operand(gp(false, w_of(form.gp_x as u32), rt));
    }

    out.push_operand(mem_ext(rn, rm, option, s, form.scale));
}

/// Build the register-offset memory operand, applying Binary Ninja's rendering
/// rule: the index uses the `W` view for `uxtw`/`sxtw` and the `X` view
/// otherwise; the shift amount is `S ? scale : 0` and is suppressed entirely for
/// the LSL/uxtx case with `S==0` (encoded here as the `Uxtx` extend with a
/// sentinel that the formatter elides).
#[inline]
fn mem_ext(rn: u32, rm: u32, option: u32, s: u32, scale: u32) -> Operand {
    let extend = ExtendType::from_bits(option as u8);
    // Index register width: W for the `*xtw` extends, X otherwise.
    let idx_w = matches!(extend, ExtendType::Uxtw | ExtendType::Sxtw);
    let index = gp_register(
        false,
        if idx_w { RegWidth::W32 } else { RegWidth::X64 },
        (rm & 0x1f) as u8,
    );
    let base = gp_register(true, RegWidth::X64, (rn & 0x1f) as u8);
    // The shift amount is the access-size scale when S is set, else zero.
    let amt = if s == 1 { scale as u8 } else { 0 };
    Operand::MemExt {
        base,
        index,
        extend,
        // Encode "S" by always using amt; the formatter is taught to show the
        // amount when present and to drop the uxtx keyword + amount when amt==0.
        shift: pack_shift(option, s, amt),
    }
}

/// Pack the (option, S, amount) into the `shift` byte for [`Operand::MemExt`].
/// Layout: bit7 = "force show amount" (S==1), bits<6:0> = amount. The formatter
/// uses bit7 to decide whether to print `#0`.
#[inline]
fn pack_shift(_option: u32, s: u32, amt: u8) -> u8 {
    ((s as u8) << 7) | (amt & 0x7f)
}

// ---------------------------------------------------------------------------
// Load/store register pair (and SIMD&FP pair, LDNP/STNP).
// ---------------------------------------------------------------------------

/// Load/store pair. Encoding:
/// `opc(31:30) 101 V(26) idx(24:23) L(22) imm7 Rt2 Rn Rt`, where `idx` selects
/// no-alloc(00)/post(01)/offset(10)/pre(11). `imm7` is signed, scaled.
#[inline]
fn decode_pair(word: u32, features: FeatureSet, out: &mut Instruction) {
    let opc = bits(word, 30, 2);
    let v = bit(word, 26);
    let idx = bits(word, 23, 2);
    let l = bit(word, 22);
    let imm7 = bits(word, 15, 7);
    let rt2 = bits(word, 10, 5);
    let rn = bits(word, 5, 5);
    let rt = bits(word, 0, 5);

    // opc==11, V==0 is the FEAT_THE unprivileged translation-enhanced pair
    // (LDTP/STTP for idx 01/10/11, LDTNP/STTNP for idx 00). opc==11 with V==1 is
    // the FEAT_LSUI quadword unprivileged translation-enhanced pair (same idx
    // selector, but `Q` data registers, so `imm7` is scaled by 4).
    if opc == 0b11 {
        if v == 0 {
            return decode_the_pair(idx, l, imm7, rt2, rn, rt, features, out);
        }
        return decode_the_pair_q(idx, l, imm7, rt2, rn, rt, features, out);
    }

    // STGP is the special opc=01, V=0 store-tag-and-pair (MTE). It is handled in
    // the tag path? No — STGP shares this pair encoding with idx selecting the
    // addressing mode. Detect it: opc=01, V=0, L=0.
    if opc == 0b01 && v == 0 && l == 0 {
        if !features.has(Feature::Mte) {
            return;
        }
        return emit_stgp(idx, imm7, rt2, rn, rt, out);
    }

    let is_np = idx == 0b00;
    let load = l == 1;

    if v == 1 {
        // SIMD&FP pair: opc 00->S(scale2), 01->D(scale3), 10->Q(scale4).
        let (fp_code, scale) = match opc {
            0b00 => (2u32, 2u32),
            0b01 => (3u32, 3u32),
            0b10 => (4u32, 4u32),
            _ => return,
        };
        // Code carrier: LDP/STP FP per width (the NP forms reuse it but render
        // the LDNP/STNP mnemonic).
        let code = match (load, fp_code) {
            (true, 2) => Code::LdpFp32,
            (true, 3) => Code::LdpFp64,
            (true, 4) => Code::LdpFp128,
            (false, 2) => Code::StpFp32,
            (false, 3) => Code::StpFp64,
            _ => Code::StpFp128,
        };
        let imm = sign_extend(imm7 as u64, 7) << scale;
        out.set(code);
        if is_np {
            out.set_mnemonic(if load { Mnemonic::Ldnp } else { Mnemonic::Stnp });
        }
        out.push_operand(simd_op(fp_reg(fp_code, rt)));
        out.push_operand(simd_op(fp_reg(fp_code, rt2)));
        out.push_operand(pair_mem(idx, rn, imm));
        return;
    }

    // GP pair: opc 00->W(scale2), 01->LDPSW(X,scale2,load-only), 10->X(scale3).
    let (gp_x, scale, code) = match opc {
        0b00 => (false, 2u32, pair_code_gp(idx, l, GpPairKind::W)),
        0b01 => {
            // LDPSW (load only). The store form (L==0) was STGP, handled above.
            if !load {
                return;
            }
            (true, 2u32, pair_code_gp(idx, l, GpPairKind::Sw))
        }
        0b10 => (true, 3u32, pair_code_gp(idx, l, GpPairKind::X)),
        _ => return,
    };
    let code = match code {
        Some(c) => c,
        None => return,
    };
    let imm = sign_extend(imm7 as u64, 7) << scale;
    out.set(code);
    let w = w_of(gp_x as u32);
    out.push_operand(gp(false, w, rt));
    out.push_operand(gp(false, w, rt2));
    out.push_operand(pair_mem(idx, rn, imm));
}

/// FEAT_THE unprivileged translation-enhanced load/store pair. Encoding:
/// `11 101 0 idx(24:23) L(22) imm7 Rt2 Rn Rt` with `V==0`. `idx` selects
/// non-temporal(00)/post(01)/offset(10)/pre(11); the non-temporal forms render
/// as `LDTNP`/`STTNP`, the rest as `LDTP`/`STTP`. Both data registers are 64-bit
/// `X` registers, so `imm7` is signed and scaled by 8.
#[inline]
#[allow(clippy::too_many_arguments)]
fn decode_the_pair(
    idx: u32,
    l: u32,
    imm7: u32,
    rt2: u32,
    rn: u32,
    rt: u32,
    features: FeatureSet,
    out: &mut Instruction,
) {
    if !features.has(Feature::The) {
        return;
    }
    let load = l == 1;
    let code = match (idx, load) {
        (0b00, true) => Code::Ldtnp,
        (0b00, false) => Code::Sttnp,
        (0b01, true) => Code::LdtpPost,
        (0b01, false) => Code::SttpPost,
        (0b10, true) => Code::LdtpOff,
        (0b10, false) => Code::SttpOff,
        (0b11, true) => Code::LdtpPre,
        (0b11, false) => Code::SttpPre,
        _ => return,
    };
    let imm = sign_extend(imm7 as u64, 7) << 3;
    out.set(code);
    out.push_operand(gp(false, RegWidth::X64, rt));
    out.push_operand(gp(false, RegWidth::X64, rt2));
    out.push_operand(pair_mem(idx, rn, imm));
}

/// FEAT_LSUI quadword unprivileged translation-enhanced load/store pair.
/// Encoding: `11 101 1 idx(24:23) L(22) imm7 Rt2 Rn Rt` (i.e. the [`decode_the_pair`]
/// layout with `V==1`). The `idx` selector matches the THE pair — non-temporal(00)
/// renders as `LDTNP`/`STTNP`, post(01)/offset(10)/pre(11) as `LDTP`/`STTP` — but
/// both data registers are 128-bit `Q` registers, so `imm7` is signed and scaled
/// by 4.
#[inline]
#[allow(clippy::too_many_arguments)]
fn decode_the_pair_q(
    idx: u32,
    l: u32,
    imm7: u32,
    rt2: u32,
    rn: u32,
    rt: u32,
    features: FeatureSet,
    out: &mut Instruction,
) {
    if !features.has(Feature::Lsui) {
        return;
    }
    let load = l == 1;
    let code = match (idx, load) {
        (0b00, true) => Code::Ldtnpq,
        (0b00, false) => Code::Sttnpq,
        (0b01, true) => Code::LdtpqPost,
        (0b01, false) => Code::SttpqPost,
        (0b10, true) => Code::LdtpqOff,
        (0b10, false) => Code::SttpqOff,
        (0b11, true) => Code::LdtpqPre,
        (0b11, false) => Code::SttpqPre,
        _ => return,
    };
    let imm = sign_extend(imm7 as u64, 7) << 4;
    out.set(code);
    out.push_operand(simd_op(fp_reg(4, rt)));
    out.push_operand(simd_op(fp_reg(4, rt2)));
    out.push_operand(pair_mem(idx, rn, imm));
}

/// The pair memory operand for the 2-bit `idx` addressing selector.
#[inline]
fn pair_mem(idx: u32, rn: u32, imm: i64) -> Operand {
    match idx {
        0b01 => mem_post(rn, imm), // post-index
        0b11 => mem_pre(rn, imm),  // pre-index
        // 00 (no-alloc) and 10 (signed offset) both render as `[Xn, #imm]`.
        _ => mem_off(rn, imm),
    }
}

enum GpPairKind {
    W,
    X,
    Sw,
}

/// GP pair code from `(idx, L, kind)`. `idx` 00=NP, 01=post, 10=offset, 11=pre.
fn pair_code_gp(idx: u32, l: u32, kind: GpPairKind) -> Option<Code> {
    let load = l == 1;
    Some(match kind {
        GpPairKind::W => match (idx, load) {
            (0b00, true) => Code::Ldnp32,
            (0b00, false) => Code::Stnp32,
            (0b01, true) => Code::Ldp32Post,
            (0b01, false) => Code::Stp32Post,
            (0b10, true) => Code::Ldp32,
            (0b10, false) => Code::Stp32,
            (0b11, true) => Code::Ldp32Pre,
            (0b11, false) => Code::Stp32Pre,
            _ => return None,
        },
        GpPairKind::X => match (idx, load) {
            (0b00, true) => Code::Ldnp64,
            (0b00, false) => Code::Stnp64,
            (0b01, true) => Code::Ldp64Post,
            (0b01, false) => Code::Stp64Post,
            (0b10, true) => Code::Ldp64,
            (0b10, false) => Code::Stp64,
            (0b11, true) => Code::LdpPre64,
            (0b11, false) => Code::Stp64Pre,
            _ => return None,
        },
        GpPairKind::Sw => match idx {
            // LDPSW only (load). NP form (idx 00) is not allocated for LDPSW.
            0b01 => Code::LdpswPost,
            0b10 => Code::Ldpsw,
            0b11 => Code::LdpswPre,
            _ => return None,
        },
    })
}


/// Emit `STGP <Xt1>, <Xt2>, [...]` (store tag and pair of registers, MTE).
/// `idx` selects the addressing mode (post/offset/pre); scale is 16 bytes (×16
/// tag-granule), i.e. shift 4.
#[inline]
fn emit_stgp(idx: u32, imm7: u32, rt2: u32, rn: u32, rt: u32, out: &mut Instruction) {
    let code = match idx {
        0b01 => Code::StgpPost,
        0b10 => Code::StgpOff,
        0b11 => Code::StgpPre,
        _ => return,
    };
    let imm = sign_extend(imm7 as u64, 7) << 4;
    out.set(code);
    out.push_operand(gp(false, RegWidth::X64, rt));
    out.push_operand(gp(false, RegWidth::X64, rt2));
    out.push_operand(pair_mem(idx, rn, imm));
}

// ---------------------------------------------------------------------------
// Load/store exclusive and ordered.
// ---------------------------------------------------------------------------

/// Exclusive / ordered / CAS / CASP, all under `sz 001000 o2 L o1 Rs o0 Rt2 Rn Rt`.
#[inline]
fn decode_exclusive(word: u32, features: FeatureSet, out: &mut Instruction) {
    let sz = bits(word, 30, 2);
    let o2 = bit(word, 23);
    let l = bit(word, 22);
    let o1 = bit(word, 21);
    let rs = bits(word, 16, 5);
    let o0 = bit(word, 15);
    let rt2 = bits(word, 10, 5);
    let rn = bits(word, 5, 5);
    let rt = bits(word, 0, 5);

    if o2 == 0 {
        if o1 == 0 {
            // Single exclusive: LDXR/STXR/LDAXR/STLXR (+B/H by sz).
            decode_excl_single(sz, l, o0, rs, rn, rt, out);
        } else {
            // Pair exclusive (sz<1>==1) OR CASP (sz<1>==0, LSE).
            if bit(sz, 1) == 1 {
                decode_excl_pair(sz, l, o0, rs, rt2, rn, rt, out);
            } else if rt2 == 0b11111 {
                // CASP fixes the Rt2 field (bits<14:10>) as all-ones; any other
                // value is UNDEFINED (ARM ARM), so leave it Invalid.
                decode_casp(sz, l, o0, rs, rn, rt, features, out);
            }
        }
    } else {
        // o2 == 1.
        if o1 == 1 {
            // CAS (LSE). The Rt2 field (bits<14:10>) is fixed all-ones; any other
            // value is UNDEFINED.
            if rt2 == 0b11111 {
                decode_cas(sz, l, o0, rs, rn, rt, features, out);
            }
        } else {
            // Ordered: LDAR/STLR/LDLAR/STLLR (+B/H by sz).
            decode_ordered(sz, l, o0, rn, rt, out);
        }
    }
}

/// Single-register exclusive. `[Xn|SP]` only (offset is `#0`, elided).
#[inline]
fn decode_excl_single(sz: u32, l: u32, o0: u32, rs: u32, rn: u32, rt: u32, out: &mut Instruction) {
    let load = l == 1;
    let acquire = o0 == 1;
    // (load, acquire) -> base mnemonic family; sz -> width / byte / half.
    let code = match (sz, load, acquire) {
        (0, true, false) => Code::Ldxrb,
        (0, true, true) => Code::Ldaxrb,
        (0, false, false) => Code::Stxrb,
        (0, false, true) => Code::Stlxrb,
        (1, true, false) => Code::Ldxrh,
        (1, true, true) => Code::Ldaxrh,
        (1, false, false) => Code::Stxrh,
        (1, false, true) => Code::Stlxrh,
        (2, true, false) => Code::Ldxr32,
        (2, true, true) => Code::Ldaxr32,
        (2, false, false) => Code::Stxr32,
        (2, false, true) => Code::Stlxr32,
        (3, true, false) => Code::Ldxr64,
        (3, true, true) => Code::Ldaxr64,
        (3, false, false) => Code::Stxr64,
        _ => Code::Stlxr64,
    };
    out.set(code);
    let rt_x = sz == 3;
    if load {
        out.push_operand(gp(false, w_of(rt_x as u32), rt));
    } else {
        // Store: Ws status register first, then Wt/Xt data.
        out.push_operand(gp(false, RegWidth::W32, rs));
        out.push_operand(gp(false, w_of(rt_x as u32), rt));
    }
    out.push_operand(mem_off(rn, 0));
}

/// Pair exclusive (LDXP/STXP/LDAXP/STLXP). 32-bit when sz==2, 64-bit when sz==3.
// Raw ARM ARM bitfields; grouping into a struct would obscure the 1:1 mapping.
#[allow(clippy::too_many_arguments)]
#[inline]
fn decode_excl_pair(sz: u32, l: u32, o0: u32, rs: u32, rt2: u32, rn: u32, rt: u32, out: &mut Instruction) {
    let load = l == 1;
    let acquire = o0 == 1;
    let x = sz == 3;
    let code = match (load, acquire, x) {
        (true, false, false) => Code::Ldxp32,
        (true, false, true) => Code::Ldxp64,
        (true, true, false) => Code::Ldaxp32,
        (true, true, true) => Code::Ldaxp64,
        (false, false, false) => Code::Stxp32,
        (false, false, true) => Code::Stxp64,
        (false, true, false) => Code::Stlxp32,
        _ => Code::Stlxp64,
    };
    out.set(code);
    let w = w_of(x as u32);
    if load {
        out.push_operand(gp(false, w, rt));
        out.push_operand(gp(false, w, rt2));
    } else {
        out.push_operand(gp(false, RegWidth::W32, rs));
        out.push_operand(gp(false, w, rt));
        out.push_operand(gp(false, w, rt2));
    }
    out.push_operand(mem_off(rn, 0));
}

/// Ordered load-acquire / store-release: LDAR/STLR/LDLAR/STLLR (+ B/H).
#[inline]
fn decode_ordered(sz: u32, l: u32, o0: u32, rn: u32, rt: u32, out: &mut Instruction) {
    let load = l == 1;
    // o0==1 -> LDAR/STLR (acquire-release, RCsc); o0==0 -> LDLAR/STLLR (LOR).
    let code = match (sz, load, o0) {
        (0, true, 1) => Code::Ldarb,
        (0, false, 1) => Code::Stlrb,
        (0, true, 0) => Code::Ldlarb,
        (0, false, 0) => Code::Stllrb,
        (1, true, 1) => Code::Ldarh,
        (1, false, 1) => Code::Stlrh,
        (1, true, 0) => Code::Ldlarh,
        (1, false, 0) => Code::Stllrh,
        (2, true, 1) => Code::Ldar32,
        (2, false, 1) => Code::Stlr32,
        (2, true, 0) => Code::Ldlar32,
        (2, false, 0) => Code::Stllr32,
        (3, true, 1) => Code::Ldar64,
        (3, false, 1) => Code::Stlr64,
        (3, true, 0) => Code::Ldlar64,
        _ => Code::Stllr64,
    };
    out.set(code);
    let x = sz == 3;
    out.push_operand(gp(false, w_of(x as u32), rt));
    out.push_operand(mem_off(rn, 0));
}

/// `CAS{A}{L}{B|H}` compare-and-swap (LSE). `Rs` is the compare reg, `Rt` data.
// Raw ARM ARM bitfields; grouping into a struct would obscure the 1:1 mapping.
#[allow(clippy::too_many_arguments)]
#[inline]
fn decode_cas(sz: u32, l: u32, o0: u32, rs: u32, rn: u32, rt: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Lse) {
        return;
    }
    // acquire = L (bit22), release = o0 (bit15).
    let code = match (sz, l, o0) {
        (0, 0, 0) => Code::Casb,
        (0, 1, 0) => Code::Casab,
        (0, 0, 1) => Code::Caslb,
        (0, 1, 1) => Code::Casalb,
        (1, 0, 0) => Code::Cash,
        (1, 1, 0) => Code::Casah,
        (1, 0, 1) => Code::Caslh,
        (1, 1, 1) => Code::Casalh,
        (2, 0, 0) => Code::Cas32,
        (2, 1, 0) => Code::Casa32,
        (2, 0, 1) => Code::Casl32,
        (2, 1, 1) => Code::Casal32,
        (3, 0, 0) => Code::Cas64,
        (3, 1, 0) => Code::Casa64,
        (3, 0, 1) => Code::Casl64,
        _ => Code::Casal64,
    };
    out.set(code);
    let x = sz == 3;
    let w = w_of(x as u32);
    out.push_operand(gp(false, w, rs));
    out.push_operand(gp(false, w, rt));
    out.push_operand(mem_off(rn, 0));
}

/// `CASP{A}{L}` compare-and-swap pair (LSE). Even-numbered register pairs.
// Raw ARM ARM bitfields; grouping into a struct would obscure the 1:1 mapping.
#[allow(clippy::too_many_arguments)]
#[inline]
fn decode_casp(sz: u32, l: u32, o0: u32, rs: u32, rn: u32, rt: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Lse) {
        return;
    }
    // CASP operates on consecutive even-numbered register pairs: `Rs` and `Rt`
    // must both be even, otherwise the encoding is UNDEFINED (ARM ARM). Single
    // CAS has no such constraint, so this check lives only on the pair path.
    if (rs & 1) != 0 || (rt & 1) != 0 {
        return;
    }
    // sz here is 0 (32-bit) or 1 (64-bit); top bit was checked 0 by caller.
    let x = sz == 1;
    let code = match (x, l, o0) {
        (false, 0, 0) => Code::Casp32,
        (false, 1, 0) => Code::Caspa32,
        (false, 0, 1) => Code::Caspl32,
        (false, 1, 1) => Code::Caspal32,
        (true, 0, 0) => Code::Casp64,
        (true, 1, 0) => Code::Caspa64,
        (true, 0, 1) => Code::Caspl64,
        _ => Code::Caspal64,
    };
    out.set(code);
    let w = w_of(x as u32);
    // Pair: Rs, Rs+1, Rt, Rt+1, [Xn].
    out.push_operand(gp(false, w, rs));
    out.push_operand(gp(false, w, rs + 1));
    out.push_operand(gp(false, w, rt));
    out.push_operand(gp(false, w, rt + 1));
    out.push_operand(mem_off(rn, 0));
}

// ---------------------------------------------------------------------------
// FEAT_LSUI unprivileged atomics.
// ---------------------------------------------------------------------------

/// FEAT_LSUI unprivileged atomics, all under `sz 001001 o2 L o1 Rs o0 Rt2 Rn Rt`
/// (the exclusive/CAS layout with the group's low bit `word<24>` set). `o2`
/// selects between unprivileged load/store-exclusive (`o2==0`) and unprivileged
/// compare-and-swap (`o2==1`); `o1` is always `0` (no unprivileged pair-exclusive
/// form exists).
#[inline]
fn decode_lsui(word: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Lsui) {
        return;
    }
    let sz = bits(word, 30, 2);
    let o2 = bit(word, 23);
    let l = bit(word, 22);
    let o1 = bit(word, 21);
    let rs = bits(word, 16, 5);
    let o0 = bit(word, 15);
    let rn = bits(word, 5, 5);
    let rt = bits(word, 0, 5);

    // o1 is reserved-0 for every LSUI form (there is no unprivileged
    // exclusive-pair); any other value is UNDEFINED.
    if o1 != 0 {
        return;
    }

    if o2 == 0 {
        // Unprivileged load/store-exclusive register (W/X only).
        decode_lsui_excl_single(sz, l, o0, rs, rn, rt, out);
    } else {
        // Unprivileged compare-and-swap. Unlike the standard CAS/CASP class, the
        // LSUI forms treat the Rt2 field (bits<14:10>) as should-be-one but
        // IGNORED on decode (LLVM flags non-all-ones as "potentially undefined"
        // yet still decodes it), so it is not constrained here.
        // sz<1>==1 -> single CAS (CAST...); sz<1>==0 -> pair CASP (CASPT...).
        if bit(sz, 1) == 1 {
            decode_lsui_cas(sz, l, o0, rs, rn, rt, out);
        } else {
            decode_lsui_casp(sz, l, o0, rs, rn, rt, out);
        }
    }
}

/// Unprivileged single-register exclusive (LDTXR/LDATXR/STTXR/STLTXR). Only the
/// W (`sz==2`) and X (`sz==3`) forms exist — byte/half are not defined.
#[inline]
fn decode_lsui_excl_single(
    sz: u32,
    l: u32,
    o0: u32,
    rs: u32,
    rn: u32,
    rt: u32,
    out: &mut Instruction,
) {
    let load = l == 1;
    let acquire = o0 == 1;
    let code = match (sz, load, acquire) {
        (2, true, false) => Code::Ldtxr32,
        (2, true, true) => Code::Ldatxr32,
        (2, false, false) => Code::Sttxr32,
        (2, false, true) => Code::Stltxr32,
        (3, true, false) => Code::Ldtxr64,
        (3, true, true) => Code::Ldatxr64,
        (3, false, false) => Code::Sttxr64,
        (3, false, true) => Code::Stltxr64,
        _ => return, // byte/half (sz 0/1) are not allocated for LSUI.
    };
    out.set(code);
    let rt_x = sz == 3;
    if load {
        out.push_operand(gp(false, w_of(rt_x as u32), rt));
    } else {
        // Store: Ws status register first, then Wt/Xt data.
        out.push_operand(gp(false, RegWidth::W32, rs));
        out.push_operand(gp(false, w_of(rt_x as u32), rt));
    }
    out.push_operand(mem_off(rn, 0));
}

/// Unprivileged compare-and-swap (CAST/CASAT/CASLT/CASALT). Only the 64-bit form
/// (`sz==3`) is defined; the 32-bit slot (`sz==2`) is UNDEFINED.
#[inline]
fn decode_lsui_cas(sz: u32, l: u32, o0: u32, rs: u32, rn: u32, rt: u32, out: &mut Instruction) {
    if sz != 3 {
        return;
    }
    let code = match (l, o0) {
        (0, 0) => Code::Cast64,
        (1, 0) => Code::Casat64,
        (0, 1) => Code::Caslt64,
        _ => Code::Casalt64,
    };
    out.set(code);
    out.push_operand(gp(false, RegWidth::X64, rs));
    out.push_operand(gp(false, RegWidth::X64, rt));
    out.push_operand(mem_off(rn, 0));
}

/// Unprivileged compare-and-swap pair (CASPT/CASPAT/CASPLT/CASPALT). Only the
/// 64-bit pair (`sz==1`) is defined; the 32-bit slot (`sz==0`) is UNDEFINED.
/// `Rs`/`Rt` must be even (the consecutive odd register is implied).
#[inline]
fn decode_lsui_casp(sz: u32, l: u32, o0: u32, rs: u32, rn: u32, rt: u32, out: &mut Instruction) {
    if sz != 1 {
        return;
    }
    if (rs & 1) != 0 || (rt & 1) != 0 {
        return;
    }
    let code = match (l, o0) {
        (0, 0) => Code::Caspt64,
        (1, 0) => Code::Caspat64,
        (0, 1) => Code::Casplt64,
        _ => Code::Caspalt64,
    };
    out.set(code);
    out.push_operand(gp(false, RegWidth::X64, rs));
    out.push_operand(gp(false, RegWidth::X64, rs + 1));
    out.push_operand(gp(false, RegWidth::X64, rt));
    out.push_operand(gp(false, RegWidth::X64, rt + 1));
    out.push_operand(mem_off(rn, 0));
}

// ---------------------------------------------------------------------------
// Atomic memory operations (LSE): LDADD / LDCLR / ... / SWP and ST* aliases.
// ---------------------------------------------------------------------------

/// LSE atomic memory ops. Encoding:
/// `size 111 V 00 A R 1 Rs o3 opc 00 Rn Rt`, with `A`=bit23, `R`=bit22,
/// `o3`=bit15, `opc`=bits<14:12>. `o3:opc` selects the operation; `A:R` selects
/// the ordering. `Rt==31` with `A==0` renders the `ST<op>` alias.
#[inline]
fn decode_atomic(word: u32, features: FeatureSet, out: &mut Instruction) {
    if bit(word, 26) != 0 {
        // V==1 is the FEAT_LSFE atomic floating-point in-memory family
        // (`LDF*`/`STF*`/`LDBF*`/`STBF*`), which shares this LSE atomic major.
        return decode_lsfe(word, features, out);
    }
    let size = bits(word, 30, 2);
    let a = bit(word, 23);
    let r = bit(word, 22);
    let rs = bits(word, 16, 5);
    let o3 = bit(word, 15);
    let opc = bits(word, 12, 3);
    let rn = bits(word, 5, 5);
    let rt = bits(word, 0, 5);

    // LDAPR/LDAPRB/LDAPRH (FEAT_LRCPC, not LSE): o3==1, opc==100, A==1, R==0,
    // Rs==0b11111. These load-acquire-RCpc forms are base-decodable.
    if o3 == 1 && opc == 0b100 && a == 1 && r == 0 && rs == 0b11111 {
        let code = match size {
            0 => Code::Ldaprb,
            1 => Code::Ldaprh,
            2 => Code::Ldapr32,
            _ => Code::Ldapr64,
        };
        out.set(code);
        out.push_operand(gp(false, w_of((size == 3) as u32), rt));
        out.push_operand(mem_off(rn, 0));
        return;
    }

    // FEAT_LS64 single-copy atomic 64-byte ops share this encoding class with
    // size==11, A==0, R==0, o3==1 (b11:10==00 enforced by the caller). They are
    // distinguished by opc and Rs:
    //   LD64B   opc=101, Rs=11111  -> LD64B   <Xt>, [<Xn|SP>]
    //   ST64B   opc=001, Rs=11111  -> ST64B   <Xt>, [<Xn|SP>]
    //   ST64BV  opc=011            -> ST64BV  <Xs>, <Xt>, [<Xn|SP>]
    //   ST64BV0 opc=010            -> ST64BV0 <Xs>, <Xt>, [<Xn|SP>]
    if size == 3 && o3 == 1 && a == 0 && r == 0 {
        match (opc, rs) {
            (0b101, 0b11111) | (0b001, 0b11111) => {
                if !features.has(Feature::Ls64) {
                    return;
                }
                out.set(if opc == 0b101 { Code::Ld64b } else { Code::St64b });
                out.push_operand(gp(false, RegWidth::X64, rt));
                out.push_operand(mem_off(rn, 0));
                return;
            }
            (0b011, _) | (0b010, _) => {
                if !features.has(Feature::Ls64) {
                    return;
                }
                out.set(if opc == 0b011 { Code::St64bv } else { Code::St64bv0 });
                out.push_operand(gp(false, RegWidth::X64, rs));
                out.push_operand(gp(false, RegWidth::X64, rt));
                out.push_operand(mem_off(rn, 0));
                return;
            }
            _ => {}
        }
    }

    // FEAT_THE single-register read-check-write RMW ops share this LSE atomic
    // major (`o3==1`) but use `opc` 1/2/3 (SWP is `opc==0`). They are 64-bit only,
    // with size 0 -> RCW* and size 1 -> RCWS* (the PSTATE.C check variant). Sizes
    // 2/3 are reserved here (and size 3 is the LS64 region handled above), so only
    // sizes 0/1 decode. Operand order is `Rs, Rt, [Xn|SP]` (xzr permitted).
    if o3 == 1 && opc != 0b000 && size <= 1 {
        let ord = ((a << 1) | r) as usize;
        let code = match (opc, size) {
            (0b001, 0) => [Code::Rcwclr, Code::Rcwclrl, Code::Rcwclra, Code::Rcwclral][ord],
            (0b010, 0) => [Code::Rcwswp, Code::Rcwswpl, Code::Rcwswpa, Code::Rcwswpal][ord],
            (0b011, 0) => [Code::Rcwset, Code::Rcwsetl, Code::Rcwseta, Code::Rcwsetal][ord],
            (0b001, _) => [Code::Rcwsclr, Code::Rcwsclrl, Code::Rcwsclra, Code::Rcwsclral][ord],
            (0b010, _) => [Code::Rcwsswp, Code::Rcwsswpl, Code::Rcwsswpa, Code::Rcwsswpal][ord],
            (0b011, _) => [Code::Rcwsset, Code::Rcwssetl, Code::Rcwsseta, Code::Rcwssetal][ord],
            _ => return, // opc 4..7 unallocated.
        };
        if !features.has(Feature::The) {
            return;
        }
        out.set(code);
        out.push_operand(gp(false, RegWidth::X64, rs));
        out.push_operand(gp(false, RegWidth::X64, rt));
        out.push_operand(mem_off(rn, 0));
        return;
    }

    if !features.has(Feature::Lse) {
        return;
    }

    // SWP has o3==1, opc==000. The arithmetic/logical ops have o3==0.
    if o3 == 1 {
        if opc != 0b000 {
            return;
        }
        return emit_swp(size, a, r, rs, rn, rt, out);
    }
    // o3 == 0: LDADD/LDCLR/LDEOR/LDSET/LDSMAX/LDSMIN/LDUMAX/LDUMIN by opc.
    let op = match opc {
        0b000 => AtomicOp::Add,
        0b001 => AtomicOp::Clr,
        0b010 => AtomicOp::Eor,
        0b011 => AtomicOp::Set,
        0b100 => AtomicOp::Smax,
        0b101 => AtomicOp::Smin,
        0b110 => AtomicOp::Umax,
        0b111 => AtomicOp::Umin,
        _ => return,
    };
    emit_atomic_rmw(size, a, r, op, rs, rn, rt, out);
}

// ---------------------------------------------------------------------------
// FEAT_LSFE atomic floating-point in-memory ops (LDF*/STF* + BF16 LDBF*/STBF*).
// ---------------------------------------------------------------------------

/// FEAT_LSFE atomic float read-modify-write. Encoding (the `V==1` sibling of the
/// integer LSE atomic major):
/// `size 111 1 00 A R 1 Rs o3 opc 00 Rn Rt`, with `A`=bit23, `R`=bit22,
/// `o3`=bit15, `opc`=bits<14:12>. `opc` selects the op (`000`=FADD, `100`=FMAX,
/// `101`=FMIN, `110`=FMAXNM, `111`=FMINNM); `A:R` the ordering; `o3` the load
/// (`0`, `LDF<op> <V>s,<V>t,[Xn]`) vs store (`1`, `STF<op> <V>s,[Xn]`, which
/// requires `Rt==31` and `A==0`). `size` selects the data type: `00`=BF16
/// (`LDBF*`/`STBF*`, rendered with an `H` register), `01`=H, `10`=S, `11`=D.
#[inline]
fn decode_lsfe(word: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Lsfe) {
        return;
    }
    let size = bits(word, 30, 2);
    let a = bit(word, 23);
    let r = bit(word, 22);
    let rs = bits(word, 16, 5);
    let o3 = bit(word, 15);
    let opc = bits(word, 12, 3);
    let rn = bits(word, 5, 5);
    let rt = bits(word, 0, 5);

    // opc -> op index (FADD/FMAX/FMIN/FMAXNM/FMINNM); other values unallocated.
    let op_idx = match opc {
        0b000 => 0,
        0b100 => 1,
        0b101 => 2,
        0b110 => 3,
        0b111 => 4,
        _ => return,
    };
    let is_bf = size == 0;
    // Register class: BF16 and FP16 both use the `H` view; S for size==10, D for 11.
    let sc = if size == 0 { 1 } else { size };

    if o3 == 0 {
        // LDF<op> / LDBF<op>: <V>s, <V>t, [Xn|SP]. ordering 0..=3 from (A,R).
        let ord = ((a << 1) | r) as usize;
        out.set(lsfe_ld_code(op_idx, ord, is_bf));
        out.push_operand(simd_op(fp_reg(sc, rs)));
        out.push_operand(simd_op(fp_reg(sc, rt)));
        out.push_operand(mem_off(rn, 0));
    } else {
        // STF<op> / STBF<op>: <V>s, [Xn|SP]. Pure store -> Rt==31 and A==0.
        if rt != 0b11111 || a != 0 {
            return;
        }
        out.set(lsfe_st_code(op_idx, r, is_bf));
        out.push_operand(simd_op(fp_reg(sc, rs)));
        out.push_operand(mem_off(rn, 0));
    }
}

/// LSFE load-form [`Code`] for `(op_idx, ord, is_bf)` (`ord`: 0=plain,1=L,2=A,3=AL).
fn lsfe_ld_code(op_idx: usize, ord: usize, is_bf: bool) -> Code {
    use Code::*;
    #[rustfmt::skip]
    const F: [[Code; 4]; 5] = [
        [Ldfadd, Ldfaddl, Ldfadda, Ldfaddal],
        [Ldfmax, Ldfmaxl, Ldfmaxa, Ldfmaxal],
        [Ldfmin, Ldfminl, Ldfmina, Ldfminal],
        [Ldfmaxnm, Ldfmaxnml, Ldfmaxnma, Ldfmaxnmal],
        [Ldfminnm, Ldfminnml, Ldfminnma, Ldfminnmal],
    ];
    #[rustfmt::skip]
    const BF: [[Code; 4]; 5] = [
        [Ldbfadd, Ldbfaddl, Ldbfadda, Ldbfaddal],
        [Ldbfmax, Ldbfmaxl, Ldbfmaxa, Ldbfmaxal],
        [Ldbfmin, Ldbfminl, Ldbfmina, Ldbfminal],
        [Ldbfmaxnm, Ldbfmaxnml, Ldbfmaxnma, Ldbfmaxnmal],
        [Ldbfminnm, Ldbfminnml, Ldbfminnma, Ldbfminnmal],
    ];
    if is_bf { BF[op_idx][ord] } else { F[op_idx][ord] }
}

/// LSFE store-form [`Code`] for `(op_idx, r, is_bf)` (`r`: 0=plain,1=L).
fn lsfe_st_code(op_idx: usize, r: u32, is_bf: bool) -> Code {
    use Code::*;
    #[rustfmt::skip]
    const F: [[Code; 2]; 5] = [
        [Stfadd, Stfaddl],
        [Stfmax, Stfmaxl],
        [Stfmin, Stfminl],
        [Stfmaxnm, Stfmaxnml],
        [Stfminnm, Stfminnml],
    ];
    #[rustfmt::skip]
    const BF: [[Code; 2]; 5] = [
        [Stbfadd, Stbfaddl],
        [Stbfmax, Stbfmaxl],
        [Stbfmin, Stbfminl],
        [Stbfmaxnm, Stbfmaxnml],
        [Stbfminnm, Stbfminnml],
    ];
    let i = r as usize;
    if is_bf { BF[op_idx][i] } else { F[op_idx][i] }
}

#[derive(Clone, Copy)]
enum AtomicOp {
    Add,
    Clr,
    Eor,
    Set,
    Smax,
    Smin,
    Umax,
    Umin,
}

/// Emit a read-modify-write atomic (`LD<op>{A}{L}{B|H}` or its `ST<op>` alias).
///
/// Binary Ninja prints the fully-suffixed spelling (`ldaddalh`, `staddl`, ...),
/// so we set a representative [`Code`] and then install the exact [`Mnemonic`].
// Raw ARM ARM bitfields; grouping into a struct would obscure the 1:1 mapping.
#[allow(clippy::too_many_arguments)]
#[inline]
fn emit_atomic_rmw(
    size: u32,
    a: u32,
    r: u32,
    op: AtomicOp,
    rs: u32,
    rn: u32,
    rt: u32,
    out: &mut Instruction,
) {
    let code = match atomic_code(size, a, r, op) {
        Some(c) => c,
        None => return,
    };
    out.set(code);

    // ST<op> alias: when Rt==31 and A==0 the load result is discarded and Binary
    // Ninja prints `ST<op>{L}{B|H} <Rs>, [<Xn>]`.
    let st_alias = rt == 31 && a == 0;
    let x = size == 3;
    let w = w_of(x as u32);
    if st_alias {
        out.set_mnemonic(st_mnemonic(op, size, r));
        out.push_operand(gp(false, w, rs));
        out.push_operand(mem_off(rn, 0));
    } else {
        out.set_mnemonic(ld_mnemonic(op, size, a, r));
        out.push_operand(gp(false, w, rs));
        out.push_operand(gp(false, w, rt));
        out.push_operand(mem_off(rn, 0));
    }
}

/// The `LD<op>{A}{L}{B|H}` mnemonic for `(op, size, A, R)`.
fn ld_mnemonic(op: AtomicOp, size: u32, a: u32, r: u32) -> Mnemonic {
    // Suffix order: base + A + L (acquire then release) + size (B/H), matching
    // the ARM UAL spelling Binary Ninja emits.
    let ord = (a << 1) | r; // 0 none, 1 L, 2 A, 3 AL
    use AtomicOp::*;
    use Mnemonic as M;
    match op {
        Add => pick(size, ord, M::Ldadd, M::Ldaddl, M::Ldadda, M::Ldaddal, M::Ldaddb, M::Ldaddlb, M::Ldaddab, M::Ldaddalb, M::Ldaddh, M::Ldaddlh, M::Ldaddah, M::Ldaddalh),
        Clr => pick(size, ord, M::Ldclr, M::Ldclrl, M::Ldclra, M::Ldclral, M::Ldclrb, M::Ldclrlb, M::Ldclrab, M::Ldclralb, M::Ldclrh, M::Ldclrlh, M::Ldclrah, M::Ldclralh),
        Eor => pick(size, ord, M::Ldeor, M::Ldeorl, M::Ldeora, M::Ldeoral, M::Ldeorb, M::Ldeorlb, M::Ldeorab, M::Ldeoralb, M::Ldeorh, M::Ldeorlh, M::Ldeorah, M::Ldeoralh),
        Set => pick(size, ord, M::Ldset, M::Ldsetl, M::Ldseta, M::Ldsetal, M::Ldsetb, M::Ldsetlb, M::Ldsetab, M::Ldsetalb, M::Ldseth, M::Ldsetlh, M::Ldsetah, M::Ldsetalh),
        Smax => pick(size, ord, M::Ldsmax, M::Ldsmaxl, M::Ldsmaxa, M::Ldsmaxal, M::Ldsmaxb, M::Ldsmaxlb, M::Ldsmaxab, M::Ldsmaxalb, M::Ldsmaxh, M::Ldsmaxlh, M::Ldsmaxah, M::Ldsmaxalh),
        Smin => pick(size, ord, M::Ldsmin, M::Ldsminl, M::Ldsmina, M::Ldsminal, M::Ldsminb, M::Ldsminlb, M::Ldsminab, M::Ldsminalb, M::Ldsminh, M::Ldsminlh, M::Ldsminah, M::Ldsminalh),
        Umax => pick(size, ord, M::Ldumax, M::Ldumaxl, M::Ldumaxa, M::Ldumaxal, M::Ldumaxb, M::Ldumaxlb, M::Ldumaxab, M::Ldumaxalb, M::Ldumaxh, M::Ldumaxlh, M::Ldumaxah, M::Ldumaxalh),
        Umin => pick(size, ord, M::Ldumin, M::Lduminl, M::Ldumina, M::Lduminal, M::Lduminb, M::Lduminlb, M::Lduminab, M::Lduminalb, M::Lduminh, M::Lduminlh, M::Lduminah, M::Lduminalh),
    }
}

/// Select an LD-atomic mnemonic across the (size, ordering) matrix.
/// `b32` are the word/dword spellings (none/L/A/AL); `b*`/`h*` are the byte/half.
#[allow(clippy::too_many_arguments)]
fn pick(
    size: u32,
    ord: u32,
    base: Mnemonic,
    l: Mnemonic,
    a: Mnemonic,
    al: Mnemonic,
    b: Mnemonic,
    lb: Mnemonic,
    ab: Mnemonic,
    alb: Mnemonic,
    h: Mnemonic,
    lh: Mnemonic,
    ah: Mnemonic,
    alh: Mnemonic,
) -> Mnemonic {
    match size {
        0 => [b, lb, ab, alb][ord as usize],
        1 => [h, lh, ah, alh][ord as usize],
        // 32 and 64 share the same spelling (width comes from the register).
        _ => [base, l, a, al][ord as usize],
    }
}

/// The `ST<op>{L}{B|H}` alias mnemonic for `(op, size, R)`. (A is always 0 here.)
fn st_mnemonic(op: AtomicOp, size: u32, r: u32) -> Mnemonic {
    use AtomicOp::*;
    use Mnemonic as M;
    match op {
        Add => pick_st(size, r, M::Stadd, M::Staddl, M::Staddb, M::Staddlb, M::Staddh, M::Staddlh),
        Clr => pick_st(size, r, M::Stclr, M::Stclrl, M::Stclrb, M::Stclrlb, M::Stclrh, M::Stclrlh),
        Eor => pick_st(size, r, M::Steor, M::Steorl, M::Steorb, M::Steorlb, M::Steorh, M::Steorlh),
        Set => pick_st(size, r, M::Stset, M::Stsetl, M::Stsetb, M::Stsetlb, M::Stseth, M::Stsetlh),
        Smax => pick_st(size, r, M::Stsmax, M::Stsmaxl, M::Stsmaxb, M::Stsmaxlb, M::Stsmaxh, M::Stsmaxlh),
        Smin => pick_st(size, r, M::Stsmin, M::Stsminl, M::Stsminb, M::Stsminlb, M::Stsminh, M::Stsminlh),
        Umax => pick_st(size, r, M::Stumax, M::Stumaxl, M::Stumaxb, M::Stumaxlb, M::Stumaxh, M::Stumaxlh),
        Umin => pick_st(size, r, M::Stumin, M::Stuminl, M::Stuminb, M::Stuminlb, M::Stuminh, M::Stuminlh),
    }
}

/// Select an ST-atomic alias mnemonic across (size, release).
#[allow(clippy::too_many_arguments)]
fn pick_st(
    size: u32,
    r: u32,
    base: Mnemonic,
    l: Mnemonic,
    b: Mnemonic,
    lb: Mnemonic,
    h: Mnemonic,
    lh: Mnemonic,
) -> Mnemonic {
    match size {
        0 => [b, lb][r as usize],
        1 => [h, lh][r as usize],
        _ => [base, l][r as usize],
    }
}

/// Resolve the `LD<op>` Code from `(size, A, R, op)`.
fn atomic_code(size: u32, a: u32, r: u32, op: AtomicOp) -> Option<Code> {
    // ordering index: 0=plain,1=L,2=A,3=AL from (A,R).
    let ord = (a << 1) | r;
    // For byte/half (size 0/1) only the base/A/L/AL "B"/"H" suffixless catalog
    // entries exist (the catalog folds A/L/AL byte/half onto the base Code with
    // the right mnemonic). We pick the width-specific codes for 32/64 and the
    // *B/*H codes for 0/1.
    Some(match op {
        AtomicOp::Add => match (size, ord) {
            (2, 0) => Code::Ldadd32,
            (2, 1) => Code::Ldaddl32,
            (2, 2) => Code::Ldadda32,
            (2, 3) => Code::Ldaddal32,
            (3, 0) => Code::Ldadd64,
            (3, 1) => Code::Ldaddl64,
            (3, 2) => Code::Ldadda64,
            (3, 3) => Code::Ldaddal64,
            (0, _) => Code::Ldaddb,
            (1, _) => Code::Ldaddh,
            _ => return None,
        },
        AtomicOp::Clr => match (size, ord) {
            (2, 0) => Code::Ldclr32,
            (2, 1) => Code::Ldclrl32,
            (2, 2) => Code::Ldclra32,
            (2, 3) => Code::Ldclral32,
            (3, 0) => Code::Ldclr64,
            (3, 1) => Code::Ldclrl64,
            (3, 2) => Code::Ldclra64,
            (3, 3) => Code::Ldclral64,
            (0, _) => Code::Ldclrb,
            (1, _) => Code::Ldclrh,
            _ => return None,
        },
        AtomicOp::Eor => match (size, ord) {
            (2, 0) => Code::Ldeor32,
            (2, 1) => Code::Ldeorl32,
            (2, 2) => Code::Ldeora32,
            (2, 3) => Code::Ldeoral32,
            (3, 0) => Code::Ldeor64,
            (3, 1) => Code::Ldeorl64,
            (3, 2) => Code::Ldeora64,
            (3, 3) => Code::Ldeoral64,
            (0, _) => Code::Ldeorb,
            (1, _) => Code::Ldeorh,
            _ => return None,
        },
        AtomicOp::Set => match (size, ord) {
            (2, 0) => Code::Ldset32,
            (2, 1) => Code::Ldsetl32,
            (2, 2) => Code::Ldseta32,
            (2, 3) => Code::Ldsetal32,
            (3, 0) => Code::Ldset64,
            (3, 1) => Code::Ldsetl64,
            (3, 2) => Code::Ldseta64,
            (3, 3) => Code::Ldsetal64,
            (0, _) => Code::Ldsetb,
            (1, _) => Code::Ldseth,
            _ => return None,
        },
        AtomicOp::Smax => atomic_simple(size, Code::Ldsmax32, Code::Ldsmax64, Code::Ldsmaxb, Code::Ldsmaxh),
        AtomicOp::Smin => atomic_simple(size, Code::Ldsmin32, Code::Ldsmin64, Code::Ldsminb, Code::Ldsminh),
        AtomicOp::Umax => atomic_simple(size, Code::Ldumax32, Code::Ldumax64, Code::Ldumaxb, Code::Ldumaxh),
        AtomicOp::Umin => atomic_simple(size, Code::Ldumin32, Code::Ldumin64, Code::Lduminb, Code::Lduminh),
    })
}

/// SMAX/SMIN/UMAX/UMIN only have base + B/H + 32/64 catalog Codes (the A/L/AL
/// ordering variants reuse the base Code with the mnemonic carrying the suffix).
fn atomic_simple(size: u32, c32: Code, c64: Code, cb: Code, ch: Code) -> Code {
    match size {
        0 => cb,
        1 => ch,
        3 => c64,
        _ => c32,
    }
}

/// `SWP{A}{L}{B|H}` swap (LSE).
#[inline]
fn emit_swp(size: u32, a: u32, r: u32, rs: u32, rn: u32, rt: u32, out: &mut Instruction) {
    let ord = (a << 1) | r;
    let code = match (size, ord) {
        (2, 0) => Code::Swp32,
        (2, 1) => Code::Swpl32,
        (2, 2) => Code::Swpa32,
        (2, 3) => Code::Swpal32,
        (3, 0) => Code::Swp64,
        (3, 1) => Code::Swpl64,
        (3, 2) => Code::Swpa64,
        (3, 3) => Code::Swpal64,
        (0, 0) => Code::Swpb,
        (0, 1) => Code::Swplb,
        (0, 2) => Code::Swpab,
        (0, 3) => Code::Swpalb,
        (1, 0) => Code::Swph,
        (1, 1) => Code::Swplh,
        (1, 2) => Code::Swpah,
        _ => Code::Swpalh,
    };
    out.set(code);
    let x = size == 3;
    let w = w_of(x as u32);
    out.push_operand(gp(false, w, rs));
    out.push_operand(gp(false, w, rt));
    out.push_operand(mem_off(rn, 0));
}

// ---------------------------------------------------------------------------
// Pointer-authenticated loads (LDRAA / LDRAB).
// ---------------------------------------------------------------------------

/// `LDRAA`/`LDRAB` (pointer-authenticated, key A/B). Encoding:
/// `11 111 0 00 M S 1 imm9 W 1 Rn Rt`. The signed 10-bit offset is
/// `SignExtend(S:imm9, 10) << 3`. `W`=bit11 selects pre-index writeback.
#[inline]
fn decode_pac(word: u32, features: FeatureSet, out: &mut Instruction) {
    if bit(word, 26) != 0 {
        return; // V must be 0
    }
    if !features.has(Feature::PAuth) {
        return;
    }
    // size must be 11 and the fixed bits hold (these forms are 64-bit only).
    if bits(word, 30, 2) != 0b11 {
        return;
    }
    let m = bit(word, 23); // 0 = A, 1 = B
    let s = bit(word, 22); // sign bit of the 10-bit offset
    let imm9 = bits(word, 12, 9);
    let wbit = bit(word, 11);
    let rn = bits(word, 5, 5);
    let rt = bits(word, 0, 5);

    let off10 = (s << 9) | imm9;
    let imm = sign_extend(off10 as u64, 10) << 3;

    let code = match (m, wbit) {
        (0, 0) => Code::LdraaOff,
        (0, 1) => Code::LdraaPre,
        (1, 0) => Code::LdrabOff,
        _ => Code::LdrabPre,
    };
    out.set(code);
    out.push_operand(gp(false, RegWidth::X64, rt));
    if wbit == 1 {
        out.push_operand(mem_pre(rn, imm));
    } else {
        out.push_operand(mem_off(rn, imm));
    }
}

// ---------------------------------------------------------------------------
// LDAPUR/STLUR (RCpc unscaled) and memory tagging (STG/LDG/...).
// ---------------------------------------------------------------------------

/// Dispatch for `word<29:24> == 0b011001`: this region holds both the
/// `LDAPUR`/`STLUR` (RCpc, unscaled, immediate) forms and the memory-tagging
/// `STG`/`STZG`/`ST2G`/`STZ2G`/`LDG`/`LDGM`/`STGM`/`STZGM` forms. They are split
/// on `word<26>`(V) — tags use the `0b11011 0 ...` major while ldapstl uses the
/// `size 011001 ...` major; concretely we distinguish on `size`(bits<31:30>) and
/// `word<23:22>`.
#[inline]
fn decode_ldst_tags_or_ldapstl(word: u32, features: FeatureSet, out: &mut Instruction) {
    // The FEAT_MOPS forms that share this `word<29:24> == 0b011001` major (the
    // `CPYF*`/`SET*` family, `word<26> == 0`, `word<21> == 0`,
    // `word<11:10> == 0b01`) are already handled by the dedicated MOPS dispatch
    // in [`decode`] before this function is reached, so here we only see the
    // `LDAPUR`/`STLUR` (RCpc unscaled), `LDIAPP`/`STILP` (RCPC3 ordered pair) and
    // memory-tagging forms. They are distinguished by bit21 and word<11:10>: tag
    // forms set bit21; with bit21 clear, word<11:10>==0b00 is the RCpc unscaled
    // GP form and word<11:10>==0b10 is the RCPC3 LDIAPP/STILP pair.
    if bit(word, 21) == 1 {
        // bit21==1 splits two unrelated families sharing this major: memory
        // tagging (size==0b11) and the FEAT_THE / FEAT_LSE128 atomic block
        // (size==0b00 for the 32/64-bit unprivileged + RCW forms, size==0b01 for
        // the 64-bit / RCWS forms). Sizes 0b10/0b11 are tags-only territory.
        if bits(word, 30, 2) == 0b11 {
            decode_tags(word, features, out);
        } else {
            decode_the_atomic(word, features, out);
        }
    } else if bits(word, 10, 2) == 0b00 {
        decode_ldapstl(word, out);
    } else if bits(word, 10, 2) == 0b10 {
        // word<23> splits the two RCPC3 families sharing this slot: bit23==0 is
        // the LDIAPP/STILP ordered pair, bit23==1 is the writeback STLR/LDAPR.
        if bit(word, 23) == 0 {
            decode_ldiapp_stilp(word, features, out);
        } else {
            decode_stlr_ldapr_wb(word, features, out);
        }
    }
    // `word<11:10> == 0b11` with bit21 clear is unallocated here (the `0b01`
    // MOPS case never reaches this function).
}

/// `LDAPUR`/`STLUR` and the signed/byte/half variants (RCpc, unscaled). Encoding:
/// `size 011001 opc(23:22) 0 imm9 00 Rn Rt`. `imm9` signed, unscaled. No V forms.
#[inline]
fn decode_ldapstl(word: u32, out: &mut Instruction) {
    let size = bits(word, 30, 2);
    let opc = bits(word, 22, 2);
    let imm9 = bits(word, 12, 9);
    let rn = bits(word, 5, 5);
    let rt = bits(word, 0, 5);
    let imm = sign_extend(imm9 as u64, 9);

    macro_rules! emit {
        ($code:expr, $x:expr) => {{
            out.set($code);
            out.push_operand(gp(false, w_of($x as u32), rt));
            out.push_operand(mem_off(rn, imm));
            return;
        }};
    }
    match (size, opc) {
        (0, 0b00) => emit!(Code::Stlurb, 0),
        (0, 0b01) => emit!(Code::Ldapurb, 0),
        (0, 0b10) => emit!(Code::Ldapursb64, 1),
        (0, 0b11) => emit!(Code::Ldapursb32, 0),
        (1, 0b00) => emit!(Code::Stlurh, 0),
        (1, 0b01) => emit!(Code::Ldapurh, 0),
        (1, 0b10) => emit!(Code::Ldapursh64, 1),
        (1, 0b11) => emit!(Code::Ldapursh32, 0),
        (2, 0b00) => emit!(Code::Stlur32, 0),
        (2, 0b01) => emit!(Code::Ldapur32, 0),
        (2, 0b10) => emit!(Code::Ldapursw, 1),
        (3, 0b00) => emit!(Code::Stlur64, 1),
        (3, 0b01) => emit!(Code::Ldapur64, 1),
        _ => {}
    }
}

/// FEAT_LRCPC3 SIMD&FP `LDAPUR`/`STLUR` (release/acquire, unscaled). Encoding:
/// `size 011 1 01 opc 0 imm9 10 Rn Rt` with `V=1`. The `(size, opc)` pair selects
/// the register view and the load/store direction exactly as the regular SIMD&FP
/// unscaled forms: `opc<0>` is the load bit, and `(size==00, opc<1>==1)` selects
/// the 128-bit `Q` view; sizes `01/10/11` with `opc<1>==1` are unallocated.
#[inline]
fn decode_fp_ldapstl(word: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Rcpc3) {
        return;
    }
    let size = bits(word, 30, 2);
    let opc = bits(word, 22, 2);
    let imm9 = bits(word, 12, 9);
    let rn = bits(word, 5, 5);
    let rt = bits(word, 0, 5);
    let imm = sign_extend(imm9 as u64, 9);
    let load = (opc & 1) == 1;

    // (size, opc<1>) -> B/H/S/D/Q access code (0..4).
    let fp_code = match (size, opc >> 1) {
        (0b00, 0) => 0, // B
        (0b01, 0) => 1, // H
        (0b10, 0) => 2, // S
        (0b11, 0) => 3, // D
        (0b00, 1) => 4, // Q
        _ => return,    // size 01/10/11 with opc<1>==1 is unallocated.
    };
    let code = match (fp_code, load) {
        (0, true) => Code::LdapurFp8,
        (1, true) => Code::LdapurFp16,
        (2, true) => Code::LdapurFp32,
        (3, true) => Code::LdapurFp64,
        (4, true) => Code::LdapurFp128,
        (0, false) => Code::StlurFp8,
        (1, false) => Code::StlurFp16,
        (2, false) => Code::StlurFp32,
        (3, false) => Code::StlurFp64,
        _ => Code::StlurFp128,
    };
    out.set(code);
    out.push_operand(simd_op(fp_reg(fp_code, rt)));
    out.push_operand(mem_off(rn, imm));
}

/// FEAT_LRCPC3 `LDIAPP`/`STILP` (RCPC3 ordered load/store pair). Encoding:
/// `sz 011001 0 L 0 Rt2 0 0 0 o(12) 1 0 Rn Rt`, where `sz` is `10`(W)/`11`(X),
/// `L` is the load bit (1 = `LDIAPP`, 0 = `STILP`), `Rt2` is the explicitly
/// encoded second transfer register and `o`(bit12) selects the offset form
/// (`o==1`, no writeback) versus the indexed form (`o==0`: post-index for the
/// load, pre-index for the store, with an implicit datasize displacement).
#[inline]
fn decode_ldiapp_stilp(word: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Rcpc3) {
        return;
    }
    // Fixed-bit constraint: bits<15:13> == 0. The dispatcher has already pinned
    // word<29:24> == 0b011001, word<23> == 0, word<21> == 0 and
    // word<11:10> == 0b10. A nonzero reserved field is unallocated (LLVM rejects).
    if bits(word, 13, 3) != 0 {
        return;
    }
    let sz = bits(word, 30, 2);
    // Only the 32-bit (W) and 64-bit (X) variants are allocated.
    let w = match sz {
        0b10 => RegWidth::W32,
        0b11 => RegWidth::X64,
        _ => return,
    };
    let load = bit(word, 22) == 1;
    let rt2 = bits(word, 16, 5);
    let offset_form = bit(word, 12) == 1;
    let rn = bits(word, 5, 5);
    let rt = bits(word, 0, 5);

    let rt_op = gp(false, w, rt);
    let rt2_op = gp(false, w, rt2);

    if offset_form {
        let code = if load {
            Code::LdiappOff
        } else {
            Code::StilpOff
        };
        out.set(code);
        out.push_operand(rt_op);
        out.push_operand(rt2_op);
        out.push_operand(mem_off(rn, 0));
    } else {
        // Indexed: load is post-index +datasize, store is pre-index -datasize.
        // datasize (bytes for the pair) = 8 for W, 16 for X.
        let bytes = if sz == 0b11 { 16 } else { 8 };
        out.set(if load {
            Code::LdiappPost
        } else {
            Code::StilpPre
        });
        out.push_operand(rt_op);
        out.push_operand(rt2_op);
        if load {
            out.push_operand(mem_post(rn, bytes));
        } else {
            out.push_operand(mem_pre(rn, -bytes));
        }
    }
}

/// FEAT_LRCPC3 writeback `STLR` (pre-index) / `LDAPR` (post-index). Encoding:
/// `sz 011001 1 L 0 000000000 10 Rn Rt`, `sz` is `10`(W)/`11`(X). `L`(bit22) is
/// the load bit: `0` is the store-release pre-index form (`STLR <Wt>, [<Xn>,
/// #-dsz]!`), `1` is the load-acquire post-index form (`LDAPR <Wt>, [<Xn>],
/// #dsz`). The `imm9` field (bits<20:12>) is reserved and must be zero; the
/// displacement is the implicit access size (`4` for W, `8` for X).
#[inline]
fn decode_stlr_ldapr_wb(word: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Rcpc3) {
        return;
    }
    // imm9 (bits<20:12>) is a reserved zero field for these forms.
    if bits(word, 12, 9) != 0 {
        return;
    }
    let sz = bits(word, 30, 2);
    let w = match sz {
        0b10 => RegWidth::W32,
        0b11 => RegWidth::X64,
        _ => return,
    };
    let load = bit(word, 22) == 1;
    let rn = bits(word, 5, 5);
    let rt = bits(word, 0, 5);
    let bytes: i64 = if sz == 0b11 { 8 } else { 4 };

    if load {
        // LDAPR post-index, writeback: [Xn], #dsz.
        out.set(if sz == 0b11 {
            Code::LdaprPost64
        } else {
            Code::LdaprPost32
        });
        out.push_operand(gp(false, w, rt));
        out.push_operand(mem_post(rn, bytes));
    } else {
        // STLR pre-index, writeback: [Xn, #-dsz]!.
        out.set(if sz == 0b11 {
            Code::StlrPre64
        } else {
            Code::StlrPre32
        });
        out.push_operand(gp(false, w, rt));
        out.push_operand(mem_pre(rn, -bytes));
    }
}

/// FEAT_THE / FEAT_LSE128 atomic memory operations that share the
/// `word<29:24> == 0b011001`, `word<21> == 1`, `size != 0b11` major with the
/// memory-tagging forms. The block layout is
/// `size 011 0 01 A R 1 Rs o3 opc(14:12) op2(11:10) Rn Rt`, with `A`=bit23,
/// `R`=bit22 selecting the acquire/release ordering and `(op2, o3, opc)` selecting
/// the family:
///   op2=01 o3=0 opc=0/1/3  -> LDTADD/LDTCLR/LDTSET (size 00->W, 01->X)
///   op2=01 o3=1 opc=0      -> SWPT                  (size 00->W, 01->X)
///   op2=10 o3=0 opc=0      -> RCWCAS  (size 00) / RCWSCAS  (size 01) [64-bit]
///   op2=11 o3=0 opc=0      -> RCWCASP (size 00) / RCWSCASP (size 01) [pair]
///   op2=00 o3=0 opc=1/3    -> LDCLRP/LDSETP   (FEAT_LSE128, size 00 only) [pair]
///   op2=00 o3=1 opc=0      -> SWPP            (FEAT_LSE128, size 00 only) [pair]
///   op2=00 o3=1 opc=1/2/3  -> RCWCLRP/RCWSWPP/RCWSETP (size 00)
///                             RCWSCLRP/RCWSSWPP/RCWSSETP (size 01) [pair]
#[inline]
fn decode_the_atomic(word: u32, features: FeatureSet, out: &mut Instruction) {
    let sz = bits(word, 30, 2); // 0 -> 32-bit / non-S, 1 -> 64-bit / RCWS.
    let a = bit(word, 23);
    let r = bit(word, 22);
    let rs = bits(word, 16, 5);
    let o3 = bit(word, 15);
    let opc = bits(word, 12, 3);
    let op2 = bits(word, 10, 2);
    let rn = bits(word, 5, 5);
    let rt = bits(word, 0, 5);
    // ordering index from (A, R): 0 none, 1 L (release), 2 A (acquire), 3 AL.
    let ord = ((a << 1) | r) as usize;

    match op2 {
        // op2==01: the unprivileged LDTADD/LDTCLR/LDTSET and SWPT forms. Both the
        // 32-bit (size 00 -> W) and 64-bit (size 01 -> X) widths exist; sizes
        // 10/11 never reach here (handled as tags above / size!=00..01 guard).
        0b01 => {
            if sz > 1 {
                return;
            }
            if !features.has(Feature::The) {
                return;
            }
            // o3==0: LDT<op> by opc (0 add, 1 clr, 3 set). o3==1, opc==0: SWPT.
            let (codes_w, codes_x): (&[Code; 4], &[Code; 4]) = if o3 == 1 {
                if opc != 0 {
                    return;
                }
                (
                    &[Code::Swpt32, Code::Swptl32, Code::Swpta32, Code::Swptal32],
                    &[Code::Swpt64, Code::Swptl64, Code::Swpta64, Code::Swptal64],
                )
            } else {
                match opc {
                    0b000 => (
                        &[Code::Ldtadd32, Code::Ldtaddl32, Code::Ldtadda32, Code::Ldtaddal32],
                        &[Code::Ldtadd64, Code::Ldtaddl64, Code::Ldtadda64, Code::Ldtaddal64],
                    ),
                    0b001 => (
                        &[Code::Ldtclr32, Code::Ldtclrl32, Code::Ldtclra32, Code::Ldtclral32],
                        &[Code::Ldtclr64, Code::Ldtclrl64, Code::Ldtclra64, Code::Ldtclral64],
                    ),
                    0b011 => (
                        &[Code::Ldtset32, Code::Ldtsetl32, Code::Ldtseta32, Code::Ldtsetal32],
                        &[Code::Ldtset64, Code::Ldtsetl64, Code::Ldtseta64, Code::Ldtsetal64],
                    ),
                    _ => return,
                }
            };
            let x = sz == 1;
            let code = if x { codes_x[ord] } else { codes_w[ord] };
            out.set(code);
            let w = w_of(x as u32);
            // Rs, Rt, [Xn|SP].
            out.push_operand(gp(false, w, rs));
            out.push_operand(gp(false, w, rt));
            out.push_operand(mem_off(rn, 0));
        }
        // op2==10: RCWCAS (size 00) / RCWSCAS (size 01). 64-bit, o3==0, opc==0.
        0b10 => {
            if sz > 1 || o3 != 0 || opc != 0 {
                return;
            }
            if !features.has(Feature::The) {
                return;
            }
            let code = if sz == 1 {
                [Code::Rcwscas, Code::Rcwscasl, Code::Rcwscasa, Code::Rcwscasal][ord]
            } else {
                [Code::Rcwcas, Code::Rcwcasl, Code::Rcwcasa, Code::Rcwcasal][ord]
            };
            out.set(code);
            // Rs, Rt, [Xn|SP] (both 64-bit).
            out.push_operand(gp(false, RegWidth::X64, rs));
            out.push_operand(gp(false, RegWidth::X64, rt));
            out.push_operand(mem_off(rn, 0));
        }
        // op2==11: RCWCASP (size 00) / RCWSCASP (size 01). Pair: Rs even, Rt even.
        0b11 => {
            if sz > 1 || o3 != 0 || opc != 0 {
                return;
            }
            if !features.has(Feature::The) {
                return;
            }
            // Even-register pairs only; odd Rs/Rt is UNDEFINED (ARM ARM).
            if (rs & 1) != 0 || (rt & 1) != 0 {
                return;
            }
            let code = if sz == 1 {
                [Code::Rcwscasp, Code::Rcwscaspl, Code::Rcwscaspa, Code::Rcwscaspal][ord]
            } else {
                [Code::Rcwcasp, Code::Rcwcaspl, Code::Rcwcaspa, Code::Rcwcaspal][ord]
            };
            out.set(code);
            // Rs, Rs+1, Rt, Rt+1, [Xn|SP].
            out.push_operand(gp(false, RegWidth::X64, rs));
            out.push_operand(gp(false, RegWidth::X64, (rs + 1) & 0x1f));
            out.push_operand(gp(false, RegWidth::X64, rt));
            out.push_operand(gp(false, RegWidth::X64, (rt + 1) & 0x1f));
            out.push_operand(mem_off(rn, 0));
        }
        // op2==00: the pair load-op forms. LSE128 (LDCLRP/LDSETP/SWPP, size 00
        // only) and FEAT_THE RCW pair ops (RCWCLRP/RCWSWPP/RCWSETP, size 00;
        // RCWSCLRP/RCWSSWPP/RCWSSETP, size 01). All print `Rt, Rs, [Xn|SP]`.
        _ => {
            // Resolve (code, feature) for this (size, o3, opc, ordering).
            let resolved: Option<(Code, Feature)> = if o3 == 0 {
                // LSE128 LDCLRP / LDSETP — size 00 only.
                if sz != 0 {
                    None
                } else {
                    match opc {
                        0b001 => Some((
                            [Code::Ldclrp, Code::Ldclrpl, Code::Ldclrpa, Code::Ldclrpal][ord],
                            Feature::Lse128,
                        )),
                        0b011 => Some((
                            [Code::Ldsetp, Code::Ldsetpl, Code::Ldsetpa, Code::Ldsetpal][ord],
                            Feature::Lse128,
                        )),
                        _ => None,
                    }
                }
            } else {
                // o3==1: opc 0 -> SWPP (LSE128, size 00); opc 1/2/3 -> RCW pair
                // ops (RCW* size 00, RCWS* size 01).
                match opc {
                    0b000 => {
                        if sz != 0 {
                            None
                        } else {
                            Some((
                                [Code::Swpp, Code::Swppl, Code::Swppa, Code::Swppal][ord],
                                Feature::Lse128,
                            ))
                        }
                    }
                    0b001 => Some((
                        if sz == 1 {
                            [Code::Rcwsclrp, Code::Rcwsclrpl, Code::Rcwsclrpa, Code::Rcwsclrpal][ord]
                        } else {
                            [Code::Rcwclrp, Code::Rcwclrpl, Code::Rcwclrpa, Code::Rcwclrpal][ord]
                        },
                        Feature::The,
                    )),
                    0b010 => Some((
                        if sz == 1 {
                            [Code::Rcwsswpp, Code::Rcwsswppl, Code::Rcwsswppa, Code::Rcwsswppal][ord]
                        } else {
                            [Code::Rcwswpp, Code::Rcwswppl, Code::Rcwswppa, Code::Rcwswppal][ord]
                        },
                        Feature::The,
                    )),
                    0b011 => Some((
                        if sz == 1 {
                            [Code::Rcwssetp, Code::Rcwssetpl, Code::Rcwssetpa, Code::Rcwssetpal][ord]
                        } else {
                            [Code::Rcwsetp, Code::Rcwsetpl, Code::Rcwsetpa, Code::Rcwsetpal][ord]
                        },
                        Feature::The,
                    )),
                    _ => None,
                }
            };
            let (code, feat) = match resolved {
                Some(v) => v,
                None => return,
            };
            if !features.has(feat) {
                return;
            }
            // The load-op-pair forms reserve register 31 in both the Rs and Rt
            // fields (no xzr/SP form); such encodings are UNDEFINED (ARM ARM), so
            // leave them Invalid rather than printing a bogus `xzr` operand.
            if rs == 31 || rt == 31 {
                return;
            }
            out.set(code);
            // Pair forms print Rt first, then Rs, then the base.
            out.push_operand(gp(false, RegWidth::X64, rt));
            out.push_operand(gp(false, RegWidth::X64, rs));
            out.push_operand(mem_off(rn, 0));
        }
    }
}

/// Memory-tagging stores/loads (FEAT_MTE). Encoding:
/// `1101 1001 opc(23:22) 1 imm9 op2(11:10) Rn Rt`, where `opc` selects the
/// instruction family and `op2` the addressing mode (01 post, 10 offset, 11 pre).
/// The `bulk` forms (`LDGM`/`STGM`/`STZGM`) use `op2==00` with a zero offset and
/// `LDG` uses `op2==00` with an offset.
#[inline]
fn decode_tags(word: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Mte) {
        return;
    }
    // The memory-tagging encoding is fixed `1101 1001 ...`: the size field
    // (word<31:30>) must be 0b11. Sizes 00/01/10 share this `word<29:24>` major
    // (via the LDAPUR/STLUR dispatch) but are UNALLOCATED for the tag forms, so
    // leave them Invalid rather than over-decoding.
    if bits(word, 30, 2) != 0b11 {
        return;
    }
    let opc = bits(word, 22, 2);
    let imm9 = bits(word, 12, 9);
    let op2 = bits(word, 10, 2);
    let rn = bits(word, 5, 5);
    let rt = bits(word, 0, 5);
    let imm = sign_extend(imm9 as u64, 9) << 4; // tag granule = 16 bytes

    // opc: 00 STG, 01 STZG, 10 ST2G, 11 STZ2G (for op2 != 00).
    // op2: 01 post-index, 10 signed offset, 11 pre-index.
    if op2 != 0b00 {
        let codes = match opc {
            0b00 => [Code::StgPost, Code::StgOff, Code::StgPre],
            0b01 => [Code::StzgPost, Code::StzgOff, Code::StzgPre],
            0b10 => [Code::St2gPost, Code::St2gOff, Code::St2gPre],
            _ => [Code::Stz2gPost, Code::Stz2gOff, Code::Stz2gPre],
        };
        let (code, mem) = match op2 {
            0b01 => (codes[0], mem_post(rn, imm)),
            0b10 => (codes[1], mem_off(rn, imm)),
            _ => (codes[2], mem_pre(rn, imm)),
        };
        out.set(code);
        // STG/STZG/ST2G/STZ2G: Xt|SP, [Xn|SP, ...].
        out.push_operand(gp(true, RegWidth::X64, rt));
        out.push_operand(mem);
        return;
    }

    // op2 == 00: LDG (opc 01, with offset) or the bulk forms
    //   opc 00 -> STZGM, 10 -> STGM, 11 -> LDGM (all `Xt, [Xn]`, no offset).
    // The bulk forms encode no immediate: their imm9 field (bits<20:12>) is fixed
    // zero, so a non-zero value is UNDEFINED (ARM ARM). LDG genuinely uses imm9 as
    // a tag-scaled offset, so only the bulk forms get the zero-imm guard.
    match opc {
        0b00 => {
            if imm9 != 0 {
                return;
            }
            out.set(Code::Stzgm);
            out.push_operand(gp(false, RegWidth::X64, rt));
            out.push_operand(mem_off(rn, 0));
        }
        0b01 => {
            // LDG: Xt, [Xn, #imm].
            out.set(Code::LdgOff);
            out.push_operand(gp(false, RegWidth::X64, rt));
            out.push_operand(mem_off(rn, imm));
        }
        0b10 => {
            if imm9 != 0 {
                return;
            }
            out.set(Code::Stgm);
            out.push_operand(gp(false, RegWidth::X64, rt));
            out.push_operand(mem_off(rn, 0));
        }
        _ => {
            if imm9 != 0 {
                return;
            }
            out.set(Code::Ldgm);
            out.push_operand(gp(false, RegWidth::X64, rt));
            out.push_operand(mem_off(rn, 0));
        }
    }
}

// ---------------------------------------------------------------------------
// Shared: push the data register for a GP/PRFM form.
// ---------------------------------------------------------------------------

/// Push the leading data register for a classified GP / PRFM register form.
#[inline]
fn push_data_reg(out: &mut Instruction, form: &LdStForm, rt: u32) {
    if form.is_prfm {
        out.push_operand(prefetch_op(rt));
    } else {
        out.push_operand(gp(false, w_of(form.gp_x as u32), rt));
    }
}

#[cfg(test)]
mod tests {
    use crate::decode::decode;
    use crate::features::{Feature, FeatureSet};
    use crate::format::{BufSink, FmtFormatter, Formatter};
    use crate::mnemonic::Code;

    /// The differential-corpus anchor address (PC-relative literal targets match
    /// the corpus verbatim when decoded here).
    const ADDRESS: u64 = 0x8000_0000_0000_0004;

    /// Decode `word` at the corpus anchor, render with the default formatter into
    /// a fixed buffer, and assert the text equals `expected`. Allocation-free so
    /// it builds on the default no-`alloc` test tier.
    #[track_caller]
    fn assert_dis(word: u32, expected: &str) {
        let insn = decode(word, ADDRESS, FeatureSet::ALL);
        let mut buf = [0u8; 128];
        let mut sink = BufSink::new(&mut buf);
        FmtFormatter::new().format(&insn, &mut sink);
        assert!(!sink.overflowed(), "BufSink overflowed rendering {expected:?}");
        assert_eq!(sink.as_str(), expected, "word={word:#010x}");
    }

    #[test]
    fn unsigned_offset_forms() {
        assert_dis(0xF96ECD59, "ldr     x25, [x10, #0x5d98]");
        assert_dis(0xB9205D5A, "str     w26, [x10, #0x205c]");
        assert_dis(0x394FEDDF, "ldrb    wzr, [x14, #0x3fb]");
        assert_dis(0x7983E17B, "ldrsh   x27, [x11, #0x1f0]");
        // SIMD&FP unsigned-offset: B/S/Q register views.
        assert_dis(0x3D4666EE, "ldr     b14, [x23, #0x199]");
        assert_dis(0x3DDEABF2, "ldr     q18, [sp, #0x7aa0]");
    }

    #[test]
    fn register_offset_shift_rules() {
        // uxtw/sxtw use the W index view and show the amount only when S==1.
        assert_dis(0xF87B5ADC, "ldr     x28, [x22, w27, uxtw #0x3]");
        assert_dis(0xF86D7B64, "ldr     x4, [x27, x13, lsl #0x3]");
        // uxtx (LSL-equiv) with S==0: no decoration at all.
        assert_dis(0xBC7F6A15, "ldr     s21, [x16, xzr]");
        // sxtx with S==0: keyword only, no amount.
        assert_dis(0x3863E8A0, "ldrb    w0, [x5, x3, sxtx]");
        // sxtx with S==1 and a zero amount: the `#0x0` is shown.
        assert_dis(0x3862F9C1, "ldrb    w1, [x14, x2, sxtx #0x0]");
        // lsl with S==1 and zero amount.
        assert_dis(0x387E7B04, "ldrb    w4, [x24, x30, lsl #0x0]");
    }

    #[test]
    fn unscaled_and_index() {
        assert_dis(0xF843F3F3, "ldur    x19, [sp, #0x3f]");
        assert_dis(0xF845EC10, "ldr     x16, [x0, #0x5e]!");
        assert_dis(0xF85FA74E, "ldr     x14, [x26], #-0x6");
        assert_dis(0xF85E09FB, "ldtr    x27, [x15, #-0x20]");
    }

    #[test]
    fn literal_loads_and_prefetch() {
        assert_dis(0x58564E32, "ldr     x18, 0x80000000000ac9c8");
        assert_dis(0x98993236, "ldrsw   x22, 0x7ffffffffff32648");
        // PRFM literal with a named prefetch op.
        assert_dis(0xD80580C5, "prfm    pldl3strm, 0x800000000000b01c");
        // PRFM unsigned-offset named, and PRFUM with a raw (reserved) op.
        assert_dis(0xF9818708, "prfm    plil1keep, [x24, #0x308]");
        assert_dis(0xF89961AF, "prfum   #0xf, [x13, #-0x6a]");
    }

    #[test]
    fn pair_forms() {
        assert_dis(0xA943A9CC, "ldp     x12, x10, [x14, #0x38]");
        assert_dis(0xA8EA2124, "ldp     x4, x8, [x9], #-0x160");
        assert_dis(0xA9EACE40, "ldp     x0, x19, [x18, #-0x158]!");
        assert_dis(0x69723695, "ldpsw   x21, x13, [x20, #-0x70]");
        assert_dis(0xA86520FA, "ldnp    x26, x8, [x7, #-0x1b0]");
        assert_dis(0xAD576824, "ldp     q4, q26, [x1, #0x2e0]");
    }

    #[test]
    fn rcpc3_ldapur_stlur_fp() {
        // FEAT_LRCPC3 SIMD&FP LDAPUR/STLUR (unscaled, imm9 = 4). Mnemonic +
        // register view matches LLVM `--mattr=+all`.
        assert_dis(0x1d404820, "ldapur  b0, [x1, #0x4]");
        assert_dis(0xdd404820, "ldapur  d0, [x1, #0x4]");
        assert_dis(0x1dc04820, "ldapur  q0, [x1, #0x4]");
        assert_dis(0x9d004820, "stlur   s0, [x1, #0x4]");
        assert_dis(0x1d804820, "stlur   q0, [x1, #0x4]");
    }

    #[test]
    fn rcpc3_ldiapp_stilp() {
        // Ordered load/store pair (offset + indexed). Rt2 is explicitly encoded.
        assert_dis(0x99411840, "ldiapp  w0, w1, [x2]");
        assert_dis(0xd9411840, "ldiapp  x0, x1, [x2]");
        assert_dis(0x99410840, "ldiapp  w0, w1, [x2], #0x8");
        assert_dis(0xd9410840, "ldiapp  x0, x1, [x2], #0x10");
        assert_dis(0x99011840, "stilp   w0, w1, [x2]");
        assert_dis(0xd9010840, "stilp   x0, x1, [x2, #-0x10]!");
        // Rt2 != Rt+1 (decoded verbatim).
        assert_dis(0x99421860, "ldiapp  w0, w2, [x3]");
        // Reserved bits<15:13> != 0 -> unallocated.
        assert_dis(0x990038b8, "");
    }

    #[test]
    fn the_ldtp_sttp_ldtnp_sttnp() {
        // FEAT_THE unprivileged translation-enhanced pairs. X-register pairs,
        // imm7 scaled by 8. Verified against llvm-mc --mattr=+all.
        // LDTP/STTP: offset (idx=10), post (idx=01), pre (idx=11).
        assert_dis(0xe9400440, "ldtp    x0, x1, [x2]");
        assert_dis(0xe9600440, "ldtp    x0, x1, [x2, #-0x200]");
        assert_dis(0xe95f8440, "ldtp    x0, x1, [x2, #0x1f8]");
        assert_dis(0xe8a00440, "sttp    x0, x1, [x2], #-0x200");
        assert_dis(0xe9c10440, "ldtp    x0, x1, [x2, #0x10]!");
        assert_dis(0xe8810440, "sttp    x0, x1, [x2], #0x10");
        assert_dis(0xe9410440, "ldtp    x0, x1, [x2, #0x10]");
        // LDTNP/STTNP: non-temporal (idx=00), rendered `[Xn{, #imm}]`.
        assert_dis(0xe8400440, "ldtnp   x0, x1, [x2]");
        assert_dis(0xe8200440, "sttnp   x0, x1, [x2, #-0x200]");
        assert_dis(0xe85fffff, "ldtnp   xzr, xzr, [sp, #0x1f8]");
        // Rt2 != Rt+1 is decoded verbatim.
        assert_dis(0xe9400840, "ldtp    x0, x2, [x2]");
    }

    #[test]
    fn the_feature_gated_and_roundtrip() {
        use crate::encode::encode;
        for &word in &[
            0xe9400440u32,
            0xe9600440,
            0xe95f8440,
            0xe8a00440,
            0xe9c10440,
            0xe8810440,
            0xe9410440,
            0xe8400440,
            0xe8200440,
            0xe85fffff,
            0xe9400840,
        ] {
            let insn = decode(word, ADDRESS, FeatureSet::ALL);
            assert_ne!(insn.code(), Code::Invalid, "{word:#010x} should decode");
            assert_eq!(encode(&insn), Ok(word), "round-trip {word:#010x}");
        }
        // Gated off when FEAT_THE is absent.
        let off = FeatureSet {
            features0: FeatureSet::ALL.features0 & !(1u64 << (Feature::The as u32)),
            features1: FeatureSet::ALL.features1,
        };
        for &word in &[0xe9400440u32, 0xe8400440, 0xe9c10440] {
            assert_eq!(decode(word, ADDRESS, off).code(), Code::Invalid);
        }
    }

    #[test]
    fn rcpc3_stlr_ldapr_writeback() {
        // STLR pre-index / LDAPR post-index with implicit datasize displacement.
        assert_dis(0x99800820, "stlr    w0, [x1, #-0x4]!");
        assert_dis(0xd9800820, "stlr    x0, [x1, #-0x8]!");
        assert_dis(0x99c00820, "ldapr   w0, [x1], #0x4");
        assert_dis(0xd9c00820, "ldapr   x0, [x1], #0x8");
        // Reserved imm9 (bits<20:12>) != 0 -> unallocated.
        assert_dis(0x99801820, "");
    }

    #[test]
    fn rcpc3_feature_gated_and_roundtrip() {
        use crate::encode::encode;
        // Round-trip: decode -> encode reproduces the word; re-decode equal code.
        for &word in &[
            0x1d404820u32,
            0x1dc04820,
            0x9d004820,
            0x99411840,
            0xd9410840,
            0xd9010840,
            0x99421860,
            0x99800820,
            0xd9c00820,
        ] {
            let insn = decode(word, ADDRESS, FeatureSet::ALL);
            assert_ne!(insn.code(), Code::Invalid, "{word:#010x} should decode");
            assert_eq!(encode(&insn), Ok(word), "round-trip {word:#010x}");
        }
        // Gated off when FEAT_LRCPC3 is absent.
        let off = FeatureSet {
            features0: FeatureSet::ALL.features0 & !(1u64 << (Feature::Rcpc3 as u32)),
            features1: FeatureSet::ALL.features1,
        };
        for &word in &[0x1d404820u32, 0x99411840, 0x99800820] {
            assert_eq!(decode(word, ADDRESS, off).code(), Code::Invalid);
        }
    }

    #[test]
    fn exclusive_and_ordered() {
        assert_dis(0xC8572629, "ldxr    x9, [x17]");
        assert_dis(0xC806455E, "stxr    w6, x30, [x10]");
        assert_dis(0xC8677530, "ldxp    x16, x29, [x9]");
        assert_dis(0xC82E295F, "stxp    w14, xzr, x10, [x10]");
        assert_dis(0xC8DFFD72, "ldar    x18, [x11]");
        assert_dis(0xC88DD332, "stlr    x18, [x25]");
        assert_dis(0xC8C124B7, "ldlar   x23, [x5]");
    }

    #[test]
    fn lse_atomics_and_cas() {
        assert_dis(0xC8BB7E15, "cas     x27, x21, [x16]");
        assert_dis(0x48227F0A, "casp    x2, x3, x10, x11, [x24]");
        assert_dis(0x88EDFC7E, "casal   w13, w30, [x3]");
        assert_dis(0xF82400B8, "ldadd   x4, x24, [x5]");
        assert_dis(0xF8E00025, "ldaddal x0, x5, [x1]");
        assert_dis(0xF82183F0, "swp     x1, x16, [sp]");
        // ST<op> alias when Rt==31 && A==0.
        assert_dis(0xF83F021F, "stadd   xzr, [x16]");
        // LDAPR (FEAT_LRCPC, decodes without LSE).
        assert_dis(0xF8BFC058, "ldapr   x24, [x2]");
    }

    #[test]
    fn pac_ldapstl_and_tags() {
        // Pointer-authenticated loads.
        assert_dis(0xF8246596, "ldraa   x22, [x12, #0x230]");
        assert_dis(0xF8AC1C5A, "ldrab   x26, [x2, #0x608]!");
        // RCpc unscaled.
        assert_dis(0xD94CC1FF, "ldapur  xzr, [x15, #0xcc]");
        assert_dis(0x999D5150, "ldapursw x16, [x10, #-0x2b]");
        // Memory tagging.
        assert_dis(0xD9A55BC6, "st2g    x6, [x30, #0x550]");
        assert_dis(0xD938D482, "stg     x2, [x4], #-0x730");
        assert_dis(0xD96583D1, "ldg     x17, [x30, #0x580]");
        assert_dis(0xD92003C5, "stzgm   x5, [x30]");
        assert_dis(0x691FC8DC, "stgp    x28, x18, [x6, #0x3f0]");
    }

    #[test]
    fn pac_gated_off_without_feature() {
        // Without FEAT_PAuth the LDRAA encoding is not admitted.
        let insn = decode(0xF8246596, ADDRESS, FeatureSet::BASE.with(Feature::Lse));
        assert!(insn.is_invalid(), "LDRAA must be gated by FEAT_PAuth");
    }

    #[test]
    fn simd_ldst_multiple_structures() {
        // No-offset, post-index immediate (= transferred bytes), post-index reg.
        assert_dis(0x0C407FEE, "ld1     {v14.1d}, [sp]");
        assert_dis(0x4C4073AC, "ld1     {v12.16b}, [x29]");
        assert_dis(0x4CDF70DD, "ld1     {v29.16b}, [x6], #0x10");
        assert_dis(0x4CDFA129, "ld1     {v9.16b, v10.16b}, [x9], #0x20");
        // Register-list wraparound past v31 and 4-register .8b post-imm.
        assert_dis(0x0CDF22DE, "ld1     {v30.8b, v31.8b, v0.8b, v1.8b}, [x22], #0x20");
        // LD2/LD3/LD4 + ST4 multiple, no-offset.
        assert_dis(0x4C000080, "st4     {v0.16b, v1.16b, v2.16b, v3.16b}, [x4]");
    }

    #[test]
    fn simd_ldst_single_structure() {
        // Single-lane forms use the truncated element suffix and a `[index]`.
        assert_dis(0x4D9F1CBC, "st1     {v28.b}[15], [x5], #0x1");
        assert_dis(0x4DCD8712, "ld1     {v18.d}[1], [x24], x13");
        assert_dis(0x0DD7A0B2, "ld3     {v18.s, v19.s, v20.s}[0], [x5], x23");
        assert_dis(0x0D80A4C5, "st3     {v5.d, v6.d, v7.d}[0], [x6], x0");
        // LD4 single (.s, four-register) post-index immediate (= 4*4 bytes).
        assert_dis(0x0DBFB139, "st4     {v25.s, v26.s, v27.s, v28.s}[1], [x9], #0x10");
    }

    #[test]
    fn simd_ld_replicate() {
        // LDnR use the full arrangement; post-imm = count * element-bytes.
        assert_dis(0x0DDFC4A5, "ld1r    {v5.4h}, [x5], #0x2");
        assert_dis(0x0DFFE507, "ld4r    {v7.4h, v8.4h, v9.4h, v10.4h}, [x8], #0x8");
        assert_dis(0x4DF8EED7, "ld4r    {v23.2d, v24.2d, v25.2d, v26.2d}, [x22], x24");
    }

    #[test]
    fn simd_ldst_store_replicate_is_invalid() {
        // The replicate opcode (scale==0b11) has no store form: L==0 must be
        // left invalid. Take a valid LD1R and clear its L bit (bit 22).
        let ld1r = 0x0DDFC4A5u32;
        let st1r = ld1r & !(1 << 22);
        let insn = decode(st1r, ADDRESS, FeatureSet::ALL);
        assert!(insn.is_invalid(), "store-replicate must be invalid: {st1r:#010x}");
    }

    #[test]
    fn ls64_atomic_64byte_ops() {
        // LLVM-confirmed (+ls64) encodings.
        assert_dis(0xF83FD020, "ld64b   x0, [x1]");
        assert_dis(0xF83F9020, "st64b   x0, [x1]");
        assert_dis(0xF822B020, "st64bv  x2, x0, [x1]");
        assert_dis(0xF822A020, "st64bv0 x2, x0, [x1]");
    }

    #[test]
    fn ls64_gated_off_without_feature() {
        // Without FEAT_LS64 these encodings are not admitted (and are not LSE).
        for word in [0xF83FD020u32, 0xF83F9020, 0xF822B020, 0xF822A020] {
            let insn = decode(word, ADDRESS, FeatureSet::BASE.with(Feature::Lse));
            assert!(insn.is_invalid(), "LS64 must be gated by FEAT_LS64: {word:#010x}");
        }
    }

    #[test]
    fn never_panics_on_ldst_space() {
        // Sweep the four load/store op0 nibbles (word<27:24> high bits) across the
        // full low 24 bits. The decoder must be total and panic-free for both the
        // all-features and base feature sets.
        for hi in [0x08u32, 0x0C, 0x18, 0x1C, 0x38, 0x3C, 0x48, 0x4C, 0x88, 0x8C, 0xC8, 0xCC, 0xF8, 0xFC] {
            for lo in 0..=0xffffu32 {
                let word = (hi << 24) | (lo << 4) | (lo & 0xf);
                let _ = decode(word, ADDRESS, FeatureSet::ALL);
                let _ = decode(word, ADDRESS, FeatureSet::BASE);
            }
        }
    }
}
