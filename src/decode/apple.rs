//! Apple-proprietary AArch64 instructions (IMPLEMENTATION DEFINED).
//!
//! These are **not** part of the Arm architecture and are **not** decoded by
//! LLVM. They live in the reserved `op0 == 0b0000` encoding region (dispatched
//! here from [`crate::decode::decode_reserved`], which has already split off the
//! permanently-undefined `UDF` space and the SME `word<31> == 1` space). The
//! encodings are reverse-engineered from public Apple-silicon research:
//!
//! * **Apple AMX** — the matrix coprocessor on pre-M4 Apple silicon (A14–A17,
//!   M1–M3). Each instruction is `0x0020_1000 | (op << 5) | operand`, with the
//!   5-bit `op` in `word<9:5>` (0..=22) and a 5-bit `operand` in `word<4:0>`.
//!   For most ops the operand is a general-purpose source/destination `Xn`; for
//!   `op == 17` it selects `set` (0) / `clr` (1). Reference: <https://github.com/corsix/amx>.
//!   M4 and later replaced AMX with the Arm-standard SME.
//! * **Apple GXF** — Guarded Execution Feature. `GEXIT` (`0x0020_1400`) and
//!   `GENTER #imm5` (`0x0020_1420 | imm5`) enter/exit Apple's lateral "guarded"
//!   exception levels. They share the AMX base word but set `word<10>`.
//!
//! Both families are gated by a runtime [`Feature`] ([`Feature::AppleAmx`] /
//! [`Feature::Gxf`]); with neither enabled the word is left invalid, exactly as
//! before, so enabling Arm-only decoding never surfaces an Apple opcode.

use crate::features::{Feature, FeatureSet};
use crate::instruction::Instruction;
use crate::mnemonic::Code;
use crate::operand::Operand;
use crate::register::{gp_register, RegWidth};

/// A plain 64-bit `Xn` operand (register `31` renders as `xzr`, matching the
/// general-register convention; AMX takes a raw GP register number 0..=31).
#[inline]
fn xn(n: u32) -> Operand {
    Operand::Reg {
        reg: gp_register(false, RegWidth::X64, (n & 0x1f) as u8),
        arr: None,
        lane: None,
        shift: None,
        extend: None,
        pred: None,
    }
}

/// Decode an Apple IMPLEMENTATION-DEFINED instruction in the reserved region.
///
/// Recognizes the AMX (`0x0020_10xx`) and GXF (`0x0020_14xx`) clusters; leaves
/// `out` invalid for anything else (or when the gating feature is off).
#[inline]
pub(crate) fn decode_apple(word: u32, features: FeatureSet, out: &mut Instruction) {
    // AMX cluster: word<31:10> == 0b00..0_1000_0000_01 (i.e. base 0x0020_1000,
    // `word<10> == 0`). `op` = word<9:5>, `operand` = word<4:0>.
    if (word & 0xffff_fc00) == 0x0020_1000 {
        if features.has(Feature::AppleAmx) {
            decode_amx((word >> 5) & 0x1f, word & 0x1f, out);
        }
        return;
    }

    // GXF cluster: same base with `word<10> == 1` (base 0x0020_1400). The
    // sub-opcode in word<9:5> selects gexit (0) / genter (1); other values are
    // unallocated.
    if (word & 0xffff_fc00) == 0x0020_1400 && features.has(Feature::Gxf) {
        decode_gxf((word >> 5) & 0x1f, word & 0x1f, out);
    }
}

/// Apple AMX: map the 5-bit `op` (and `operand` for the GP-register / set-clr
/// distinction) to a [`Code`]. Ops 23..=31 are unallocated and left invalid.
#[inline]
fn decode_amx(op: u32, operand: u32, out: &mut Instruction) {
    // op 17 is the only operand-less form: it toggles AMX state by `operand`.
    if op == 17 {
        match operand {
            0 => out.set(Code::AmxSet),
            1 => out.set(Code::AmxClr),
            // Other operand values for op 17 are not architected; leave invalid.
            _ => {}
        }
        return;
    }

    let code = match op {
        0 => Code::AmxLdx,
        1 => Code::AmxLdy,
        2 => Code::AmxStx,
        3 => Code::AmxSty,
        4 => Code::AmxLdz,
        5 => Code::AmxStz,
        6 => Code::AmxLdzi,
        7 => Code::AmxStzi,
        8 => Code::AmxExtrx,
        9 => Code::AmxExtry,
        10 => Code::AmxFma64,
        11 => Code::AmxFms64,
        12 => Code::AmxFma32,
        13 => Code::AmxFms32,
        14 => Code::AmxMac16,
        15 => Code::AmxFma16,
        16 => Code::AmxFms16,
        18 => Code::AmxVecint,
        19 => Code::AmxVecfp,
        20 => Code::AmxMatint,
        21 => Code::AmxMatfp,
        22 => Code::AmxGenlut,
        _ => return,
    };
    out.set(code);
    out.push_operand(xn(operand));
}

/// Apple GXF: `word<9:5> == 0` → `GEXIT` (operand must be 0); `== 1` →
/// `GENTER #imm5` (operand is the 5-bit immediate).
#[inline]
fn decode_gxf(sub: u32, imm5: u32, out: &mut Instruction) {
    match sub {
        0 => {
            // GEXIT is the single fixed word 0x0020_1400.
            if imm5 == 0 {
                out.set(Code::Gexit);
            }
        }
        1 => {
            out.set(Code::Genter);
            out.push_operand(Operand::ImmUnsigned(imm5 as u64));
        }
        _ => {}
    }
}
