//! Data Processing -- Scalar Floating-Point and Advanced SIMD (ARM ARM C4.1.97).
//!
//! Hand-written decoder dispatched here from [`crate::decode::decode_into`] when
//! `op0 = word<28:25>` selects `x111`. This is the floating-point and NEON
//! group; per-extension detail is gated by [`crate::features::Feature`] so a
//! base-only [`FeatureSet`] omits the corresponding forms.
//!
//! The group is sub-classed exactly as the ARM ARM C4.1.97 classification table,
//! using the four discriminator fields:
//!
//! | field | bits | meaning |
//! |-|-|-|
//! | `op0` | `word<31:28>` | top opcode (M / `0` / U / size-ish) |
//! | `op1` | `word<24:23>` | sub-class selector |
//! | `op2` | `word<22:19>` | sub-class selector |
//! | `op3` | `word<18:10>` | sub-class selector |
//!
//! Three sub-decoders own the leaves:
//!
//! * [`scalar_fp`] — the scalar floating-point data-processing encodings
//!   (`(op0&5)==1`, i.e. `word<28>==1 && word<30>==0`): conversions to/from
//!   integer and fixed-point, 1/2/3-source FP data-processing, compares,
//!   conditional compare/select and the FP immediate move.
//! * [`simd_arith`] — Advanced SIMD arithmetic (three-same / three-different /
//!   pairwise / across-lanes / scalar variants). Currently a compiling stub.
//! * [`simd_data`] — Advanced SIMD data-movement (permute / table / copy /
//!   modified-immediate / shift-by-immediate / extract). Currently a stub.
//!
//! Modified-immediate helpers live in [`crate::decode::bits`]
//! ([`adv_simd_expand_imm`](crate::decode::bits::adv_simd_expand_imm),
//! [`vfp_expand_imm`](crate::decode::bits::vfp_expand_imm)).

#[cfg(feature = "crypto")]
pub mod crypto;
pub mod scalar_fp;
pub mod simd_arith;
pub mod simd_data;

use crate::decode::bits::{bit, bits};
use crate::features::FeatureSet;
use crate::instruction::Instruction;

/// Decode a Scalar-FP / Advanced-SIMD instruction into `out`.
///
/// `ip` is accepted for signature uniformity (this group has no PC-relative
/// forms). Pure, total and panic-free for every input; unallocated encodings
/// leave `out` as the invalid default. Extension-specific encodings are only
/// produced when accepted at runtime (`features`).
#[inline]
pub fn decode(word: u32, ip: u64, features: FeatureSet, out: &mut Instruction) {
    let _ = ip;

    // ARM ARM C4.1.97 discriminator fields for the SIMD&FP top group.
    let op0 = bits(word, 28, 4); // word<31:28>
    let op1 = bits(word, 23, 2); // word<24:23>
    let op2 = bits(word, 19, 4); // word<22:19>
    let op3 = bits(word, 10, 9); // word<18:10>

    // ---- Scalar floating-point data-processing rows: (op0 & 5) == 1. ------
    // `op0 & 5` isolates word<28> (bit 0) and word<30> (bit 2): the scalar-FP
    // rows require word<28>==1 and word<30>==0.
    if (op0 & 0b0101) == 0b0001 {
        if (op1 & 0b10) == 0 {
            // op1<1> == 0 : the non-DP3 floating-point rows, split on op2/op3.
            if (op2 & 0b0100) == 0 {
                // Conversion between floating-point and fixed-point.
                scalar_fp::decode_float2fix(word, features, out);
                return;
            }
            // (op2 & 4) == 4 from here: split on op3 (word<18:10>).
            if (op3 & 0b0_0011_1111) == 0 {
                // Conversion between floating-point and integer.
                scalar_fp::decode_float2int(word, features, out);
                return;
            }
            if (op3 & 0b0_0001_1111) == 0b0_0001_0000 {
                // Floating-point data-processing (1 source).
                scalar_fp::decode_floatdp1(word, features, out);
                return;
            }
            if (op3 & 0b0_0000_1111) == 0b0_0000_1000 {
                // Floating-point compare.
                scalar_fp::decode_floatcmp(word, features, out);
                return;
            }
            if (op3 & 0b0_0000_0111) == 0b0_0000_0100 {
                // Floating-point immediate.
                scalar_fp::decode_floatimm(word, features, out);
                return;
            }
            if (op3 & 0b0_0000_0011) == 0b0_0000_0001 {
                // Floating-point conditional compare.
                scalar_fp::decode_floatccmp(word, features, out);
                return;
            }
            if (op3 & 0b0_0000_0011) == 0b0_0000_0010 {
                // Floating-point data-processing (2 source).
                scalar_fp::decode_floatdp2(word, features, out);
                return;
            }
            if (op3 & 0b0_0000_0011) == 0b0_0000_0011 {
                // Floating-point conditional select.
                scalar_fp::decode_floatsel(word, features, out);
                return;
            }
            // Any remaining op3 in this region is UNALLOCATED.
            return;
        }
        // op1<1> == 1 : Floating-point data-processing (3 source).
        scalar_fp::decode_floatdp3(word, features, out);
        return;
    }

    // ---- Advanced SIMD cryptographic instructions (feature-gated). --------
    // These eight encoding classes (AES / SHA1 / SHA256 / SHA512 / SM3 / SM4 /
    // EOR3 / BCAX / RAX1 / XAR) sit at `op0 ∈ {4, 5, 12}` within the SIMD&FP
    // group; intercept them before the general SIMD arithmetic/data routing so
    // they are never misclassified. `crypto::decode` claims a word only when it
    // matches a crypto class AND `Feature::Crypto` is accepted.
    #[cfg(feature = "crypto")]
    if crypto::decode(word, op0, op1, op2, op3, features, out) {
        return;
    }

    // ---- Advanced SIMD rows. ----------------------------------------------
    // The detailed SIMD classification is owned by the two SIMD sub-decoders
    // (filled in by later agents). Route by the coarse scalar-vs-vector bit
    // (`word<28>`, op0<0>) so each sub-decoder sees its own slice; both are
    // currently compiling stubs that leave `out` invalid.
    let _ = bit(word, 28);
    simd_arith::decode(word, ip, features, out);
    if out.is_invalid() {
        simd_data::decode(word, ip, features, out);
    }
}
