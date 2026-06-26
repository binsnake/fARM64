//! SVE / SVE2 encoding group (`op0 = word<28:25> = 0b0010`) — hand-written.
//!
//! Transcribed from the *ARM Architecture Reference Manual* SVE encoding index.
//! The SVE space is enormous (~34% of the differential corpus); this module is
//! the top-level classifier that dispatches to the family submodules:
//!
//! * [`sve_int`] — integer arithmetic / logical / shift / reduction / INDEX /
//!   INC-DEC / CNT / compare-immediate / MOV-DUP-CPY / DOT and the SVE2 integer
//!   multiply-add and widening families.
//! * [`sve_perm`] — permute / predicate-logical / table / unpack (stub for now).
//! * [`sve_fp`] — floating-point (stub for now).
//! * [`sve_mem`] — loads / stores / prefetch (stub for now).
//!
//! Dispatch key. The eight SVE quadrants are selected by `word<31:29>`:
//!
//! | `word<31:29>` | Area |
//! |-|-|
//! | `000` | SVE integer/shift/misc + permute (op-dependent) |
//! | `001` | SVE integer (imm), compare, predicate, INC/DEC-by-predicate |
//! | `010` | SVE2 integer multiply-add / DOT / widening |
//! | `011` | SVE floating-point |
//! | `100`..`111` | SVE memory (load / store / prefetch) |
//!
//! Within an integer quadrant the precise family is selected by the inner
//! sub-fields; that fine-grained work lives in [`sve_int`]. Every path is total
//! and panic-free, leaving [`crate::mnemonic::Code::Invalid`] for unallocated
//! encodings.

pub mod sve_fp;
pub mod sve_int;
pub mod sve_lut;
pub mod sve_mem;
pub mod sve_perm;

use crate::decode::bits::bits;
use crate::features::FeatureSet;
use crate::instruction::Instruction;

/// Decode a single SVE/SVE2 instruction `word` at `ip` into `out`.
///
/// Called from [`crate::decode::decode_into`] (under `#[cfg(feature = "sve")]`)
/// once the top-level `op0` has selected the SVE group. Dispatches on
/// `word<31:29>` to the family submodules; integer quadrants try [`sve_int`]
/// first and fall back to [`sve_perm`] for the permute/predicate leaves that
/// share the quadrant. Total and panic-free for all inputs.
#[inline]
pub fn decode(word: u32, ip: u64, features: FeatureSet, out: &mut Instruction) {
    // word<31:29> selects the SVE quadrant.
    match bits(word, 29, 3) {
        // Integer / shift / misc and the permute/predicate leaves share 000/001.
        // Try the integer decoder first; if it declines (leaves Invalid), fall
        // back to the permute/predicate decoder, and finally to the handful of
        // floating-point leaves that share these quadrants (FABS/FNEG/FEXPA/
        // FTSSEL in 0x04, FCPY in 0x05, FDUP in 0x25).
        0b000 | 0b001 => {
            sve_int::decode(word, ip, features, out);
            if out.is_invalid() {
                sve_perm::decode(word, ip, features, out);
            }
            if out.is_invalid() {
                match bits(word, 24, 8) {
                    0x04 => sve_fp::decode_fp_misc_04(word, features, out),
                    0x05 => sve_fp::decode_fcpy_05(word, features, out),
                    0x25 => sve_fp::decode_fdup_25(word, features, out),
                    _ => {}
                }
            }
        }
        // SVE2 integer multiply-add / DOT / widening (and a few permute leaves),
        // plus the FEAT_LUT lookup-table reads (top byte 0x45).
        0b010 => {
            sve_int::decode(word, ip, features, out);
            if out.is_invalid() {
                sve_perm::decode(word, ip, features, out);
            }
            if out.is_invalid() {
                sve_lut::decode(word, features, out);
            }
        }
        // SVE floating-point.
        0b011 => sve_fp::decode(word, ip, features, out),
        // SVE memory: contiguous / gather / scatter loads, stores, prefetch.
        0b100..=0b111 => sve_mem::decode(word, ip, features, out),
        // word<31:29> is 3 bits; fully covered above.
        _ => {}
    }
}
