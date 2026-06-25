//! Inverse immediate helpers for the encoder — the exact inverses of the
//! shared decode pseudocode in [`crate::decode::bits`].
//!
//! These reconstruct the raw bit-fields of an A64 instruction from the
//! *semantic* value the decoder produced. Every function is pure, total, and
//! panic-free, and returns `None` for any value that has no valid encoding (so
//! the encoder can stay total and never panic).

use crate::decode::bits::{adv_simd_expand_imm, decode_bit_masks, vfp_expand_imm};

/// `EncodeBitMasks` — the inverse of
/// [`decode_bit_masks`](crate::decode::bits::decode_bit_masks).
///
/// Given a logical-immediate `value` and the GP `datasize` (32 or 64), find the
/// `(N, immr, imms)` triple that the bit-mask decoder would expand back into
/// exactly `value`. Returns `None` when `value` is not a valid A64 logical
/// (bitmask) immediate (all-zeros, all-ones, or any pattern that is not a
/// rotated run of ones replicated across an element size that divides
/// `datasize`).
///
/// The algorithm mirrors the ARM ARM `DecodeBitMasks` structure in reverse:
///
/// 1. The value must be periodic with some power-of-two element size
///    `esize ∈ {2,4,8,16,32,64}` that divides `datasize`. The smallest such
///    period is the element size the decoder would have chosen.
/// 2. Within one element the set bits must form a single rotated contiguous run
///    of `S+1` ones (a "run of ones" possibly wrapped around the element). The
///    rotation gives `immr` (R), the population count gives `imms` (S).
/// 3. `N` and the high bits of `imms` encode `len = log2(esize)` via the ARM
///    ARM `immN:NOT(imms<5:0>)` packing.
#[inline]
pub fn encode_bit_masks(value: u64, datasize: u32) -> Option<(u32, u32, u32)> {
    let datasize = if datasize == 32 { 32 } else { 64 };

    // Mask the value to the data size we are working in.
    let value = if datasize == 32 {
        value & 0xffff_ffff
    } else {
        value
    };

    // All-zeros and all-ones are not representable as logical immediates.
    let full = if datasize == 64 {
        u64::MAX
    } else {
        0xffff_ffff
    };
    if value == 0 || value == full {
        return None;
    }

    // (1) Find the smallest power-of-two element size whose period replicates the
    // whole value. Candidate esizes are 2,4,8,16,32,64 (must divide datasize).
    let mut esize = 2u32;
    while esize <= datasize {
        if datasize % esize == 0 {
            let elem = value & mask(esize);
            if replicates(elem, esize, datasize, value) {
                if let Some((immr, imms)) = encode_element(elem, esize) {
                    // (3) Pack len = log2(esize) into immN:imms<5:?> per the ARM
                    // ARM. len = HighestSetBit(immN:NOT(imms)). We rebuild the
                    // 7-bit `immN:NOT(imms<5:0>)` so its top set bit is at `len`.
                    let len = esize.trailing_zeros(); // log2(esize), 1..=6
                    let (n, imms_field) = pack_n_imms(len, imms);
                    // Final guard: confirm the chosen fields round-trip exactly.
                    if let Some(m) = decode_bit_masks(n, imms_field, immr, true, datasize) {
                        let got = if datasize == 32 {
                            m.wmask & 0xffff_ffff
                        } else {
                            m.wmask
                        };
                        if got == value {
                            return Some((n, immr, imms_field));
                        }
                    }
                }
            }
        }
        esize <<= 1;
    }
    None
}

/// Low-`width` bit mask (`width` in `1..=64`).
#[inline]
const fn mask(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

/// `true` when the low `esize` bits (`elem`) replicated across `datasize` equal
/// `value` exactly (so `esize` is a true period of `value`).
#[inline]
fn replicates(elem: u64, esize: u32, datasize: u32, value: u64) -> bool {
    let mut acc = 0u64;
    let mut shift = 0u32;
    while shift < datasize {
        acc |= elem << shift;
        shift += esize;
    }
    acc == value
}

/// Encode one element value (`elem`, occupying the low `esize` bits) as
/// `(immr, imms)` where the element is `ROR(Ones(S+1), R)` within `esize` bits.
///
/// Returns `None` if `elem` is not a single rotated contiguous run of ones
/// (i.e. it is not a valid logical-immediate element). The decoder computes
/// `welem = Ones(S+1)` then `wmask = ROR(welem, R)`, so we recover `S+1` as the
/// population count and `R` as the rotation that brings the run down to the low
/// bits.
#[inline]
fn encode_element(elem: u64, esize: u32) -> Option<(u32, u32)> {
    let elem = elem & mask(esize);
    // All-zeros / all-ones elements are rejected (handled by the caller too, but
    // also reachable for sub-elements that happen to be uniform).
    if elem == 0 || elem == mask(esize) {
        return None;
    }
    let count = elem.count_ones(); // = S + 1
    // To recover R: ROR(Ones(count), R) == elem. Ones(count) is `count` ones in
    // the low bits. Find the rotation R such that rotating those ones right by R
    // (within esize) yields `elem`. R is the rotation amount; equivalently the
    // run of ones in `elem` starts at bit position `esize - R` (mod esize).
    let ones = mask(count);
    let mut r = 0u32;
    while r < esize {
        if ror_elem(ones, r, esize) == elem {
            return Some((r, count - 1));
        }
        r += 1;
    }
    None
}

/// Rotate the low `width` bits of `x` right by `shift` (`shift < width`).
#[inline]
fn ror_elem(x: u64, shift: u32, width: u32) -> u64 {
    let x = x & mask(width);
    if shift == 0 {
        return x;
    }
    let lo = x >> shift;
    let hi = x << (width - shift);
    (lo | hi) & mask(width)
}

/// Pack `len = log2(esize)` and the `S` value into the `(immN, imms_field)` the
/// ARM ARM expects: `len = HighestSetBit(immN : NOT(imms<5:0>))`.
///
/// For `len == 6` (esize 64) the high bit lands in `immN`; for `len < 6` the
/// top set bit lands inside `NOT(imms)`, which fixes the high bits of `imms`
/// above `len` to a pattern of ones (so those bits of `imms` are zero). The low
/// `len` bits of `imms` carry `S`.
#[inline]
fn pack_n_imms(len: u32, s: u32) -> (u32, u32) {
    if len == 6 {
        // immN = 1, imms<5:0> = S (NOT(imms) has no set bit at position 6, so the
        // highest set bit of the 7-bit concat is bit 6 = immN).
        (1, s & 0x3f)
    } else {
        // immN = 0. NOT(imms<5:0>) must have its highest set bit at position
        // `len`, i.e. bits <5:len+1> are 0 in NOT(imms) -> 1 in imms, bit <len>
        // is 1 in NOT(imms) -> 0 in imms, and the low `len` bits are S.
        // imms<5:len> = 0b1...10 (a leading run of ones then a single zero at
        // bit `len`); imms<len-1:0> = S.
        let high_ones = ((0x3f) >> (len + 1)) << (len + 1); // bits <5:len+1> set
        let imms = high_ones | (s & mask(len) as u32);
        (0, imms & 0x3f)
    }
}

// ---------------------------------------------------------------------------
// Inverse VFP / AdvSIMD floating-point immediate.
// ---------------------------------------------------------------------------

/// Inverse of [`vfp_expand_imm`](crate::decode::bits::vfp_expand_imm): find the
/// 8-bit `imm8` whose expansion at element width `n` (16/32/64) equals the value
/// represented by the `f32` the decoder stored.
///
/// The decoder materialises the FP immediate as an `f32` (via the per-precision
/// reinterpretation in `scalar_fp`), so the encoder takes that same `f32` and
/// searches the 256-entry `imm8` space, comparing on the value the *decoder*
/// would have produced. Returns `None` if no `imm8` reproduces it (the value was
/// not an 8-bit-representable FP immediate).
#[inline]
pub fn encode_vfp_imm(value: f32, n: u32) -> Option<u32> {
    let mut imm8 = 0u32;
    while imm8 < 256 {
        let bits_val = vfp_expand_imm(imm8, n);
        let f = match n {
            16 => f16_bits_to_f32_enc(bits_val as u16),
            32 => f32::from_bits(bits_val as u32),
            64 => f64::from_bits(bits_val) as f32,
            _ => return None,
        };
        // Bit-exact comparison (handles signed zero correctly via to_bits).
        if f.to_bits() == value.to_bits() {
            return Some(imm8);
        }
        imm8 += 1;
    }
    None
}

/// Local copy of the half-precision bit-pattern -> `f32` conversion, matching
/// `scalar_fp::f16_bits_to_f32` exactly so the inverse search compares against
/// the same value the decoder produced (only the well-behaved normal / signed-
/// zero values out of `vfp_expand_imm` occur here).
#[inline]
fn f16_bits_to_f32_enc(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut e = -1i32;
            let mut f = frac;
            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3ff;
            let exp32 = (e + 127 - 15) as u32;
            (sign << 31) | (exp32 << 23) | (f << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        let exp32 = exp + (127 - 15);
        (sign << 31) | (exp32 << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

// ---------------------------------------------------------------------------
// Inverse AdvSIMD modified immediate (64-bit MOVI forms).
// ---------------------------------------------------------------------------

/// Inverse of [`adv_simd_expand_imm`](crate::decode::bits::adv_simd_expand_imm)
/// for the 64-bit per-byte MOVI form (`cmode == 0b1110`, `op == 1`): given the
/// expanded 64-bit `value` (each byte either `0x00` or `0xff`), recover the
/// 8-bit `imm8` whose bit `i` selects byte `i`. Returns `None` if any byte is
/// not uniformly all-zero or all-ones.
#[inline]
pub fn encode_advsimd_movi64(value: u64) -> Option<u32> {
    let mut imm8 = 0u32;
    let mut i = 0u32;
    while i < 8 {
        let byte = (value >> (i * 8)) & 0xff;
        if byte == 0xff {
            imm8 |= 1 << i;
        } else if byte != 0 {
            return None;
        }
        i += 1;
    }
    // Confirm round-trip exactly (defensive).
    if adv_simd_expand_imm(1, 0b1110, imm8) == value {
        Some(imm8)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::bits::decode_bit_masks;

    /// `encode_bit_masks` must be the exact inverse of `decode_bit_masks` for
    /// every valid `(N, imms, immr)` field combination, on both data sizes.
    #[test]
    fn round_trip_against_decode() {
        for &datasize in &[32u32, 64u32] {
            for n in 0u32..=1 {
                // sf==0 (datasize 32) requires N==0.
                if datasize == 32 && n == 1 {
                    continue;
                }
                for imms in 0u32..64 {
                    for immr in 0u32..64 {
                        let Some(m) = decode_bit_masks(n, imms, immr, true, datasize) else {
                            continue;
                        };
                        let value = if datasize == 32 {
                            m.wmask & 0xffff_ffff
                        } else {
                            m.wmask
                        };
                        let enc = encode_bit_masks(value, datasize)
                            .unwrap_or_else(|| panic!(
                                "no encoding for value={value:#x} ds={datasize} (n={n} imms={imms} immr={immr})"
                            ));
                        // The recovered fields must decode back to the SAME value
                        // (the canonical field triple may differ, but must be
                        // value-equivalent).
                        let (rn, rimmr, rimms) = enc;
                        let m2 = decode_bit_masks(rn, rimms, rimmr, true, datasize)
                            .expect("recovered fields must decode");
                        let v2 = if datasize == 32 {
                            m2.wmask & 0xffff_ffff
                        } else {
                            m2.wmask
                        };
                        assert_eq!(
                            v2, value,
                            "round-trip value mismatch: in fields n={n} imms={imms} immr={immr} -> value={value:#x}, out fields n={rn} imms={rimms} immr={rimmr} -> {v2:#x}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_trivial_values() {
        assert_eq!(encode_bit_masks(0, 64), None);
        assert_eq!(encode_bit_masks(u64::MAX, 64), None);
        assert_eq!(encode_bit_masks(0, 32), None);
        assert_eq!(encode_bit_masks(0xffff_ffff, 32), None);
    }

    #[test]
    fn known_values() {
        // AND x0, x0, #0xff : N=1, immr=0, imms=7.
        let (n, immr, imms) = encode_bit_masks(0xff, 64).expect("0xff valid");
        let m = decode_bit_masks(n, imms, immr, true, 64).unwrap();
        assert_eq!(m.wmask, 0xff);

        // 0x5555_5555 (32-bit period-2 pattern).
        let (n, immr, imms) = encode_bit_masks(0x5555_5555, 32).expect("valid");
        let m = decode_bit_masks(n, imms, immr, true, 32).unwrap();
        assert_eq!(m.wmask & 0xffff_ffff, 0x5555_5555);

        // 0x0001_0001_0001_0001 (period-16 single bit).
        let v = 0x0001_0001_0001_0001u64;
        let (n, immr, imms) = encode_bit_masks(v, 64).expect("valid");
        let m = decode_bit_masks(n, imms, immr, true, 64).unwrap();
        assert_eq!(m.wmask, v);
    }
}
