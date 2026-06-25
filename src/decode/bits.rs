//! Shared bit-field extraction and ARM pseudocode helpers — hand-written.
//!
//! Every function here is transcribed from the *ARM Architecture Reference
//! Manual* (the "ARM ARM") shared-pseudocode library, not ported from any other
//! decoder. All are pure integer logic: no heap, no panics-as-control-flow, and
//! no floating-point *arithmetic* (FP immediates are produced as raw bit
//! patterns and bit-cast by the formatter at render time).
//!
//! Intentional spec divergence policy: where another disassembler approximates
//! the spec (e.g. Binary Ninja's `DecodeBitMasks` `tmask == wmask` shortcut at
//! `pcode.c:88`), fARM64 implements the ARM ARM value and documents the
//! divergence at the call site rather than reproducing the approximation.

// ---------------------------------------------------------------------------
// Bit-field extraction primitives.
// ---------------------------------------------------------------------------

/// Extract `width` bits of `word` starting at bit `lsb` (zero-extended).
///
/// `width == 0` yields `0`; `lsb + width` must not exceed 32 (the A64 word
/// width). This is the ARM ARM `word<lsb+width-1:lsb>` slice.
#[inline]
pub const fn bits(word: u32, lsb: u32, width: u32) -> u32 {
    if width == 0 {
        return 0;
    }
    let mask = if width >= 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    };
    (word >> lsb) & mask
}

/// Extract a single bit (`0` or `1`) at position `pos` of `word`.
#[inline]
pub const fn bit(word: u32, pos: u32) -> u32 {
    (word >> pos) & 1
}

/// Extract `width` bits of a 64-bit `value` starting at `lsb` (zero-extended).
///
/// The 64-bit analogue of [`bits`], used by pseudocode that operates on
/// already-widened values.
#[inline]
pub const fn bits64(value: u64, lsb: u32, width: u32) -> u64 {
    if width == 0 {
        return 0;
    }
    let mask = if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    (value >> lsb) & mask
}

// ---------------------------------------------------------------------------
// ARM ARM shared pseudocode (hand-written from the spec).
// ---------------------------------------------------------------------------

/// `SignExtend` — sign-extend the low `bits` of `value` to a signed `i64`.
///
/// ARM ARM shared pseudocode. `bits == 0` or `bits >= 64` returns the value
/// reinterpreted as signed without modification.
#[inline]
pub const fn sign_extend(value: u64, bits: u32) -> i64 {
    if bits == 0 || bits >= 64 {
        return value as i64;
    }
    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}

/// `HighestSetBit` — index of the most-significant set bit of `value`, or `-1`
/// if `value == 0`.
///
/// ARM ARM shared pseudocode (used by `DecodeBitMasks` and `CountLeadingZeros`).
#[inline]
pub const fn highest_set_bit(value: u64) -> i32 {
    if value == 0 {
        -1
    } else {
        63 - value.leading_zeros() as i32
    }
}

/// `Replicate` — replicate the low `from` bits of `value` to fill `to` bits.
///
/// ARM ARM shared pseudocode. `to` must be a positive multiple of `from`
/// (`1 <= from <= to <= 64`); out-of-range inputs return `value` unchanged so
/// the function is total.
#[inline]
pub const fn replicate(value: u64, from: u32, to: u32) -> u64 {
    if from == 0 || from > 64 || to == 0 || to > 64 || from > to {
        return value;
    }
    let unit = bits64(value, 0, from);
    let mut acc: u64 = 0;
    let mut shift = 0u32;
    while shift < to {
        acc |= unit << shift;
        shift += from;
    }
    bits64(acc, 0, to)
}

/// Result of [`decode_bit_masks`]: the work mask (`wmask`) and tell mask
/// (`tmask`).
///
/// Per the ARM ARM, `wmask` and `tmask` are computed *independently*; fARM64
/// follows the spec rather than the `tmask == wmask` shortcut taken by some
/// other decoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BitMasks {
    /// The logical-immediate work mask.
    pub wmask: u64,
    /// The tell mask (ARM ARM value, independent of `wmask`).
    pub tmask: u64,
}

/// `DecodeBitMasks` — decode an A64 logical (bitmask) immediate from
/// `(immN, imms, immr)` for datasize `m` (32 or 64).
///
/// Hand-written from the ARM ARM. `immediate` selects the logical-immediate
/// validity checks (the `imms == all-ones` reserved case). Returns `None` for
/// the reserved/UNALLOCATED encodings the ARM ARM rejects.
///
/// Divergence note: the ARM ARM computes `tmask` from `diff`/`tmask_and`
/// separately from `wmask`; fARM64 honours that. Do not collapse to
/// `tmask == wmask`.
#[inline]
pub fn decode_bit_masks(imm_n: u32, imms: u32, immr: u32, immediate: bool, m: u32) -> Option<BitMasks> {
    // Only the low 6 bits of imms/immr and the single bit immN are meaningful.
    let imms = imms & 0x3f;
    let immr = immr & 0x3f;
    let imm_n = imm_n & 1;

    // ARM ARM: len = HighestSetBit(immN:NOT(imms<5:0>)).
    // immN:NOT(imms) is a 7-bit value (immN in bit 6, the inverted 6-bit imms below).
    let concat = (imm_n << 6) | ((!imms) & 0x3f);
    let len = highest_set_bit(concat as u64);
    // len < 1 is the reserved/UNALLOCATED encoding.
    if len < 1 {
        return None;
    }
    let len = len as u32;

    // levels = ZeroExtend(Ones(len), 6).
    let levels = ones(len) as u32; // len <= 6 here, fits in 6 bits.

    // For logical immediates the all-ones field is reserved.
    if immediate && (imms & levels) == levels {
        return None;
    }

    let s = (imms & levels) as u64; // UInt(imms AND levels)
    let r = (immr & levels) as u64; // UInt(immr AND levels)
    let diff = s.wrapping_sub(r); // len-bit subtraction (we mask below)

    let esize = 1u32 << len; // 1 << len, in range 2..=64.
    // d = UInt(diff<len-1:0>).
    let d = bits64(diff, 0, len);

    // welem = ZeroExtend(Ones(S+1), esize); telem = ZeroExtend(Ones(d+1), esize).
    let welem = ones(s as u32 + 1);
    let telem = ones(d as u32 + 1);

    // wmask = Replicate(ROR(welem, R)); tmask = Replicate(telem).
    // ROR is performed within `esize` bits; reduce R modulo esize first.
    let welem_ror = ror(welem, r as u32 % esize, esize);
    let wmask = replicate(welem_ror, esize, m);
    let tmask = replicate(telem, esize, m);

    Some(BitMasks { wmask, tmask })
}

/// `Ones(n)` — the ARM ARM bit-vector with the low `n` bits set (`n <= 64`).
#[inline]
const fn ones(n: u32) -> u64 {
    if n == 0 {
        0
    } else if n >= 64 {
        u64::MAX
    } else {
        (1u64 << n) - 1
    }
}

/// `ROR(x, shift)` — rotate the low `width` bits of `x` right by `shift`.
///
/// ARM ARM shared pseudocode (`ROR`/`ROR_C`); `width` is the element size and
/// `shift` is taken modulo `width` by the caller. Bits above `width` in `x` are
/// ignored, and the result is confined to `width` bits.
#[inline]
const fn ror(x: u64, shift: u32, width: u32) -> u64 {
    let x = bits64(x, 0, width);
    if shift == 0 {
        return x;
    }
    let lo = x >> shift;
    let hi = x << (width - shift);
    bits64(lo | hi, 0, width)
}

/// `MoveWidePreferred` — `true` when a `MOV (bitmask immediate)` alias is the
/// preferred disassembly over `MOVZ`/`MOVN`/`ORR` for `(sf, immN, imms, immr)`.
///
/// Hand-written from the ARM ARM alias-condition pseudocode.
#[inline]
pub fn move_wide_preferred(sf: u32, imm_n: u32, imms: u32, immr: u32) -> bool {
    let sf = sf & 1;
    let imm_n = imm_n & 1;
    let imms = imms & 0x3f;
    let immr = immr & 0x3f;

    let s = imms as i32; // S = UInt(imms)
    let r = immr as i32; // R = UInt(immr)
    let width: i32 = if sf == 1 { 64 } else { 32 };

    // ARM ARM:
    //   if sf == '1' && immN != '1' then return FALSE;
    //   if sf == '0' && !(immN == '0' && imms<5> == '0') then return FALSE;
    if sf == 1 && imm_n != 1 {
        return false;
    }
    if sf == 0 && !(imm_n == 0 && bit(imms, 5) == 0) {
        return false;
    }

    // For MOVZ-style: must not contain more than 16 ones.
    if s < 16 {
        // -R MOD 16 <= 15 - S  (rem_euclid gives the non-negative modulus)
        return (-r).rem_euclid(16) <= (15 - s);
    }
    // For MOVN-style: must not contain more than 16 ones.
    if s >= width - 15 {
        return r.rem_euclid(16) <= (s - (width - 15));
    }

    false
}

/// `AdvSIMDExpandImm` — expand a NEON modified immediate `(op, cmode, imm8)`
/// into its 64-bit element value.
///
/// Hand-written from the ARM ARM shared pseudocode.
#[inline]
pub fn adv_simd_expand_imm(op: u32, cmode: u32, imm8: u32) -> u64 {
    let op = op & 1;
    let cmode = cmode & 0xf;
    let imm8 = (imm8 & 0xff) as u64;

    let cmode_hi = (cmode >> 1) & 0x7; // cmode<3:1>
    let cmode0 = cmode & 1; // cmode<0>

    match cmode_hi {
        // '000': 32-bit, imm8 in byte 0.
        0b000 => replicate(imm8, 32, 64),
        // '001': 32-bit, imm8 << 8.
        0b001 => replicate(imm8 << 8, 32, 64),
        // '010': 32-bit, imm8 << 16.
        0b010 => replicate(imm8 << 16, 32, 64),
        // '011': 32-bit, imm8 << 24.
        0b011 => replicate(imm8 << 24, 32, 64),
        // '100': 16-bit, imm8 in low byte.
        0b100 => replicate(imm8, 16, 64),
        // '101': 16-bit, imm8 << 8.
        0b101 => replicate(imm8 << 8, 16, 64),
        // '110': 32-bit with trailing ones (one-shot / two-shot ORR-style).
        0b110 => {
            // cmode<0>==0: Zeros(16):imm8:Ones(8)  -> (imm8<<8) | 0xff
            // cmode<0>==1: Zeros(8):imm8:Ones(16)  -> (imm8<<16) | 0xffff
            let elem = if cmode0 == 0 {
                (imm8 << 8) | 0xff
            } else {
                (imm8 << 16) | 0xffff
            };
            replicate(elem, 32, 64)
        }
        // '111'.
        _ => {
            if cmode0 == 0 {
                if op == 0 {
                    // cmode=1110, op=0: MOVI byte — replicate imm8 to all 8 bytes.
                    replicate(imm8, 8, 64)
                } else {
                    // cmode=1110, op=1: each bit of imm8 expands to a full byte.
                    let mut imm64 = 0u64;
                    let mut i = 0;
                    while i < 8 {
                        if (imm8 >> i) & 1 != 0 {
                            imm64 |= 0xffu64 << (i * 8);
                        }
                        i += 1;
                    }
                    imm64
                }
            } else if op == 0 {
                // cmode=1111, op=0: FMOV (vector) single-precision.
                // imm32 = imm8<7> : NOT(imm8<6>) : Replicate(imm8<6>,5) : imm8<5:0> : Zeros(19)
                let imm32 = fp_imm32(imm8);
                replicate(imm32, 32, 64)
            } else {
                // cmode=1111, op=1: FMOV (vector) double-precision.
                // imm64 = imm8<7> : NOT(imm8<6>) : Replicate(imm8<6>,8) : imm8<5:0> : Zeros(48)
                fp_imm64(imm8)
            }
        }
    }
}

/// Build the single-precision (f32) bit pattern from an 8-bit FP immediate.
///
/// `imm8<7>` is the sign, `imm8<6>` selects the exponent bias direction
/// (`NOT(imm8<6>)` then 5 copies of `imm8<6>`), and `imm8<5:0>` is the high
/// fraction; the remaining 19 fraction bits are zero. Shared by
/// `AdvSIMDExpandImm` (cmode=1111,op=0) and `VFPExpandImm(.., 32)`.
#[inline]
const fn fp_imm32(imm8: u64) -> u64 {
    let sign = (imm8 >> 7) & 1;
    let b6 = (imm8 >> 6) & 1;
    let not_b6 = b6 ^ 1;
    // exp high bits = NOT(b6):Replicate(b6,5); imm8<5:4> (in frac below) supplies
    // the low two exp bits, so frac=imm8<5:0> placed at <<19 lands them correctly.
    let exp = (not_b6 << 5) | (replicate(b6, 1, 5) & 0x1f);
    let frac = imm8 & 0x3f; // imm8<5:0>
    (sign << 31) | (exp << 25) | (frac << 19)
}

/// Build the double-precision (f64) bit pattern from an 8-bit FP immediate.
///
/// Analogue of [`fp_imm32`] with an 11-bit exponent (`NOT(imm8<6>)` then 8
/// copies of `imm8<6>`) and 52-bit fraction (the low 48 bits zero). Shared by
/// `AdvSIMDExpandImm` (cmode=1111,op=1) and `VFPExpandImm(.., 64)`.
#[inline]
const fn fp_imm64(imm8: u64) -> u64 {
    let sign = (imm8 >> 7) & 1;
    let b6 = (imm8 >> 6) & 1;
    let not_b6 = b6 ^ 1;
    // exp high bits = NOT(b6):Replicate(b6,8); imm8<5:4> (in frac) supplies the
    // low two exp bits via frac=imm8<5:0> placed at <<48.
    let exp = (not_b6 << 8) | (replicate(b6, 1, 8) & 0xff);
    let frac = imm8 & 0x3f; // imm8<5:0>
    (sign << 63) | (exp << 54) | (frac << 48)
}

/// Build the half-precision (f16) bit pattern from an 8-bit FP immediate.
///
/// Analogue of [`fp_imm32`] with a 5-bit exponent (`NOT(imm8<6>)` then 2 copies
/// of `imm8<6>`) and 10-bit fraction (the low 6 bits zero). Used by
/// `VFPExpandImm(.., 16)`.
#[inline]
const fn fp_imm16(imm8: u64) -> u64 {
    let sign = (imm8 >> 7) & 1;
    let b6 = (imm8 >> 6) & 1;
    let not_b6 = b6 ^ 1;
    let exp = (not_b6 << 2) | (replicate(b6, 1, 2) & 0x3); // 3-bit exponent (NOT(b6):b6:b6)
    let frac = imm8 & 0x3f; // imm8<5:0>
    // sign@15, exp(3 bits)@[14:12], imm8<5:0>@[11:6], Zeros(6)@[5:0].
    (sign << 15) | (exp << 12) | (frac << 6)
}

/// `VFPExpandImm` — expand an 8-bit scalar FP immediate into the raw bit pattern
/// for element width `n` (16 / 32 / 64).
///
/// Hand-written from the ARM ARM. Returns the bits only (no FP arithmetic); the
/// formatter bit-casts when rendering, keeping the decode path FP-free.
#[inline]
pub fn vfp_expand_imm(imm8: u32, n: u32) -> u64 {
    let imm8 = (imm8 & 0xff) as u64;
    // ARM ARM: result = sign : NOT(imm8<6>) : Replicate(imm8<6>, E-3) : imm8<5:0>
    //          : Zeros(F-4), where (E,F) depend on N. The AdvSIMD f32/f64 forms
    //          and these VFP forms produce identical bit patterns, so the shared
    //          `fp_imm{16,32,64}` builders are reused here.
    match n {
        16 => fp_imm16(imm8),
        32 => fp_imm32(imm8),
        64 => fp_imm64(imm8),
        // N is always 16/32/64 per the spec; be total and return 0 otherwise.
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- primitives -----------------------------------------------------

    #[test]
    fn ones_edges() {
        assert_eq!(ones(0), 0);
        assert_eq!(ones(1), 1);
        assert_eq!(ones(8), 0xff);
        assert_eq!(ones(63), 0x7fff_ffff_ffff_ffff);
        assert_eq!(ones(64), u64::MAX);
    }

    #[test]
    fn ror_basic() {
        // Rotate 0b0000_1111 right by 4 within an 8-bit element -> 0b1111_0000.
        assert_eq!(ror(0x0f, 4, 8), 0xf0);
        // Rotate by 0 is identity (confined to width).
        assert_eq!(ror(0xab, 0, 8), 0xab);
        // 64-bit rotate of 0xf right by 4 -> top nibble.
        assert_eq!(ror(0xf, 4, 64), 0xf000_0000_0000_0000);
    }

    #[test]
    fn highest_set_bit_and_replicate_present() {
        assert_eq!(highest_set_bit(0), -1);
        assert_eq!(highest_set_bit(0x78), 6);
        assert_eq!(replicate(0x01, 2, 32), 0x5555_5555);
    }

    // ---- DecodeBitMasks -------------------------------------------------

    #[test]
    fn dbm_and_x0_0xff() {
        // AND x0, x0, #0xff : N=1, immr=0, imms=0b000111 (S=7), 64-bit.
        let m = decode_bit_masks(1, 0b000111, 0, true, 64).expect("valid");
        assert_eq!(m.wmask, 0xff);
        assert_eq!(m.tmask, 0xff);
    }

    #[test]
    fn dbm_period2_w_reg() {
        // 32-bit pattern 0x55555555: N=0, imms=0b111100, immr=0 (esize=2, S=0).
        let m = decode_bit_masks(0, 0b111100, 0, true, 32).expect("valid");
        assert_eq!(m.wmask, 0x5555_5555);
    }

    #[test]
    fn dbm_period16_single_bit() {
        // N=0, imms=0x20 (0b100000): NOT(imms)<5:0>=0b011111, concat=0x1f,
        // len=4 -> esize=16. levels=0xf, S=imms&0xf=0, single bit per 16.
        let m = decode_bit_masks(0, 0x20, 0, true, 64).expect("valid");
        assert_eq!(m.wmask, 0x0001_0001_0001_0001);
    }

    #[test]
    fn dbm_esize32_two_bits() {
        // N=0, imms=0x01 (S=1 within esize=32): NOT(imms)=0b111110, concat=0x3e,
        // len=5 -> esize=32, levels=0x1f, S=1 -> Ones(2)=0b11 per 32-bit lane.
        let m = decode_bit_masks(0, 0x01, 0, true, 64).expect("valid");
        assert_eq!(m.wmask, 0x0000_0003_0000_0003);
    }

    #[test]
    fn dbm_tmask_independent_of_wmask() {
        // N=1, imms=0b000011 (S=3), immr=4 (R=4), 64-bit.
        // welem = Ones(4)=0xf, ROR by 4 -> 0xf<<60.
        // diff = 3-4 = -1 -> d=63 -> telem = Ones(64) -> tmask = all ones.
        let m = decode_bit_masks(1, 0b000011, 4, true, 64).expect("valid");
        assert_eq!(m.wmask, 0xf000_0000_0000_0000);
        assert_eq!(m.tmask, u64::MAX);
        assert_ne!(m.wmask, m.tmask, "tmask must be computed independently");
    }

    #[test]
    fn dbm_reserved_when_len_zero() {
        // N=0, imms=0x3f -> NOT=0, concat=0, len=-1 -> reserved.
        assert!(decode_bit_masks(0, 0x3f, 0, true, 64).is_none());
    }

    #[test]
    fn dbm_reserved_immediate_all_ones() {
        // N=1, imms=0x3f, immediate=true: (imms & levels)==levels -> reserved.
        assert!(decode_bit_masks(1, 0x3f, 0, true, 64).is_none());
        // The same encoding is *not* rejected by the len check alone when
        // immediate=false (used by some SVE paths): it should decode.
        assert!(decode_bit_masks(1, 0x3f, 0, false, 64).is_some());
    }

    #[test]
    fn dbm_macro_reserved_n0_cases() {
        // The ARM ARM DecodeBitMasksCheckUndefined N==0 reserved imms values.
        for &imms in &[0x3Du32, 0x3B, 0x37, 0x2F, 0x1F] {
            assert!(
                decode_bit_masks(0, imms, 0, true, 64).is_none(),
                "imms={imms:#x} must be reserved"
            );
        }
    }

    // ---- MoveWidePreferred ---------------------------------------------

    #[test]
    fn mwp_movz_encodable_true() {
        // 0xffff (low 16 bits): N=1, imms=0x0f (S=15), immr=0, 64-bit.
        assert!(move_wide_preferred(1, 1, 0x0f, 0));
        // Same field shifted to the top via immr=48: still a single MOVZ field.
        assert!(move_wide_preferred(1, 1, 0x0f, 0x30));
    }

    #[test]
    fn mwp_crosses_16bit_boundary_false() {
        // 0xffff rotated by 8 spans two 16-bit lanes -> not MOVZ/MOVN-able.
        assert!(!move_wide_preferred(1, 1, 0x0f, 0x08));
    }

    #[test]
    fn mwp_64bit_requires_immn() {
        // sf=1 with immN=0 is never move-wide preferred.
        assert!(!move_wide_preferred(1, 0, 0x07, 0));
    }

    #[test]
    fn mwp_32bit_requires_imms_bit5_clear() {
        // sf=0 needs immN==0 && imms<5>==0.
        assert!(!move_wide_preferred(0, 1, 0x00, 0)); // immN=1 -> false
        assert!(!move_wide_preferred(0, 0, 0x20, 0)); // imms<5>=1 -> false
    }

    #[test]
    fn mwp_movn_side() {
        // 32-bit, S large (>= width-15): exercises the MOVN branch.
        // width=32, width-15=17. imms=0x1f (S=31) >= 17, immr=0 -> R%16=0 <= 31-17.
        assert!(move_wide_preferred(0, 0, 0x1f, 0));
    }

    // ---- AdvSIMDExpandImm ----------------------------------------------

    #[test]
    fn asei_movi_32() {
        // MOVI Vd.4S, #0xAB, cmode=0000, op=0.
        assert_eq!(adv_simd_expand_imm(0, 0b0000, 0xAB), 0x0000_00AB_0000_00AB);
        // cmode=0010 -> imm8 << 8 per 32-bit lane (cmode<3:1>=001).
        assert_eq!(adv_simd_expand_imm(0, 0b0010, 0xAB), 0x0000_AB00_0000_AB00);
        // cmode=0100 -> imm8 << 16 per 32-bit lane (cmode<3:1>=010).
        assert_eq!(adv_simd_expand_imm(0, 0b0100, 0xAB), 0x00AB_0000_00AB_0000);
        // cmode=0110 -> imm8 << 24 per 32-bit lane (cmode<3:1>=011).
        assert_eq!(adv_simd_expand_imm(0, 0b0110, 0xAB), 0xAB00_0000_AB00_0000);
    }

    #[test]
    fn asei_movi_16() {
        // MOVI Vd.8H, #0xAB, cmode=1000, op=0.
        assert_eq!(adv_simd_expand_imm(0, 0b1000, 0xAB), 0x00AB_00AB_00AB_00AB);
        // cmode=1010 -> imm8 << 8 per 16-bit lane.
        assert_eq!(adv_simd_expand_imm(0, 0b1010, 0xAB), 0xAB00_AB00_AB00_AB00);
    }

    #[test]
    fn asei_orr_style_trailing_ones() {
        // cmode=1100: Zeros(16):imm8:Ones(8) per 32-bit lane.
        assert_eq!(adv_simd_expand_imm(0, 0b1100, 0xAB), 0x0000_ABFF_0000_ABFF);
        // cmode=1101: Zeros(8):imm8:Ones(16) per 32-bit lane.
        assert_eq!(adv_simd_expand_imm(0, 0b1101, 0xAB), 0x00AB_FFFF_00AB_FFFF);
    }

    #[test]
    fn asei_movi_byte_and_64() {
        // cmode=1110, op=0: replicate imm8 to all bytes.
        assert_eq!(adv_simd_expand_imm(0, 0b1110, 0xAB), 0xABAB_ABAB_ABAB_ABAB);
        // cmode=1110, op=1: per-bit byte expansion. imm8=0x81 -> bytes 0 and 7.
        assert_eq!(adv_simd_expand_imm(1, 0b1110, 0x81), 0xFF00_0000_0000_00FF);
        // All ones imm8 -> all ones.
        assert_eq!(adv_simd_expand_imm(1, 0b1110, 0xFF), u64::MAX);
    }

    #[test]
    fn asei_fmov_vector() {
        // FMOV Vd.2S, #1.0 (cmode=1111, op=0). imm8 for 1.0f is 0x70.
        assert_eq!(adv_simd_expand_imm(0, 0b1111, 0x70), 0x3F80_0000_3F80_0000);
        // FMOV Vd.2D, #1.0 (cmode=1111, op=1).
        assert_eq!(adv_simd_expand_imm(1, 0b1111, 0x70), 0x3FF0_0000_0000_0000);
    }

    // ---- VFPExpandImm ---------------------------------------------------

    #[test]
    fn vfp_f32_known() {
        // 1.0f -> 0x3f800000.
        assert_eq!(vfp_expand_imm(0x70, 32), 0x3F80_0000);
        // 2.0f -> 0x40000000 (imm8 = 0x00).
        assert_eq!(vfp_expand_imm(0x00, 32), 0x4000_0000);
        // -2.0f -> 0xC0000000 (imm8 = 0x80).
        assert_eq!(vfp_expand_imm(0x80, 32), 0xC000_0000);
    }

    #[test]
    fn vfp_f64_known() {
        // 1.0 -> 0x3FF0000000000000.
        assert_eq!(vfp_expand_imm(0x70, 64), 0x3FF0_0000_0000_0000);
        // 2.0 -> 0x4000000000000000.
        assert_eq!(vfp_expand_imm(0x00, 64), 0x4000_0000_0000_0000);
    }

    #[test]
    fn vfp_f16_known() {
        // 1.0h -> 0x3C00.
        assert_eq!(vfp_expand_imm(0x70, 16), 0x3C00);
        // 2.0h -> 0x4000.
        assert_eq!(vfp_expand_imm(0x00, 16), 0x4000);
        // -2.0h -> 0xC000.
        assert_eq!(vfp_expand_imm(0x80, 16), 0xC000);
    }

    #[test]
    fn vfp_matches_advsimd_fp_forms() {
        // The scalar VFP form must agree with the vector AdvSIMD FP element.
        for imm8 in 0u32..=0xff {
            let v32 = vfp_expand_imm(imm8, 32);
            let a32 = adv_simd_expand_imm(0, 0b1111, imm8) & 0xffff_ffff;
            assert_eq!(v32, a32, "f32 mismatch imm8={imm8:#x}");
            let v64 = vfp_expand_imm(imm8, 64);
            let a64 = adv_simd_expand_imm(1, 0b1111, imm8);
            assert_eq!(v64, a64, "f64 mismatch imm8={imm8:#x}");
        }
    }
}
