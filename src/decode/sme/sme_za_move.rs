//! SME2 move multi-vectors to/from a ZA *tile* slice group — `MOV` / `MOVAZ`
//! (FEAT_SME2).
//!
//! These move a 2- or 4-register `Z` group to or from a horizontal/vertical slice
//! group of a ZA tile, e.g. `mov za0h.b[w12, 6:7], {z12.b, z13.b}` (vectors → ZA)
//! and `mov {z0.b, z1.b}, za0h.b[w12, 0:1]` (ZA → vectors); `movaz` is the ZA →
//! vectors readout that also zeroes the source slices. They live in the SME `110`
//! quadrant (top byte `0xC0`, `word<24> == 0`) and are dispatched from
//! [`super::decode_mova_add`].
//!
//! Distinct from the SME2 multi-vector *array*-vector `MOV` (`za.<T>[w8, k, vgxN]`,
//! `word<11> == 1`) and from `MOVA` (single-vector tile slice, `word<18> == 0`)
//! and `ZERO` (`word<19> == 1`).
//!
//! ## Field layout (derived from the LLVM 21 oracle)
//!
//! ```text
//!  31      24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8       0
//! [ 1100_0000 | sz | 0  0 | 0 | 1 | D | 0 | V | <Ws>| 0  0 | Q | ... ]
//! ```
//!
//! * `word<23:22>` (`sz`) is the element size (`.B`=00 … `.D`=11).
//! * `word<21:19> == 000`, `word<18> == 1`, `word<16> == 0`, `word<12:11> == 00`
//!   are the fixed shell (separating this from `MOVA`/`ZERO`/array-vector `MOV`).
//! * `word<17>` (`D`): `0` → vectors → ZA, `1` → ZA → vectors.
//! * `word<15>` (`V`): slice direction (`0` → horizontal, `1` → vertical).
//! * `word<14:13>` (`Ws`): `w12 + Ws` is the slice-select register.
//! * `word<10>` (`Q`): `0` → `vgx2` (2-register), `1` → `vgx4` (4-register). The
//!   displayed slice range is `off:off+span-1` with `span` = 2/4.
//! * Direction `0` (vectors → ZA): the `Z` group base is `word<9:5>`, and the
//!   ZA tile+offset field is `word<2:0>`.
//! * Direction `1` (ZA → vectors): `word<9>` selects `MOV`(0)/`MOVAZ`(1); the `Z`
//!   group base is `word<4:0>`, `word<8>` is reserved zero, and the ZA tile+offset
//!   field is `word<7:5>`.
//! * The 3-bit tile+offset field packs `tile` in the high bits and the slice
//!   offset in the low bits, split by element size: `.B` 0 tile / 3 offset bits,
//!   `.H` 1/2, `.S` 2/1, `.D` 3/0. The displayed offset is `field_offset * span`.
//!
//! Verified bidirectionally against `llvm-mc --mattr=+all`. Total and panic-free.

use crate::decode::bits::{bit, bits};
use crate::enums::VectorArrangement as VA;
use crate::instruction::Instruction;
use crate::mnemonic::Code;
use crate::operand::{Operand, SliceIndicator};
use crate::register::{gp_register, sve_register, RegWidth};

/// A 2/4-register consecutive `Z` group `{Z<n>.<T>{, ...}}`.
#[inline]
fn zgroup(first: u32, count: u8, arr: VA) -> Operand {
    Operand::SveVecGroup {
        first: sve_register(first as u8),
        count,
        arr: Some(arr),
        range: count == 4 && (first + 3) < 32,
        stride: 1,
        lane: None,
    }
}

/// The ZA tile-slice group operand `za<tile><h|v>.<T>[<Ws>, <off>:<off+span-1>]`.
#[inline]
fn za_tile_slice(arr: VA, sel: u32, tile: u32, off: u32, span: u8, vertical: bool) -> Operand {
    Operand::SmeZaSlice {
        arr: Some(arr),
        sel: gp_register(false, RegWidth::W32, (12 + sel) as u8),
        off: off as u8,
        span,
        vg: 0,
        tile: tile as u8,
        slice: if vertical {
            SliceIndicator::Vertical
        } else {
            SliceIndicator::Horizontal
        },
    }
}

/// Decode an SME2 ZA tile-slice `MOV`/`MOVAZ` (move multi-vectors) into `out`.
///
/// Called from [`super::decode_mova_add`] for the `110` quadrant once the shell
/// `word<24> == 0`, `word<19> == 0`, `word<18> == 1`, `word<16> == 0`,
/// `word<12:11> == 0` is selected (FEAT_SME2).
#[inline]
pub fn decode(word: u32, out: &mut Instruction) {
    // Shell: word<23:22> = sz, word<21:19> == 000, word<18> == 1, word<16> == 0,
    // word<12:11> == 00. (word<31:24> == 0xC0 guaranteed by the caller.)
    if bits(word, 19, 3) != 0b000 || bit(word, 18) != 1 || bit(word, 16) != 0 || bits(word, 11, 2) != 0
    {
        return;
    }

    // (arr, size-code 0..3). The tile count is `1 << size` (`.B`→1 … `.D`→8).
    let (arr, size) = match bits(word, 22, 2) {
        0b00 => (VA::Sb, 0u32),
        0b01 => (VA::Sh, 1),
        0b10 => (VA::Ss, 2),
        _ /* 0b11 */ => (VA::Sd, 3),
    };

    let to_vector = bit(word, 17) == 1;
    let vertical = bit(word, 15) == 1;
    let ws = bits(word, 13, 2);
    let span: u8 = if bit(word, 10) == 1 { 4 } else { 2 };

    // The `Z` group base must be span-aligned (even for vgx2, multiple-of-4 for
    // vgx4) — LLVM leaves the misaligned bases UNDEFINED.
    let base_mask = (span as u32) - 1;

    // Split the 3-bit `tile:offset` field into `(tile, slice-offset)`. The offset
    // sub-field width is `3 - size` for `vgx2`, and one bit narrower for `vgx4`
    // (a `vgx4` group covers twice as many slices, so the displayed range
    // `off*span` needs one fewer index bit). The tile takes the remaining high
    // bits; a tile number `>= 1 << size` (the architectural tile count for this
    // element size) is UNDEFINED. Returns `None` when the tile is out of range.
    let off_bits = (3 - size).saturating_sub(if span == 4 { 1 } else { 0 });
    let num_tiles = 1u32 << size;
    let split = |field: u32| -> Option<(u32, u32)> {
        let off_mask = if off_bits == 0 { 0 } else { (1u32 << off_bits) - 1 };
        let off = field & off_mask;
        let tile = field >> off_bits;
        if tile >= num_tiles {
            return None;
        }
        Some((tile, off * span as u32))
    };

    if to_vector {
        // ZA → vectors: word<9> = MOV(0)/MOVAZ(1); Zd base = word<4:0>; tile:off at
        // word<7:5>; word<8> reserved zero.
        if bit(word, 8) != 0 {
            return;
        }
        let zd = bits(word, 0, 5);
        if zd & base_mask != 0 {
            return;
        }
        let movaz = bit(word, 9) == 1;
        let (tile, off) = match split(bits(word, 5, 3)) {
            Some(v) => v,
            None => return,
        };
        out.set(if movaz {
            Code::SmeMovazMultiTileToZ
        } else {
            Code::SmeMovaMultiTileToZ
        });
        out.push_operand(zgroup(zd, span, arr));
        out.push_operand(za_tile_slice(arr, ws, tile, off, span, vertical));
    } else {
        // vectors → ZA: Zn base = word<9:5>; tile:off at word<2:0>; word<4:3>
        // reserved zero.
        if bits(word, 3, 2) != 0 {
            return;
        }
        let zn = bits(word, 5, 5);
        if zn & base_mask != 0 {
            return;
        }
        let (tile, off) = match split(bits(word, 0, 3)) {
            Some(v) => v,
            None => return,
        };
        out.set(Code::SmeMovaMultiZToTile);
        out.push_operand(za_tile_slice(arr, ws, tile, off, span, vertical));
        out.push_operand(zgroup(zn, span, arr));
    }
}

/// The ZA-*array*-vector slice operand `za.<T>[<Ws>, <off>, vgxN]`.
///
/// The slice offset is a single index (`span == 1`, no `off:off+n` range), and the
/// multi-vector count is carried by `vg` (`2`/`4`) so the formatter appends the
/// `, vgx2`/`, vgx4` qualifier.
#[inline]
fn za_array_slice(arr: VA, sel: u32, off: u32, vg: u8) -> Operand {
    Operand::SmeZaSlice {
        arr: Some(arr),
        // The array-vector forms select with `w8..w11` (the 2-bit `Ws` field added
        // to `w8`), unlike the tile-slice forms which use `w12..w15`.
        sel: gp_register(false, RegWidth::W32, (8 + sel) as u8),
        off: off as u8,
        span: 1,
        vg,
        tile: 0,
        slice: SliceIndicator::None,
    }
}

/// Decode an SME2 multi-vector **ZA-array-vector** `MOV`/`MOVAZ` into `out`.
///
/// These move a 2-/4-register `Z` group to or from an *array* slice
/// `za.d[<Ws>, <off>, vgxN]` (the `.d`/64-bit element is the only allocated size),
/// e.g. `mov { z0.d, z1.d }, za.d[w8, 0, vgx2]` and the reverse
/// `mov za.d[w8, 0, vgx2], { z0.d, z1.d }`; `movaz` is the zeroing readout (ZA →
/// vectors only). They live in the SME `110` quadrant (top byte `0xC0`,
/// `word<24> == 0`) and are dispatched from [`super::decode_mova_add`] once the
/// shell `word<21:19> == 000`, `word<18> == 1`, `word<16> == 0`,
/// `word<12:11> == 10` is selected (FEAT_SME2).
///
/// Field layout (derived from the LLVM 21 oracle):
/// ```text
///  31      24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7   5 4   0
/// [ 1100_0000 | 0 0 | 0  0 | 0 | 1 | D | 0 | V | <Ws>| 1 | Q | ... ]
/// ```
/// * `word<23:22> == 00` (always `.d`); `word<17>` (`D`): `0` → vectors → ZA,
///   `1` → ZA → vectors. `word<15>` (`V`) is RES0 (the array form has no slice
///   direction). `word<14:13>` (`Ws`): `w8 + Ws`. `word<10>` (`Q`): vgx2(0)/vgx4(1).
/// * ZA → vectors (`D == 1`): `word<9>` selects `MOV`(0)/`MOVAZ`(1); the `Z` group
///   base is `word<4:1>` (vgx2, stride 2) / `word<4:2>` (vgx4, stride 4); the slice
///   offset is `word<7:5>`; `word<8>` and `word<0>` are RES0.
/// * vectors → ZA (`D == 0`): the `Z` group base is `word<9:6>` (vgx2, stride 2) /
///   `word<9:7>` (vgx4, stride 4); the slice offset is `word<2:0>`; `word<5:3>`
///   (vgx2) / `word<6:3>` (vgx4) are RES0.
#[inline]
pub fn decode_array(word: u32, out: &mut Instruction) {
    // Shell re-check (caller has already gated `word<21:19>/18/16/12:11`).
    // `word<23:22>` is RES0 (the array form is `.d` only); `word<15>` (V) is RES0.
    if bits(word, 22, 2) != 0 || bit(word, 15) != 0 {
        return;
    }
    let arr = VA::Sd;
    let to_vector = bit(word, 17) == 1;
    let ws = bits(word, 13, 2);
    let span: u8 = if bit(word, 10) == 1 { 4 } else { 2 };
    let base_mask = (span as u32) - 1;

    if to_vector {
        // ZA → vectors: Zd group base = word<4:1>/<4:2>, slice offset = word<7:5>,
        // word<9> = MOV(0)/MOVAZ(1); word<8> and word<0> RES0.
        if bit(word, 8) != 0 || bit(word, 0) != 0 {
            return;
        }
        // vgx2 base in word<4:1> (stride 2), vgx4 in word<4:2> (stride 4; word<1>
        // is then RES0).
        let zd = if span == 4 {
            if bit(word, 1) != 0 {
                return;
            }
            bits(word, 2, 3) << 2
        } else {
            bits(word, 1, 4) << 1
        };
        if zd & base_mask != 0 {
            return;
        }
        let off = bits(word, 5, 3);
        let movaz = bit(word, 9) == 1;
        out.set(if movaz {
            Code::SmeMovazArrayToVec
        } else {
            Code::SmeMovaArrayToVec
        });
        out.push_operand(zgroup(zd, span, arr));
        out.push_operand(za_array_slice(arr, ws, off, span));
    } else {
        // vectors → ZA: Zn group base = word<9:6>/<9:7>, slice offset = word<2:0>;
        // word<5:3> (and word<6> for vgx4) RES0; word<9> RES0 only for vgx4's
        // 3-bit field.
        if bits(word, 3, 3) != 0 {
            return;
        }
        let zn = if span == 4 {
            // vgx4: base in word<9:7> (stride 4), word<6> RES0.
            if bit(word, 6) != 0 {
                return;
            }
            bits(word, 7, 3) << 2
        } else {
            // vgx2: base in word<9:6> (stride 2).
            bits(word, 6, 4) << 1
        };
        if zn & base_mask != 0 {
            return;
        }
        let off = bits(word, 0, 3);
        out.set(Code::SmeMovaVecToArray);
        out.push_operand(za_array_slice(arr, ws, off, span));
        out.push_operand(zgroup(zn, span, arr));
    }
}
