//! SME `ZERO` (ZA tile mask / ZT0 / ZA array) and SME2 `MOVT` (move to/from the
//! ZT0 lookup table).
//!
//! These live in the SME `110` quadrant (top byte `0xC0`, `word<24> == 0`) and
//! are dispatched from [`super::decode_mova_add`] once the shared shell
//! `word<23> == 0`, `word<19> == 1` is selected. That shell never overlaps the
//! base `MOVA`/`ADDHA`/`ADDVA` (which have `word<19> == 0` or `word<23> == 1`)
//! nor the LUTI ZT0 (`word<23> == 1`) families. Every path is total and
//! panic-free, leaving [`Code::Invalid`] for unallocated encodings.
//!
//! ## `ZERO` (FEAT_SME / FEAT_SME2) — three shapes
//!
//! * **ZA tile mask** (`word<23:16> == 0x08`, `word<15:8> == 0`): the destination
//!   is the brace list selected by the 8-bit mask `word<7:0>`. `0x00` → `{}`,
//!   `0xFF` → `{ za }`, otherwise the *largest* element width that cleanly tiles
//!   the mask names the tiles (`.h`: `za0.h == 0x55`, `za1.h == 0xAA`; `.s`:
//!   `za<i>.s == 0x11 << i`; `.d`: `za<i>.d == 1 << i`). FEAT_SME.
//! * **ZT0** (exactly `0xC0480001`): `zero { zt0 }`. FEAT_SME2.
//! * **ZA array `.d` slice group** (`word<19:18> == 11`): `zero za.d[<Ws>, ...]`,
//!   `.d`-only. `word<14:13>` = `Ws` (`w8..w11`), `word<2:0>` = the index field,
//!   and `word<17:16:15>` select the `(span, vgxN)` shape (see the table in
//!   [`decode_zero_array`]). FEAT_SME2.
//!
//! ## `MOVT` (FEAT_SME2) — three directions
//!
//! * `movt zt0[<index>, mul vl], <Zt>` (`word<23:16> == 0x4F`): `index =
//!   word<13:12>` (`0..=3`), `Zt = word<4:0>`.
//! * `movt zt0[<offset>], <Xt>` (`word<23:16> == 0x4E`): byte `offset =
//!   word<14:12> * 8`, `Xt = word<4:0>`.
//! * `movt <Xt>, zt0[<offset>]` (`word<23:16> == 0x4C`): byte `offset =
//!   word<14:12> * 8`, `Xt = word<4:0>`.
//!
//! All three fix `word<15>` (Z-form: `word<15:14>`) and `word<11:5> == 0x1F`.
//!
//! Verified bidirectionally against `llvm-mc --mattr=+all` (LLVM 21).

use crate::decode::bits::{bit, bits};
use crate::enums::VectorArrangement as VA;
use crate::features::{Feature, FeatureSet};
use crate::instruction::Instruction;
use crate::mnemonic::Code;
use crate::operand::{Operand, SliceIndicator};
use crate::register::{gp_register, sve_register, RegWidth};

/// Decode an SME `ZERO`/`MOVT` (ZA tile mask / ZT0 / ZA array / move-to-ZT0)
/// into `out`.
///
/// Called from [`super::decode_mova_add`] once the shell `word<24> == 0`,
/// `word<23> == 0`, `word<19> == 1` is selected. Recognizes only the exact
/// `ZERO`/`MOVT` skeletons; anything else is left untouched (`Code::Invalid`).
#[inline]
pub fn decode(word: u32, features: FeatureSet, out: &mut Instruction) {
    // The shared shell (re-checked locally so the function is correct in
    // isolation): top byte `0xC0`, `word<24> == 0`, `word<23> == 0`,
    // `word<19> == 1`.
    if bits(word, 24, 8) != 0xC0 || bit(word, 23) != 0 || bit(word, 19) != 1 {
        return;
    }

    // `word<21:20>` are RES0 throughout this shell.
    if bits(word, 20, 2) != 0 {
        return;
    }
    // `word<22>` splits the plain-`ZERO` (`0`) destinations from the ZT0-table
    // forms (`1`); within each, `word<18>` splits the simpler form (`0`) from the
    // array / `MOVT` form (`1`):
    //   `<22>=0,<18>=0` → `zero { za-mask }`     (FEAT_SME)
    //   `<22>=0,<18>=1` → `zero za.d[Ws, ..]`    (FEAT_SME2)
    //   `<22>=1,<18>=0` → `zero { zt0 }`         (FEAT_SME2)
    //   `<22>=1,<18>=1` → `MOVT` (ZT0 move)      (FEAT_SME2)
    match (bit(word, 22), bit(word, 18)) {
        (0, 0) => decode_zero_mask(word, out),
        (0, 1) => decode_zero_array(word, features, out),
        (1, 0) => decode_zero_zt0(word, features, out),
        (1, 1) => decode_movt(word, features, out),
        _ => {}
    }
}

/// `zero { za-mask }` — the ZA tile-mask `ZERO` (FEAT_SME). `word<23:16> == 0x08`
/// (already gated: `word<23> == 0`, `word<22> == 0`, `word<21:20> == 00`,
/// `word<19> == 1`, `word<18> == 0`); `word<15:8>` is RES0; the 8-bit mask is
/// `word<7:0>`.
#[inline]
fn decode_zero_mask(word: u32, out: &mut Instruction) {
    // `word<17:16>` and `word<15:8>` are RES0 (only `0x08`+mask is allocated).
    if bits(word, 16, 2) != 0 || bits(word, 8, 8) != 0 {
        return;
    }
    let mask = bits(word, 0, 8) as u8;
    out.set(Code::SmeZeroMask);
    out.push_operand(Operand::SmeZaMask { mask, zt0: false });
}

/// `zero { zt0 }` (FEAT_SME2). Reached with `word<23> == 0`, `word<22> == 1`,
/// `word<21:20> == 00`, `word<19> == 1`, `word<18> == 0`. The only allocated
/// encoding is exactly `0xC0480001` (`word<17:0> == 0x00001`).
#[inline]
fn decode_zero_zt0(word: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Sme2) {
        return;
    }
    if word == 0xC048_0001 {
        out.set(Code::SmeZeroZt0);
        out.push_operand(Operand::SmeZaMask { mask: 0, zt0: true });
    }
}

/// The three `MOVT` directions (FEAT_SME2). Reached with `word<23> == 0`,
/// `word<22> == 1`, `word<21:20> == 00`, `word<19> == 1`, `word<18> == 1`.
/// `word<11:5> == 0x1F` is fixed; `word<17:16>` selects the direction.
#[inline]
fn decode_movt(word: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Sme2) {
        return;
    }
    // `word<11:5> == 0x1F` is fixed for every `MOVT` skeleton.
    if bits(word, 5, 7) != 0x1F {
        return;
    }

    // The `MOVT` directions are selected by `word<17:16>`:
    //   `11` → `movt zt0[idx, mul vl], Zt`  (Z-register, `word<23:16> == 0x4F`)
    //   `10` → `movt zt0[off], Xt`          (GP store,  `word<23:16> == 0x4E`)
    //   `00` → `movt Xt, zt0[off]`          (GP load,   `word<23:16> == 0x4C`)
    match bits(word, 16, 2) {
        0b11 => {
            // Z-register form: `word<15:14>` RES0, index = `word<13:12>`,
            // Zt = `word<4:0>`.
            if bits(word, 14, 2) != 0 {
                return;
            }
            let index = bits(word, 12, 2) as u8;
            let zt = bits(word, 0, 5);
            out.set(Code::SmeMovtZt0Z);
            out.push_operand(Operand::SmeZt0Index { index, mul_vl: true });
            out.push_operand(zreg(zt));
        }
        0b10 => {
            // GP-store form: `word<15>` RES0, byte offset = `word<14:12> * 8`,
            // Xt = `word<4:0>`.
            if bit(word, 15) != 0 {
                return;
            }
            let off = (bits(word, 12, 3) * 8) as u8;
            let xt = bits(word, 0, 5);
            out.set(Code::SmeMovtZt0X);
            out.push_operand(Operand::SmeZt0Index { index: off, mul_vl: false });
            out.push_operand(xreg(xt));
        }
        0b00 => {
            // GP-load form: `word<15>` RES0, byte offset = `word<14:12> * 8`,
            // Xt = `word<4:0>`.
            if bit(word, 15) != 0 {
                return;
            }
            let off = (bits(word, 12, 3) * 8) as u8;
            let xt = bits(word, 0, 5);
            out.set(Code::SmeMovtXZt0);
            out.push_operand(xreg(xt));
            out.push_operand(Operand::SmeZt0Index { index: off, mul_vl: false });
        }
        _ => {}
    }
}

/// `zero za.d[<Ws>, ...]` — the SME2 ZA-array `.d` slice-group `ZERO`
/// (FEAT_SME2). Reached with `word<23> == 0`, `word<19:18> == 11`. `.d`-only.
///
/// Field layout (LLVM 21 oracle):
/// ```text
///  31      24 23 22 21 20 19 18 17 16 15 14 13 12       3 2  0
/// [ 1100_0000 | 0  0 | 0  0 | 1 | 1 | g | V | R | <Ws>| 0...0 | fld ]
/// ```
/// `word<14:13>` = `Ws` (`w8..w11`); `word<12:3>` are RES0; `word<2:0>` = `fld`.
/// `(word<17> = g, word<16> = V, word<15> = R)` select the `(span, vgxN)` shape:
///
/// | g | V | R | span | vgx | off    | valid `fld` |
/// |---|---|---|------|-----|--------|-------------|
/// | 0 | 0 | 0 | 1    | 2   | `fld`  | `0..=7`     |
/// | 1 | 0 | 0 | 1    | 4   | `fld`  | `0..=7`     |
/// | 0 | 0 | 1 | 2    | —   | `fld*2`| `0..=7`     |
/// | 1 | 0 | 1 | 4    | —   | `fld*4`| `0..=3`     |
/// | 0 | 1 | 0 | 2    | 2   | `fld*2`| `0..=3`     |
/// | 0 | 1 | 1 | 2    | 4   | `fld*2`| `0..=3`     |
/// | 1 | 1 | 0 | 4    | 2   | `fld*4`| `0..=1`     |
/// | 1 | 1 | 1 | 4    | 4   | `fld*4`| `0..=1`     |
///
/// `off + span` must stay within the 16 ZA-array `.d` slices, which the
/// per-shape `fld` range above enforces.
#[inline]
fn decode_zero_array(word: u32, features: FeatureSet, out: &mut Instruction) {
    if !features.has(Feature::Sme2) {
        return;
    }
    // `word<22:20>` and `word<12:3>` are RES0.
    if bits(word, 20, 3) != 0 || bits(word, 3, 10) != 0 {
        return;
    }
    let g = bit(word, 17);
    let v = bit(word, 16);
    let r = bit(word, 15);
    let ws = bits(word, 13, 2);
    let fld = bits(word, 0, 3);

    // (span, vg, max_field) per the table above.
    let (span, vg, max_field): (u8, u8, u32) = match (g, v, r) {
        (0, 0, 0) => (1, 2, 7),
        (1, 0, 0) => (1, 4, 7),
        (0, 0, 1) => (2, 0, 7),
        (1, 0, 1) => (4, 0, 3),
        (0, 1, 0) => (2, 2, 3),
        (0, 1, 1) => (2, 4, 3),
        (1, 1, 0) => (4, 2, 1),
        (1, 1, 1) => (4, 4, 1),
        _ => return,
    };
    if fld > max_field {
        return;
    }
    let off = (fld as u8) * span;

    out.set(Code::SmeZeroArray);
    out.push_operand(Operand::SmeZaSlice {
        arr: Some(VA::Sd),
        sel: gp_register(false, RegWidth::W32, (8 + ws) as u8),
        off,
        span,
        vg,
        tile: 0,
        slice: SliceIndicator::None,
    });
}

/// `Z<n>` (no arrangement) — the `MOVT` Z-register source.
#[inline]
fn zreg(n: u32) -> Operand {
    Operand::Reg {
        reg: sve_register((n & 0x1f) as u8),
        arr: None,
        lane: None,
        shift: None,
        extend: None,
        pred: None,
    }
}

/// `X<n>` (64-bit GP) — the `MOVT` general-purpose source/destination.
#[inline]
fn xreg(n: u32) -> Operand {
    Operand::Reg {
        reg: gp_register(false, RegWidth::X64, (n & 0x1f) as u8),
        arr: None,
        lane: None,
        shift: None,
        extend: None,
        pred: None,
    }
}
