//! SME2 lookup-table reads from the ZT0 table — `LUTI2` / `LUTI4` (FEAT_LUT).
//!
//! These are the **SME2 ZT0** lookup forms, distinct from the SVE `LUTI2`/`LUTI4`
//! (which take a `Zm` table index and no ZT0; see
//! [`crate::decode::sve::sve_lut`]). They live in the SME `110` quadrant (top byte
//! `0xC0`, `word<24> == 0`) and are dispatched from
//! [`super::decode_mova_add`]. The destination is a 1-, 2- or 4-register `Z`
//! group with element size `.B`/`.H`/`.S`; the table is the fixed `ZT0`; the third
//! operand is the index source `Z<n>[<index>]` (a `Z` register with a bracketed
//! lane index and no arrangement suffix).
//!
//! ## Field layout (derived from the LLVM 21 oracle)
//!
//! ```text
//!  31      24 23 22 21 20 19 18 17        13 12 11 10 9      5 4      0
//! [ 1100_0000 | 1 | R | 0 St | 1 | L | <cnt:idx> | sz| 0  0 | <Zn>   | <Zd> ]
//! ```
//!
//! * `word<23>` is fixed `1`; `word<21> == 0`; `word<19> == 1` (selects the LUTI
//!   family — `word<19> == 0` is the ZA-tile move / `ZERO` neighbour).
//! * `word<22>` (`R`): `1` → single-register destination, `0` → multi (2/4-reg).
//! * `word<20>` (`St`): the destination-group stride — `0` → consecutive
//!   (`{Zd, Zd+1, ..}`), `1` → strided (`{Zd, Zd+8}` for 2-reg / `{Zd, Zd+4, ..}`
//!   for 4-reg). Strided is multi-register only.
//! * `word<18>` (`L`): `1` → `LUTI2` (2-bit indices), `0` → `LUTI4` (4-bit).
//!   `LUTI4` also fixes `word<17> == 1` (`word<17> == 0` is the distinct `LUTI6`).
//! * The element size occupies `word<13:12>` (`.B`=00, `.H`=01, `.S`=10; `11`
//!   unallocated). For multi-register forms a single "count marker" bit sits just
//!   above the size: `word<14> == 1` → 2-register (index from `word<15>`),
//!   `word<14> == 0, word<15> == 1` → 4-register (index from `word<16>`). For the
//!   single-register form the index starts at `word<14>`.
//! * The index field always *ends* at `word<17>` for `LUTI2` and `word<16>` for
//!   `LUTI4`, growing downward to its per-form least-significant bit.
//! * `word<11:10>` are reserved zero; `Zn = word<9:5>`; `Zd = word<4:0>`.
//!
//! Two extra `LUTI4` forms (consecutive only, `word<20> == 0`):
//!
//! * **4-register, register-pair source** (`.B` only): `word<17> == 1`,
//!   `word<16> == 1`, `word<15:14> == 00`. The index source is a 2-register group
//!   `{Zn, Zn+1}` (`Zn = word<9:5>`, even) instead of an indexed single; the
//!   destination is the consecutive 4-group `{Zd, .., Zd+3}`.
//!
//! Strided/size restrictions (verified): strided forms never allow `.S`; `LUTI4`
//! 4-register is `.H`-only when strided and never `.B` (consecutive or strided).
//!
//! Verified bidirectionally against `llvm-mc --mattr=+all`. Total and panic-free;
//! only the exact valid encodings are recognized, everything else is left
//! [`Code::Invalid`].

use crate::decode::bits::{bit, bits};
use crate::enums::VectorArrangement as VA;
use crate::instruction::Instruction;
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::Operand;
use crate::register::{sve_register, Register};

/// A 2/4-register `Z` group destination `{Z<n>.<T>{, ...}}`. `stride` is `1`
/// (consecutive), `8` (2-register strided) or `4` (4-register strided).
#[inline]
fn zgroup(first: u32, count: u8, arr: VA, stride: u8) -> Operand {
    Operand::SveVecGroup {
        first: sve_register(first as u8),
        count,
        arr: Some(arr),
        // LLVM renders a non-wrapping consecutive 4-register group as a `z.. - z..`
        // range; a 2-register group, a wrapping 4-group, and every strided group
        // print as a comma list.
        range: stride == 1 && count == 4 && (first + 3) < 32,
        stride,
        lane: None,
    }
}

/// A 2-register `Z` group *source* `{Z<n>, Z<n+1>}` (no element suffix, no index)
/// — the `LUTI4` register-pair table source.
#[inline]
fn zpair(first: u32) -> Operand {
    Operand::SveVecGroup {
        first: sve_register(first as u8),
        count: 2,
        arr: None,
        range: false,
        stride: 1,
        lane: None,
    }
}

/// The indexed table-source `Z<m>[index]` (no arrangement; bracketed lane).
#[inline]
fn zidx(m: u32, index: u32) -> Operand {
    Operand::Reg {
        reg: sve_register(m as u8),
        arr: None,
        lane: Some(index as u8),
        shift: None,
        extend: None,
        pred: None,
    }
}

/// The fixed `ZT0` lookup-table operand (renders `zt0`).
#[inline]
fn zt0() -> Operand {
    Operand::Reg {
        reg: Register::Zt0,
        arr: None,
        lane: None,
        shift: None,
        extend: None,
        pred: None,
    }
}

/// Decode an SME2 `LUTI2`/`LUTI4` (ZT0) lookup-table read into `out`.
///
/// Called from [`super::decode_mova_add`] for the `110` quadrant once the shell
/// `word<24> == 0`, `word<19> == 1` is selected (FEAT_LUT). Recognizes only the
/// exact ZT0 LUTI skeleton; anything else is left untouched.
#[inline]
pub fn decode(word: u32, out: &mut Instruction) {
    // Shell: word<31:24> == 0xC0, word<23> == 1, word<21> == 0, word<19> == 1.
    // word<20> is the strided/consecutive selector (handled below).
    if bits(word, 24, 8) != 0xC0 || bit(word, 23) != 1 || bit(word, 21) != 0 || bit(word, 19) != 1 {
        return;
    }
    // word<11:10> reserved zero.
    if bits(word, 10, 2) != 0 {
        return;
    }

    // Single-vector `LUTI6` (ZT0): `luti6 <Zd>.b, zt0, <Zn>` — the full opcode is
    // `word<23:16> == 0xC8` (`<22> == 1` single, `<20> == 0` no strided form,
    // `<18:16> == 000` the `LUTI6` slot) and `word<15:10> == 010000`. It is `.b`-only
    // and has no element index. Fields: `Zn = word<9:5>`, `Zd = word<4:0>`.
    // FEAT_LUT.
    if bits(word, 16, 8) == 0xC8 && bits(word, 10, 6) == 0b010000 {
        out.set(Code::SmeLuti6Single);
        out.push_operand(Operand::Reg {
            reg: sve_register(bits(word, 0, 5) as u8),
            arr: Some(VA::Sb),
            lane: None,
            shift: None,
            extend: None,
            pred: None,
        });
        out.push_operand(zt0());
        out.push_operand(Operand::Reg {
            reg: sve_register(bits(word, 5, 5) as u8),
            arr: None,
            lane: None,
            shift: None,
            extend: None,
            pred: None,
        });
        return;
    }

    let single = bit(word, 22) == 1;
    let strided = bit(word, 20) == 1;
    let is_l2 = bit(word, 18) == 1;
    // `LUTI4` (word<18> == 0) fixes word<17> == 1; word<18> == 0, word<17> == 0 is
    // the distinct `LUTI6` family (out of scope) — reject it here.
    if !is_l2 && bit(word, 17) != 1 {
        return;
    }

    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);

    // `LUTI4` 4-register register-PAIR source (`.B` only): word<17> == 1 (already
    // required for L4), word<16> == 1, word<15:14> == 00, size == .B. The table
    // source is a 2-register group `{Zn, Zn+1}`. This shape has no count marker, so
    // it is matched before the indexed marker scan. The destination is the 4-group
    // `{Zd, ..}` — consecutive (`word<20> == 0`, base multiple-of-4) or strided
    // (`word<20> == 1`, step 4, base `word<3:2> == 0`).
    if !is_l2
        && !single
        && bit(word, 16) == 1
        && bits(word, 14, 2) == 0
        && bits(word, 12, 2) == 0
    {
        // Pair source base even.
        if zn & 1 != 0 {
            return;
        }
        let dst = if !strided {
            if zd & 0b11 != 0 {
                return;
            }
            zgroup(zd, 4, VA::Sb, 1)
        } else {
            if zd & 0b1100 != 0 {
                return;
            }
            zgroup(zd, 4, VA::Sb, 4)
        };
        out.set(Code::SmeLuti4Zt);
        out.set_mnemonic(Mnemonic::Luti4);
        out.push_operand(dst);
        out.push_operand(zt0());
        out.push_operand(zpair(zn));
        return;
    }

    // Element size: word<13:12> (0=.b, 1=.h, 2=.s; 3 unallocated).
    let arr = match bits(word, 12, 2) {
        0 => VA::Sb,
        1 => VA::Sh,
        2 => VA::Ss,
        _ => return,
    };

    // Determine register count and the index field's least-significant bit.
    // The index field always ends at word<17> (LUTI2) / word<16> (LUTI4).
    let top = if is_l2 { 17 } else { 16 };
    let (count, idx_lsb): (u8, u32) = if single {
        // The single-register form has no strided variant.
        if strided {
            return;
        }
        (1, 14)
    } else if bit(word, 14) == 1 {
        // 2-register: marker at word<14>, index from word<15>.
        (2, 15)
    } else if bit(word, 15) == 1 {
        // 4-register: marker at word<15>, index from word<16>.
        (4, 16)
    } else {
        return;
    };
    if idx_lsb > top {
        return;
    }
    // The only unallocated *consecutive* (count, size) combination is `LUTI4`
    // 4-register `.b`. Strided narrows the set further: `.s` is never allowed, and
    // `LUTI4` 4-register strided is `.h`-only.
    if !is_l2 && count == 4 && matches!(arr, VA::Sb) {
        return;
    }
    if strided {
        if matches!(arr, VA::Ss) {
            return;
        }
        if !is_l2 && count == 4 && !matches!(arr, VA::Sh) {
            return;
        }
    }

    let idx_bits = top - idx_lsb + 1;
    let index = bits(word, idx_lsb, idx_bits);

    // Destination group base + stride. Consecutive groups are span-aligned (even
    // for 2-reg, multiple-of-4 for 4-reg). Strided groups use the raw `word<4:0>`
    // base within a restricted window: a 2-register strided group steps by 8 and
    // its base must have `word<3> == 0` (bases `z0..z7` / `z16..z23`); a
    // 4-register strided group steps by 4 with `word<3:2> == 0` (`z0..z3` /
    // `z16..z19`).
    let stride: u8 = if !strided {
        if (count as u32 - 1) & zd != 0 {
            return;
        }
        1
    } else if count == 2 {
        if zd & 0b1000 != 0 {
            return;
        }
        8
    } else {
        // count == 4
        if zd & 0b1100 != 0 {
            return;
        }
        4
    };

    out.set(if is_l2 { Code::SmeLuti2Zt } else { Code::SmeLuti4Zt });
    out.set_mnemonic(if is_l2 { Mnemonic::Luti2 } else { Mnemonic::Luti4 });
    if count == 1 {
        // Single destination renders as a bare `Z<d>.<T>` (no braces).
        out.push_operand(Operand::Reg {
            reg: sve_register(zd as u8),
            arr: Some(arr),
            lane: None,
            shift: None,
            extend: None,
            pred: None,
        });
    } else {
        out.push_operand(zgroup(zd, count, arr, stride));
    }
    out.push_operand(zt0());
    out.push_operand(zidx(zn, index));
}
