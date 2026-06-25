//! SME2 multi-vector ZA accumulate / multiply-into-ZA family + the `*TMOPA`
//! selective outer products — hand-written from the LLVM 21 oracle (these
//! encodings post-date the Binary Ninja reference tables).
//!
//! Gated by [`Feature::Sme2`] and compiled only under the `sme` cargo feature.
//! Two entry points, dispatched by [`super::decode`] on the SME quadrant
//! `word<31:29>`:
//!
//! * [`decode_tmopa`] — quadrant `100` (`0x8000_0000`), selected when the
//!   outer-product size field `word<23:22> == 0b01` (a value the base
//!   `FMOPA`/`BFMOPA` decoder leaves unallocated).
//! * [`decode_mul`] — quadrant `110` (`0xC000_0000`) with `word<24> == 1` (the
//!   `0xC1xx_xxxx` region, unallocated by the base `MOVA`/`ADDHA`/`ADDVA`
//!   decoder).
//!
//! ## Table-driven, provably exact
//!
//! The `0xC1` region is **densely** packed with neighbouring SME2 families
//! (`FDOT`/`SDOT`/`UCLAMP`/`SEL`/`ZIP`/...) that are out of scope here. Rather
//! than a fragile structural decision tree, each in-scope encoding row is matched
//! by an exact `(mask, val)` opcode key in [`SME2_FORMS`]. The full key set was
//! derived from `llvm-mc --mattr=+all` (single-bit-flip opcode/operand
//! classification) and verified **conflict-free** and free of false-accepts /
//! mis-classifications over a multi-million-word differential sweep, so a match
//! is unambiguous and never claims a neighbour or an `llvm`-invalid word.
//!
//! Each matched [`Form`] also carries the operand-field bitmasks (`ws`, `off`,
//! `zn`, `zm`, `idx`, `za`, `zk`); operand values are gathered with [`pext`]
//! (and re-scattered by the encoder with [`pdep`]). The masks' ascending bit
//! order reproduces the architecture's split index fields exactly. Every path is
//! total and panic-free.

use crate::enums::VectorArrangement as VA;
use crate::instruction::Instruction;
use crate::mnemonic::Code;
use crate::operand::{Operand, SliceIndicator};
use crate::register::{gp_register, sve_register, RegWidth};

/// Operand-shape of an SME2 multiply / outer-product form: how the source
/// operands after the `za` destination are laid out.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Sh {
    /// `<Zn>.<T>, <Zm>.<T>` — two single vectors.
    SingleSingle,
    /// `<Zn>.<T>, <Zm>.<T>[<idx>]` — single vector, indexed single.
    SingleIdx,
    /// `{ <Zn>.. }, <Zm>.<T>` — a vector group and a single vector.
    GroupSingle,
    /// `{ <Zn>.. }, <Zm>.<T>[<idx>]` — a vector group and an indexed single.
    GroupIdx,
    /// `{ <Zn>.. }, { <Zm>.. }` — two vector groups.
    GroupGroup,
    /// `{ <Zn>.. }` — a single vector group, no second source (the SME2
    /// `ADD`/`SUB`/`FADD`/`FSUB` accumulate-into-ZA two-operand forms).
    GroupOnly,
    /// `{ <Zn>, <Zn+1> }, <Zm>.<T>[<idx>]` — a **two-register** vector group and
    /// an indexed single, used by the FP8 `FVDOTB`/`FVDOTT` whose `za.<T>[...]`
    /// destination carries the `vgx4` qualifier ([`Form::vg`] is `4`) even though
    /// the source group lists only two registers.
    GroupIdxB,
    /// `<ZAda>, { <Zn>, <Zn+1> }.<T>, <Zm>.<T>, <Zk>[<idx>]` — `*TMOPA`.
    Tmopa,
}

/// One SME2 encoding row: an exact `(mask, val)` opcode key plus the operand
/// layout and per-field bitmasks used to gather/scatter operand values.
pub struct Form {
    /// Opcode mask: the bits that must equal [`val`](Form::val) to match.
    pub mask: u32,
    /// Opcode value the masked word must equal.
    pub val: u32,
    /// The [`Code`] this row decodes to.
    pub code: Code,
    /// Operand layout.
    pub shape: Sh,
    /// Accumulator (`za`) element arrangement.
    pub acc: VA,
    /// Source element arrangement.
    pub src: VA,
    /// ZA-array slice span (1/2/4); the rendered offset is `pext(off)*span`.
    pub span: u8,
    /// Multi-vector qualifier (`0`/`2`/`4`).
    pub vg: u8,
    /// `Ws` slice-select field bits.
    pub ws: u32,
    /// Slice-offset field bits (value is the slice offset divided by `span`).
    pub off: u32,
    /// `Zn` (single or group base) field bits.
    pub zn: u32,
    /// `Zm` (single or group base) field bits.
    pub zm: u32,
    /// Element-index field bits (split fields gathered in ascending bit order).
    pub idx: u32,
    /// `ZAda` tile-number field bits (`*TMOPA`).
    pub za: u32,
    /// Restricted `Zk` field bits (`*TMOPA`).
    pub zk: u32,
}

/// Parallel-bit-extract: gather the bits of `word` selected by `mask` into a
/// dense value, in ascending bit order (low mask bit → value bit 0). This is the
/// `pext` operation; for the split index fields the ascending order reproduces
/// the architecture's `hi:lo` index encoding exactly.
#[inline]
pub fn pext(word: u32, mask: u32) -> u32 {
    let mut m = mask;
    let mut out = 0u32;
    let mut pos = 0u32;
    while m != 0 {
        let lsb = m & m.wrapping_neg();
        if word & lsb != 0 {
            out |= 1 << pos;
        }
        pos += 1;
        m &= m - 1;
    }
    out
}

/// Parallel-bit-deposit: scatter the low bits of `val` into the positions
/// selected by `mask`, in ascending bit order (the inverse of [`pext`]).
#[inline]
pub fn pdep(val: u32, mask: u32) -> u32 {
    let mut m = mask;
    let mut out = 0u32;
    let mut pos = 0u32;
    while m != 0 {
        let lsb = m & m.wrapping_neg();
        if val & (1 << pos) != 0 {
            out |= lsb;
        }
        pos += 1;
        m &= m - 1;
    }
    out
}

/// The SME tile size-code nibble used by [`Operand::SmeTile`] (`1`=>`.h`,
/// `2`=>`.s`), mirroring the base SME decoder's packing.
#[inline]
const fn size_code(arr: VA) -> u16 {
    match arr {
        VA::Sh => 1,
        VA::Ss => 2,
        VA::Sd => 3,
        VA::Sq => 4,
        VA::Sb => 5,
        _ => 0,
    }
}

#[inline]
fn za_slice(f: &Form, word: u32) -> Operand {
    Operand::SmeZaSlice {
        arr: Some(f.acc),
        sel: gp_register(false, RegWidth::W32, (8 + pext(word, f.ws)) as u8),
        off: (pext(word, f.off) * f.span as u32) as u8,
        span: f.span,
        vg: f.vg,
    }
}

#[inline]
fn zsrc(n: u32, arr: VA) -> Operand {
    Operand::Reg {
        reg: sve_register(n as u8),
        arr: Some(arr),
        lane: None,
        shift: None,
        extend: None,
        pred: None,
    }
}

#[inline]
fn zsrc_idx(n: u32, arr: VA, idx: u32) -> Operand {
    Operand::Reg {
        reg: sve_register(n as u8),
        arr: Some(arr),
        lane: Some(idx as u8),
        shift: None,
        extend: None,
        pred: None,
    }
}

#[inline]
fn zgroup(first: u32, count: u8, arr: VA) -> Operand {
    Operand::SveVecGroup {
        first: sve_register(first as u8),
        count,
        arr: Some(arr),
        // LLVM renders a 4-register group as a `z.. - z..` range only when it does
        // not wrap past z31; a wrapping 4-group (and every 2-group) prints as a
        // comma list.
        range: count == 4 && (first + 3) < 32,
    }
}

/// The base register of a multi-vector group whose encoded field is `mask`:
/// `pext(word, mask) * scale`, where `scale = 2^(5 - popcount(mask))` accounts
/// for the stride of the encodable bases (a 5-bit field → stride 1 / any base; a
/// 4-bit field → stride 2 / even base; a 3-bit field → stride 4).
#[inline]
fn group_base(word: u32, mask: u32) -> u32 {
    let scale = 1u32 << (5 - mask.count_ones());
    pext(word, mask) * scale
}

/// Build the operands for a matched [`Form`] from `word`.
fn build(f: &Form, word: u32, out: &mut Instruction) {
    out.set(f.code);
    if f.shape == Sh::Tmopa {
        // ZAda, { Zn, Zn+1 }.<src>, Zm.<src>, Zk[idx].
        out.push_operand(Operand::SmeTile {
            tile: pext(word, f.za) as u16 | (size_code(f.acc) << 4),
            slice: SliceIndicator::None,
        });
        out.push_operand(zgroup(group_base(word, f.zn), 2, f.src));
        out.push_operand(zsrc(pext(word, f.zm), f.src));
        // Restricted Zk: 0..3 → z20..z23, 4..7 → z28..z31. Rendered with a lane
        // but no arrangement suffix (`z20[1]`).
        let zkf = pext(word, f.zk);
        let zk = if zkf < 4 { 20 + zkf } else { 28 + (zkf - 4) };
        out.push_operand(Operand::Reg {
            reg: sve_register(zk as u8),
            arr: None,
            lane: Some(pext(word, f.idx) as u8),
            shift: None,
            extend: None,
            pred: None,
        });
        return;
    }
    out.push_operand(za_slice(f, word));
    // Single sources read the 4/5-bit register field directly; group sources scale
    // by the field width (see `group_base`).
    match f.shape {
        Sh::SingleSingle => {
            out.push_operand(zsrc(pext(word, f.zn), f.src));
            out.push_operand(zsrc(pext(word, f.zm), f.src));
        }
        Sh::SingleIdx => {
            out.push_operand(zsrc(pext(word, f.zn), f.src));
            out.push_operand(zsrc_idx(pext(word, f.zm), f.src, pext(word, f.idx)));
        }
        Sh::GroupSingle => {
            out.push_operand(zgroup(group_base(word, f.zn), f.vg, f.src));
            out.push_operand(zsrc(pext(word, f.zm), f.src));
        }
        Sh::GroupIdx => {
            out.push_operand(zgroup(group_base(word, f.zn), f.vg, f.src));
            out.push_operand(zsrc_idx(pext(word, f.zm), f.src, pext(word, f.idx)));
        }
        Sh::GroupGroup => {
            out.push_operand(zgroup(group_base(word, f.zn), f.vg, f.src));
            out.push_operand(zgroup(group_base(word, f.zm), f.vg, f.src));
        }
        Sh::GroupOnly => {
            out.push_operand(zgroup(group_base(word, f.zn), f.vg, f.src));
        }
        Sh::GroupIdxB => {
            // FP8 FVDOTB/FVDOTT: the source list is always a two-register group
            // even though the `za` destination is `vgx4`.
            out.push_operand(zgroup(group_base(word, f.zn), 2, f.src));
            out.push_operand(zsrc_idx(pext(word, f.zm), f.src, pext(word, f.idx)));
        }
        Sh::Tmopa => {}
    }
}

/// Look up the [`Form`] matching `word` (exact `(mask,val)` key), if any.
#[inline]
fn lookup(word: u32) -> Option<&'static Form> {
    SME2_FORMS.iter().find(|f| (word & f.mask) == f.val)
}

/// Decode an SME2 `*TMOPA` outer product (quadrant `100`, size field `01`).
pub fn decode_tmopa(word: u32, out: &mut Instruction) {
    if let Some(f) = lookup(word) {
        build(f, word, out);
    }
}

/// Decode an SME2 multi-vector multiply-into-ZA form (quadrant `110`,
/// `word<24> == 1`).
pub fn decode_mul(word: u32, out: &mut Instruction) {
    if let Some(f) = lookup(word) {
        build(f, word, out);
    }
}

/// Encode a matched SME2 [`Code`] by scattering its operand fields back into the
/// form's opcode template. Returns `None` if `code` is not an SME2 form.
pub fn form_for_code(code: Code) -> Option<&'static Form> {
    SME2_FORMS.iter().find(|f| f.code == code)
}

// Short alias for the table literal.
use Form as F;

/// The complete SME2 multi-vector + `*TMOPA` encoding table. Conflict-free,
/// differentially validated against LLVM 21.
#[rustfmt::skip]
pub static SME2_FORMS: &[Form] = &[
    F { mask: 0xffe19c38, val: 0xc1e01008, code: Code::SmeBfmlaHHOV2Gg, shape: Sh::GroupGroup, acc: VA::Sh, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09030, val: 0xc1101020, code: Code::SmeBfmlaHHOV2Gi, shape: Sh::GroupIdx, acc: VA::Sh, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc08, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1601c00, code: Code::SmeBfmlaHHOV2Gs, shape: Sh::GroupSingle, acc: VA::Sh, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1e11008, code: Code::SmeBfmlaHHOV4Gg, shape: Sh::GroupGroup, acc: VA::Sh, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09070, val: 0xc1109020, code: Code::SmeBfmlaHHOV4Gi, shape: Sh::GroupIdx, acc: VA::Sh, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0xc08, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1701c00, code: Code::SmeBfmlaHHOV4Gs, shape: Sh::GroupSingle, acc: VA::Sh, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff01018, val: 0xc1801010, code: Code::SmeBfmlalSHTSi, shape: Sh::SingleIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 0, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x8c00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1200c10, code: Code::SmeBfmlalSHTSs, shape: Sh::SingleSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 0, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c3c, val: 0xc1a00810, code: Code::SmeBfmlalSHTV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1901010, code: Code::SmeBfmlalSHTV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3c0, zm: 0xf0000, idx: 0xc04, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1200810, code: Code::SmeBfmlalSHTV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c7c, val: 0xc1a10810, code: Code::SmeBfmlalSHTV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1909010, code: Code::SmeBfmlalSHTV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x380, zm: 0xf0000, idx: 0xc04, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1300810, code: Code::SmeBfmlalSHTV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c38, val: 0xc1e01018, code: Code::SmeBfmlsHHOV2Gg, shape: Sh::GroupGroup, acc: VA::Sh, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09030, val: 0xc1101030, code: Code::SmeBfmlsHHOV2Gi, shape: Sh::GroupIdx, acc: VA::Sh, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc08, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1601c08, code: Code::SmeBfmlsHHOV2Gs, shape: Sh::GroupSingle, acc: VA::Sh, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1e11018, code: Code::SmeBfmlsHHOV4Gg, shape: Sh::GroupGroup, acc: VA::Sh, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09070, val: 0xc1109030, code: Code::SmeBfmlsHHOV4Gi, shape: Sh::GroupIdx, acc: VA::Sh, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0xc08, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1701c08, code: Code::SmeBfmlsHHOV4Gs, shape: Sh::GroupSingle, acc: VA::Sh, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff01018, val: 0xc1801018, code: Code::SmeBfmlslSHTSi, shape: Sh::SingleIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 0, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x8c00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1200c18, code: Code::SmeBfmlslSHTSs, shape: Sh::SingleSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 0, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c3c, val: 0xc1a00818, code: Code::SmeBfmlslSHTV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1901018, code: Code::SmeBfmlslSHTV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3c0, zm: 0xf0000, idx: 0xc04, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1200818, code: Code::SmeBfmlslSHTV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c7c, val: 0xc1a10818, code: Code::SmeBfmlslSHTV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1909018, code: Code::SmeBfmlslSHTV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x380, zm: 0xf0000, idx: 0xc04, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1300818, code: Code::SmeBfmlslSHTV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe0e00e, val: 0x81600008, code: Code::SmeBftmopaHH, shape: Sh::Tmopa, acc: VA::Sh, src: VA::Sh, span: 0, vg: 0, ws: 0x0, off: 0x0, zn: 0x3c0, zm: 0x1f0000, idx: 0x30, za: 0x1, zk: 0x1c00 },
    F { mask: 0xffe0e00c, val: 0x81400000, code: Code::SmeBftmopaSH, shape: Sh::Tmopa, acc: VA::Ss, src: VA::Sh, span: 0, vg: 0, ws: 0x0, off: 0x0, zn: 0x3c0, zm: 0x1f0000, idx: 0x30, za: 0x3, zk: 0x1c00 },
    F { mask: 0xffe19c38, val: 0xc1e01800, code: Code::SmeFmlaDDOV2Gg, shape: Sh::GroupGroup, acc: VA::Sd, src: VA::Sd, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09838, val: 0xc1d00000, code: Code::SmeFmlaDDOV2Gi, shape: Sh::GroupIdx, acc: VA::Sd, src: VA::Sd, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0x400, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1601800, code: Code::SmeFmlaDDOV2Gs, shape: Sh::GroupSingle, acc: VA::Sd, src: VA::Sd, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1e11800, code: Code::SmeFmlaDDOV4Gg, shape: Sh::GroupGroup, acc: VA::Sd, src: VA::Sd, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09878, val: 0xc1d08000, code: Code::SmeFmlaDDOV4Gi, shape: Sh::GroupIdx, acc: VA::Sd, src: VA::Sd, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0x400, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1701800, code: Code::SmeFmlaDDOV4Gs, shape: Sh::GroupSingle, acc: VA::Sd, src: VA::Sd, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c38, val: 0xc1a01008, code: Code::SmeFmlaHHOV2Gg, shape: Sh::GroupGroup, acc: VA::Sh, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09030, val: 0xc1101000, code: Code::SmeFmlaHHOV2Gi, shape: Sh::GroupIdx, acc: VA::Sh, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc08, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1201c00, code: Code::SmeFmlaHHOV2Gs, shape: Sh::GroupSingle, acc: VA::Sh, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1a11008, code: Code::SmeFmlaHHOV4Gg, shape: Sh::GroupGroup, acc: VA::Sh, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09070, val: 0xc1109000, code: Code::SmeFmlaHHOV4Gi, shape: Sh::GroupIdx, acc: VA::Sh, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0xc08, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1301c00, code: Code::SmeFmlaHHOV4Gs, shape: Sh::GroupSingle, acc: VA::Sh, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c38, val: 0xc1a01800, code: Code::SmeFmlaSSOV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Ss, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1500000, code: Code::SmeFmlaSSOV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Ss, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1201800, code: Code::SmeFmlaSSOV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Ss, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1a11800, code: Code::SmeFmlaSSOV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Ss, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1508000, code: Code::SmeFmlaSSOV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Ss, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1301800, code: Code::SmeFmlaSSOV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Ss, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff01010, val: 0xc1c00000, code: Code::SmeFmlalHBTSi, shape: Sh::SingleIdx, acc: VA::Sh, src: VA::Sb, span: 2, vg: 0, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x8c08, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1300c00, code: Code::SmeFmlalHBTSs, shape: Sh::SingleSingle, acc: VA::Sh, src: VA::Sb, span: 2, vg: 0, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c3c, val: 0xc1a00820, code: Code::SmeFmlalHBTV2Gg, shape: Sh::GroupGroup, acc: VA::Sh, src: VA::Sb, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09030, val: 0xc1901030, code: Code::SmeFmlalHBTV2Gi, shape: Sh::GroupIdx, acc: VA::Sh, src: VA::Sb, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3c0, zm: 0xf0000, idx: 0xc0c, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1200804, code: Code::SmeFmlalHBTV2Gs, shape: Sh::GroupSingle, acc: VA::Sh, src: VA::Sb, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c7c, val: 0xc1a10820, code: Code::SmeFmlalHBTV4Gg, shape: Sh::GroupGroup, acc: VA::Sh, src: VA::Sb, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09070, val: 0xc1909020, code: Code::SmeFmlalHBTV4Gi, shape: Sh::GroupIdx, acc: VA::Sh, src: VA::Sb, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x380, zm: 0xf0000, idx: 0xc0c, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1300804, code: Code::SmeFmlalHBTV4Gs, shape: Sh::GroupSingle, acc: VA::Sh, src: VA::Sb, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff01018, val: 0xc1801000, code: Code::SmeFmlalSHTSi, shape: Sh::SingleIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 0, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x8c00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1200c00, code: Code::SmeFmlalSHTSs, shape: Sh::SingleSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 0, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c3c, val: 0xc1a00800, code: Code::SmeFmlalSHTV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1901000, code: Code::SmeFmlalSHTV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3c0, zm: 0xf0000, idx: 0xc04, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1200800, code: Code::SmeFmlalSHTV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c7c, val: 0xc1a10800, code: Code::SmeFmlalSHTV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1909000, code: Code::SmeFmlalSHTV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x380, zm: 0xf0000, idx: 0xc04, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1300800, code: Code::SmeFmlalSHTV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff0001c, val: 0xc1400000, code: Code::SmeFmlallSBQSi, shape: Sh::SingleIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x9c00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1300400, code: Code::SmeFmlallSBQSs, shape: Sh::SingleSingle, acc: VA::Ss, src: VA::Sb, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c3e, val: 0xc1a00020, code: Code::SmeFmlallSBQV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sb, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1900020, code: Code::SmeFmlallSBQV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0xf0000, idx: 0xc06, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1200002, code: Code::SmeFmlallSBQV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c7e, val: 0xc1a10020, code: Code::SmeFmlallSBQV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sb, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1108040, code: Code::SmeFmlallSBQV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0xf0000, idx: 0xc06, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1300002, code: Code::SmeFmlallSBQV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c38, val: 0xc1e01808, code: Code::SmeFmlsDDOV2Gg, shape: Sh::GroupGroup, acc: VA::Sd, src: VA::Sd, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09838, val: 0xc1d00010, code: Code::SmeFmlsDDOV2Gi, shape: Sh::GroupIdx, acc: VA::Sd, src: VA::Sd, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0x400, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1601808, code: Code::SmeFmlsDDOV2Gs, shape: Sh::GroupSingle, acc: VA::Sd, src: VA::Sd, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1e11808, code: Code::SmeFmlsDDOV4Gg, shape: Sh::GroupGroup, acc: VA::Sd, src: VA::Sd, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09878, val: 0xc1d08010, code: Code::SmeFmlsDDOV4Gi, shape: Sh::GroupIdx, acc: VA::Sd, src: VA::Sd, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0x400, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1701808, code: Code::SmeFmlsDDOV4Gs, shape: Sh::GroupSingle, acc: VA::Sd, src: VA::Sd, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c38, val: 0xc1a01018, code: Code::SmeFmlsHHOV2Gg, shape: Sh::GroupGroup, acc: VA::Sh, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09030, val: 0xc1101010, code: Code::SmeFmlsHHOV2Gi, shape: Sh::GroupIdx, acc: VA::Sh, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc08, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1201c08, code: Code::SmeFmlsHHOV2Gs, shape: Sh::GroupSingle, acc: VA::Sh, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1a11018, code: Code::SmeFmlsHHOV4Gg, shape: Sh::GroupGroup, acc: VA::Sh, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09070, val: 0xc1109010, code: Code::SmeFmlsHHOV4Gi, shape: Sh::GroupIdx, acc: VA::Sh, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0xc08, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1301c08, code: Code::SmeFmlsHHOV4Gs, shape: Sh::GroupSingle, acc: VA::Sh, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c38, val: 0xc1a01808, code: Code::SmeFmlsSSOV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Ss, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1500010, code: Code::SmeFmlsSSOV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Ss, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1201808, code: Code::SmeFmlsSSOV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Ss, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1a11808, code: Code::SmeFmlsSSOV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Ss, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1508010, code: Code::SmeFmlsSSOV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Ss, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1301808, code: Code::SmeFmlsSSOV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Ss, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff01018, val: 0xc1801008, code: Code::SmeFmlslSHTSi, shape: Sh::SingleIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 0, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x8c00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1200c08, code: Code::SmeFmlslSHTSs, shape: Sh::SingleSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 0, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c3c, val: 0xc1a00808, code: Code::SmeFmlslSHTV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1901008, code: Code::SmeFmlslSHTV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3c0, zm: 0xf0000, idx: 0xc04, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1200808, code: Code::SmeFmlslSHTV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c7c, val: 0xc1a10808, code: Code::SmeFmlslSHTV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1909008, code: Code::SmeFmlslSHTV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x380, zm: 0xf0000, idx: 0xc04, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1300808, code: Code::SmeFmlslSHTV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe0e00e, val: 0x80600008, code: Code::SmeFtmopaHB, shape: Sh::Tmopa, acc: VA::Sh, src: VA::Sb, span: 0, vg: 0, ws: 0x0, off: 0x0, zn: 0x3c0, zm: 0x1f0000, idx: 0x30, za: 0x1, zk: 0x1c00 },
    F { mask: 0xffe0e00e, val: 0x81400008, code: Code::SmeFtmopaHH, shape: Sh::Tmopa, acc: VA::Sh, src: VA::Sh, span: 0, vg: 0, ws: 0x0, off: 0x0, zn: 0x3c0, zm: 0x1f0000, idx: 0x30, za: 0x1, zk: 0x1c00 },
    F { mask: 0xffe0e00c, val: 0x80600000, code: Code::SmeFtmopaSB, shape: Sh::Tmopa, acc: VA::Ss, src: VA::Sb, span: 0, vg: 0, ws: 0x0, off: 0x0, zn: 0x3c0, zm: 0x1f0000, idx: 0x30, za: 0x3, zk: 0x1c00 },
    F { mask: 0xffe0e00c, val: 0x81600000, code: Code::SmeFtmopaSH, shape: Sh::Tmopa, acc: VA::Ss, src: VA::Sh, span: 0, vg: 0, ws: 0x0, off: 0x0, zn: 0x3c0, zm: 0x1f0000, idx: 0x30, za: 0x3, zk: 0x1c00 },
    F { mask: 0xffe0e00c, val: 0x80400000, code: Code::SmeFtmopaSS, shape: Sh::Tmopa, acc: VA::Ss, src: VA::Ss, span: 0, vg: 0, ws: 0x0, off: 0x0, zn: 0x3c0, zm: 0x1f0000, idx: 0x30, za: 0x3, zk: 0x1c00 },
    F { mask: 0xfff01018, val: 0xc1c01000, code: Code::SmeSmlalSHTSi, shape: Sh::SingleIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 0, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x8c00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1600c00, code: Code::SmeSmlalSHTSs, shape: Sh::SingleSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 0, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c3c, val: 0xc1e00800, code: Code::SmeSmlalSHTV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1d01000, code: Code::SmeSmlalSHTV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3c0, zm: 0xf0000, idx: 0xc04, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1600800, code: Code::SmeSmlalSHTV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c7c, val: 0xc1e10800, code: Code::SmeSmlalSHTV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1d09000, code: Code::SmeSmlalSHTV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x380, zm: 0xf0000, idx: 0xc04, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1700800, code: Code::SmeSmlalSHTV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff0101c, val: 0xc1800000, code: Code::SmeSmlallDHQSi, shape: Sh::SingleIdx, acc: VA::Sd, src: VA::Sh, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x8c00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1600400, code: Code::SmeSmlallDHQSs, shape: Sh::SingleSingle, acc: VA::Sd, src: VA::Sh, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c3e, val: 0xc1e00000, code: Code::SmeSmlallDHQV2Gg, shape: Sh::GroupGroup, acc: VA::Sd, src: VA::Sh, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09838, val: 0xc1900000, code: Code::SmeSmlallDHQV2Gi, shape: Sh::GroupIdx, acc: VA::Sd, src: VA::Sh, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0xf0000, idx: 0x406, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1600000, code: Code::SmeSmlallDHQV2Gs, shape: Sh::GroupSingle, acc: VA::Sd, src: VA::Sh, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c7e, val: 0xc1e10000, code: Code::SmeSmlallDHQV4Gg, shape: Sh::GroupGroup, acc: VA::Sd, src: VA::Sh, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09878, val: 0xc1908000, code: Code::SmeSmlallDHQV4Gi, shape: Sh::GroupIdx, acc: VA::Sd, src: VA::Sh, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0xf0000, idx: 0x406, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1700000, code: Code::SmeSmlallDHQV4Gs, shape: Sh::GroupSingle, acc: VA::Sd, src: VA::Sh, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff0001c, val: 0xc1000000, code: Code::SmeSmlallSBQSi, shape: Sh::SingleIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x9c00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1200400, code: Code::SmeSmlallSBQSs, shape: Sh::SingleSingle, acc: VA::Ss, src: VA::Sb, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c3e, val: 0xc1a00000, code: Code::SmeSmlallSBQV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sb, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1100000, code: Code::SmeSmlallSBQV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0xf0000, idx: 0xc06, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1200000, code: Code::SmeSmlallSBQV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c7e, val: 0xc1a10000, code: Code::SmeSmlallSBQV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sb, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1108000, code: Code::SmeSmlallSBQV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0xf0000, idx: 0xc06, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1300000, code: Code::SmeSmlallSBQV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff01018, val: 0xc1c01008, code: Code::SmeSmlslSHTSi, shape: Sh::SingleIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 0, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x8c00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1600c08, code: Code::SmeSmlslSHTSs, shape: Sh::SingleSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 0, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c3c, val: 0xc1e00808, code: Code::SmeSmlslSHTV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1d01008, code: Code::SmeSmlslSHTV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3c0, zm: 0xf0000, idx: 0xc04, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1600808, code: Code::SmeSmlslSHTV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c7c, val: 0xc1e10808, code: Code::SmeSmlslSHTV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1d09008, code: Code::SmeSmlslSHTV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x380, zm: 0xf0000, idx: 0xc04, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1700808, code: Code::SmeSmlslSHTV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff0101c, val: 0xc1800008, code: Code::SmeSmlsllDHQSi, shape: Sh::SingleIdx, acc: VA::Sd, src: VA::Sh, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x8c00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1600408, code: Code::SmeSmlsllDHQSs, shape: Sh::SingleSingle, acc: VA::Sd, src: VA::Sh, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c3e, val: 0xc1e00008, code: Code::SmeSmlsllDHQV2Gg, shape: Sh::GroupGroup, acc: VA::Sd, src: VA::Sh, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09838, val: 0xc1900008, code: Code::SmeSmlsllDHQV2Gi, shape: Sh::GroupIdx, acc: VA::Sd, src: VA::Sh, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0xf0000, idx: 0x406, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1600008, code: Code::SmeSmlsllDHQV2Gs, shape: Sh::GroupSingle, acc: VA::Sd, src: VA::Sh, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c7e, val: 0xc1e10008, code: Code::SmeSmlsllDHQV4Gg, shape: Sh::GroupGroup, acc: VA::Sd, src: VA::Sh, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09878, val: 0xc1908008, code: Code::SmeSmlsllDHQV4Gi, shape: Sh::GroupIdx, acc: VA::Sd, src: VA::Sh, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0xf0000, idx: 0x406, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1700008, code: Code::SmeSmlsllDHQV4Gs, shape: Sh::GroupSingle, acc: VA::Sd, src: VA::Sh, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff0001c, val: 0xc1000008, code: Code::SmeSmlsllSBQSi, shape: Sh::SingleIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x9c00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1200408, code: Code::SmeSmlsllSBQSs, shape: Sh::SingleSingle, acc: VA::Ss, src: VA::Sb, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c3e, val: 0xc1a00008, code: Code::SmeSmlsllSBQV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sb, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1100008, code: Code::SmeSmlsllSBQV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0xf0000, idx: 0xc06, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1200008, code: Code::SmeSmlsllSBQV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c7e, val: 0xc1a10008, code: Code::SmeSmlsllSBQV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sb, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1108008, code: Code::SmeSmlsllSBQV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0xf0000, idx: 0xc06, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1300008, code: Code::SmeSmlsllSBQV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe0e00c, val: 0x80408000, code: Code::SmeStmopaSB, shape: Sh::Tmopa, acc: VA::Ss, src: VA::Sb, span: 0, vg: 0, ws: 0x0, off: 0x0, zn: 0x3c0, zm: 0x1f0000, idx: 0x30, za: 0x3, zk: 0x1c00 },
    F { mask: 0xffe0e00c, val: 0x80408008, code: Code::SmeStmopaSH, shape: Sh::Tmopa, acc: VA::Ss, src: VA::Sh, span: 0, vg: 0, ws: 0x0, off: 0x0, zn: 0x3c0, zm: 0x1f0000, idx: 0x30, za: 0x3, zk: 0x1c00 },
    F { mask: 0xfff0001c, val: 0xc1000014, code: Code::SmeSumlallSBQSi, shape: Sh::SingleIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x9c00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1100030, code: Code::SmeSumlallSBQV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0xf0000, idx: 0xc06, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1200014, code: Code::SmeSumlallSBQV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1108030, code: Code::SmeSumlallSBQV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0xf0000, idx: 0xc06, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1300014, code: Code::SmeSumlallSBQV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe0e00c, val: 0x80608000, code: Code::SmeSutmopaSB, shape: Sh::Tmopa, acc: VA::Ss, src: VA::Sb, span: 0, vg: 0, ws: 0x0, off: 0x0, zn: 0x3c0, zm: 0x1f0000, idx: 0x30, za: 0x3, zk: 0x1c00 },
    F { mask: 0xfff01018, val: 0xc1c01010, code: Code::SmeUmlalSHTSi, shape: Sh::SingleIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 0, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x8c00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1600c10, code: Code::SmeUmlalSHTSs, shape: Sh::SingleSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 0, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c3c, val: 0xc1e00810, code: Code::SmeUmlalSHTV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1d01010, code: Code::SmeUmlalSHTV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3c0, zm: 0xf0000, idx: 0xc04, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1600810, code: Code::SmeUmlalSHTV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c7c, val: 0xc1e10810, code: Code::SmeUmlalSHTV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1d09010, code: Code::SmeUmlalSHTV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x380, zm: 0xf0000, idx: 0xc04, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1700810, code: Code::SmeUmlalSHTV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff0101c, val: 0xc1800010, code: Code::SmeUmlallDHQSi, shape: Sh::SingleIdx, acc: VA::Sd, src: VA::Sh, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x8c00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1600410, code: Code::SmeUmlallDHQSs, shape: Sh::SingleSingle, acc: VA::Sd, src: VA::Sh, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c3e, val: 0xc1e00010, code: Code::SmeUmlallDHQV2Gg, shape: Sh::GroupGroup, acc: VA::Sd, src: VA::Sh, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09838, val: 0xc1900010, code: Code::SmeUmlallDHQV2Gi, shape: Sh::GroupIdx, acc: VA::Sd, src: VA::Sh, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0xf0000, idx: 0x406, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1600010, code: Code::SmeUmlallDHQV2Gs, shape: Sh::GroupSingle, acc: VA::Sd, src: VA::Sh, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c7e, val: 0xc1e10010, code: Code::SmeUmlallDHQV4Gg, shape: Sh::GroupGroup, acc: VA::Sd, src: VA::Sh, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09878, val: 0xc1908010, code: Code::SmeUmlallDHQV4Gi, shape: Sh::GroupIdx, acc: VA::Sd, src: VA::Sh, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0xf0000, idx: 0x406, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1700010, code: Code::SmeUmlallDHQV4Gs, shape: Sh::GroupSingle, acc: VA::Sd, src: VA::Sh, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff0001c, val: 0xc1000010, code: Code::SmeUmlallSBQSi, shape: Sh::SingleIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x9c00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1200410, code: Code::SmeUmlallSBQSs, shape: Sh::SingleSingle, acc: VA::Ss, src: VA::Sb, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c3e, val: 0xc1a00010, code: Code::SmeUmlallSBQV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sb, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1100010, code: Code::SmeUmlallSBQV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0xf0000, idx: 0xc06, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1200010, code: Code::SmeUmlallSBQV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c7e, val: 0xc1a10010, code: Code::SmeUmlallSBQV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sb, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1108010, code: Code::SmeUmlallSBQV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0xf0000, idx: 0xc06, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1300010, code: Code::SmeUmlallSBQV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff01018, val: 0xc1c01018, code: Code::SmeUmlslSHTSi, shape: Sh::SingleIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 0, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x8c00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1600c18, code: Code::SmeUmlslSHTSs, shape: Sh::SingleSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 0, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c3c, val: 0xc1e00818, code: Code::SmeUmlslSHTV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1d01018, code: Code::SmeUmlslSHTV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3c0, zm: 0xf0000, idx: 0xc04, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1600818, code: Code::SmeUmlslSHTV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 2, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c7c, val: 0xc1e10818, code: Code::SmeUmlslSHTV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1d09018, code: Code::SmeUmlslSHTV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x380, zm: 0xf0000, idx: 0xc04, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1700818, code: Code::SmeUmlslSHTV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 2, vg: 4, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff0101c, val: 0xc1800018, code: Code::SmeUmlsllDHQSi, shape: Sh::SingleIdx, acc: VA::Sd, src: VA::Sh, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x8c00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1600418, code: Code::SmeUmlsllDHQSs, shape: Sh::SingleSingle, acc: VA::Sd, src: VA::Sh, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c3e, val: 0xc1e00018, code: Code::SmeUmlsllDHQV2Gg, shape: Sh::GroupGroup, acc: VA::Sd, src: VA::Sh, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09838, val: 0xc1900018, code: Code::SmeUmlsllDHQV2Gi, shape: Sh::GroupIdx, acc: VA::Sd, src: VA::Sh, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0xf0000, idx: 0x406, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1600018, code: Code::SmeUmlsllDHQV2Gs, shape: Sh::GroupSingle, acc: VA::Sd, src: VA::Sh, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c7e, val: 0xc1e10018, code: Code::SmeUmlsllDHQV4Gg, shape: Sh::GroupGroup, acc: VA::Sd, src: VA::Sh, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09878, val: 0xc1908018, code: Code::SmeUmlsllDHQV4Gi, shape: Sh::GroupIdx, acc: VA::Sd, src: VA::Sh, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0xf0000, idx: 0x406, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1700018, code: Code::SmeUmlsllDHQV4Gs, shape: Sh::GroupSingle, acc: VA::Sd, src: VA::Sh, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff0001c, val: 0xc1000018, code: Code::SmeUmlsllSBQSi, shape: Sh::SingleIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x9c00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1200418, code: Code::SmeUmlsllSBQSs, shape: Sh::SingleSingle, acc: VA::Ss, src: VA::Sb, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c3e, val: 0xc1a00018, code: Code::SmeUmlsllSBQV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sb, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1100018, code: Code::SmeUmlsllSBQV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0xf0000, idx: 0xc06, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1200018, code: Code::SmeUmlsllSBQV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c7e, val: 0xc1a10018, code: Code::SmeUmlsllSBQV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sb, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1108018, code: Code::SmeUmlsllSBQV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0xf0000, idx: 0xc06, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1300018, code: Code::SmeUmlsllSBQV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff0001c, val: 0xc1000004, code: Code::SmeUsmlallSBQSi, shape: Sh::SingleIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x9c00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1c, val: 0xc1200404, code: Code::SmeUsmlallSBQSs, shape: Sh::SingleSingle, acc: VA::Ss, src: VA::Sb, span: 4, vg: 0, ws: 0x6000, off: 0x3, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c3e, val: 0xc1a00004, code: Code::SmeUsmlallSBQV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sb, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1100020, code: Code::SmeUsmlallSBQV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3c0, zm: 0xf0000, idx: 0xc06, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1200004, code: Code::SmeUsmlallSBQV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 4, vg: 2, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c7e, val: 0xc1a10004, code: Code::SmeUsmlallSBQV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sb, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1108020, code: Code::SmeUsmlallSBQV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x380, zm: 0xf0000, idx: 0xc06, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c1e, val: 0xc1300004, code: Code::SmeUsmlallSBQV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 4, vg: 4, ws: 0x6000, off: 0x1, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe0e00c, val: 0x81408000, code: Code::SmeUstmopaSB, shape: Sh::Tmopa, acc: VA::Ss, src: VA::Sb, span: 0, vg: 0, ws: 0x0, off: 0x0, zn: 0x3c0, zm: 0x1f0000, idx: 0x30, za: 0x3, zk: 0x1c00 },
    F { mask: 0xffe0e00c, val: 0x81608000, code: Code::SmeUtmopaSB, shape: Sh::Tmopa, acc: VA::Ss, src: VA::Sb, span: 0, vg: 0, ws: 0x0, off: 0x0, zn: 0x3c0, zm: 0x1f0000, idx: 0x30, za: 0x3, zk: 0x1c00 },
    F { mask: 0xffe0e00c, val: 0x81408008, code: Code::SmeUtmopaSH, shape: Sh::Tmopa, acc: VA::Ss, src: VA::Sh, span: 0, vg: 0, ws: 0x0, off: 0x0, zn: 0x3c0, zm: 0x1f0000, idx: 0x30, za: 0x3, zk: 0x1c00 },
    F { mask: 0xffe19c38, val: 0xc1e01810, code: Code::SmeAddDDV2Gg, shape: Sh::GroupGroup, acc: VA::Sd, src: VA::Sd, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c38, val: 0xc1e01c10, code: Code::SmeAddDDV2Go, shape: Sh::GroupOnly, acc: VA::Sd, src: VA::Sd, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1601810, code: Code::SmeAddDDV2Gs, shape: Sh::GroupSingle, acc: VA::Sd, src: VA::Sd, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1e11810, code: Code::SmeAddDDV4Gg, shape: Sh::GroupGroup, acc: VA::Sd, src: VA::Sd, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c78, val: 0xc1e11c10, code: Code::SmeAddDDV4Go, shape: Sh::GroupOnly, acc: VA::Sd, src: VA::Sd, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1701810, code: Code::SmeAddDDV4Gs, shape: Sh::GroupSingle, acc: VA::Sd, src: VA::Sd, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c38, val: 0xc1a01810, code: Code::SmeAddSSV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Ss, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c38, val: 0xc1a01c10, code: Code::SmeAddSSV2Go, shape: Sh::GroupOnly, acc: VA::Ss, src: VA::Ss, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1201810, code: Code::SmeAddSSV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Ss, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1a11810, code: Code::SmeAddSSV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Ss, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c78, val: 0xc1a11c10, code: Code::SmeAddSSV4Go, shape: Sh::GroupOnly, acc: VA::Ss, src: VA::Ss, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1301810, code: Code::SmeAddSSV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Ss, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c38, val: 0xc1a01010, code: Code::SmeBfdotSHV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1501018, code: Code::SmeBfdotSHV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1201010, code: Code::SmeBfdotSHV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1a11010, code: Code::SmeBfdotSHV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1509018, code: Code::SmeBfdotSHV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1301010, code: Code::SmeBfdotSHV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1500018, code: Code::SmeBfvdotSHV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c38, val: 0xc1e01c00, code: Code::SmeFaddDDV2Go, shape: Sh::GroupOnly, acc: VA::Sd, src: VA::Sd, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c78, val: 0xc1e11c00, code: Code::SmeFaddDDV4Go, shape: Sh::GroupOnly, acc: VA::Sd, src: VA::Sd, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c38, val: 0xc1a41c00, code: Code::SmeFaddHHV2Go, shape: Sh::GroupOnly, acc: VA::Sh, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c78, val: 0xc1a51c00, code: Code::SmeFaddHHV4Go, shape: Sh::GroupOnly, acc: VA::Sh, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c38, val: 0xc1a01c00, code: Code::SmeFaddSSV2Go, shape: Sh::GroupOnly, acc: VA::Ss, src: VA::Ss, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c78, val: 0xc1a11c00, code: Code::SmeFaddSSV4Go, shape: Sh::GroupOnly, acc: VA::Ss, src: VA::Ss, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c38, val: 0xc1a01020, code: Code::SmeFdotHBV2Gg, shape: Sh::GroupGroup, acc: VA::Sh, src: VA::Sb, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09030, val: 0xc1d00020, code: Code::SmeFdotHBV2Gi, shape: Sh::GroupIdx, acc: VA::Sh, src: VA::Sb, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc08, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1201008, code: Code::SmeFdotHBV2Gs, shape: Sh::GroupSingle, acc: VA::Sh, src: VA::Sb, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1a11020, code: Code::SmeFdotHBV4Gg, shape: Sh::GroupGroup, acc: VA::Sh, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09070, val: 0xc1109040, code: Code::SmeFdotHBV4Gi, shape: Sh::GroupIdx, acc: VA::Sh, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0xc08, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1301008, code: Code::SmeFdotHBV4Gs, shape: Sh::GroupSingle, acc: VA::Sh, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c38, val: 0xc1a01000, code: Code::SmeFdotSHV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1501008, code: Code::SmeFdotSHV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1201000, code: Code::SmeFdotSHV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1a11000, code: Code::SmeFdotSHV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1509008, code: Code::SmeFdotSHV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1301000, code: Code::SmeFdotSHV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c38, val: 0xc1e01c08, code: Code::SmeFsubDDV2Go, shape: Sh::GroupOnly, acc: VA::Sd, src: VA::Sd, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c78, val: 0xc1e11c08, code: Code::SmeFsubDDV4Go, shape: Sh::GroupOnly, acc: VA::Sd, src: VA::Sd, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c38, val: 0xc1a41c08, code: Code::SmeFsubHHV2Go, shape: Sh::GroupOnly, acc: VA::Sh, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c78, val: 0xc1a51c08, code: Code::SmeFsubHHV4Go, shape: Sh::GroupOnly, acc: VA::Sh, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c38, val: 0xc1a01c08, code: Code::SmeFsubSSV2Go, shape: Sh::GroupOnly, acc: VA::Ss, src: VA::Ss, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c78, val: 0xc1a11c08, code: Code::SmeFsubSSV4Go, shape: Sh::GroupOnly, acc: VA::Ss, src: VA::Ss, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1500008, code: Code::SmeFvdotSHV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09830, val: 0xc1d00800, code: Code::SmeFvdotbSBV4GiB, shape: Sh::GroupIdxB, acc: VA::Ss, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0x408, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09830, val: 0xc1d00810, code: Code::SmeFvdottSBV4GiB, shape: Sh::GroupIdxB, acc: VA::Ss, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0x408, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c38, val: 0xc1e01400, code: Code::SmeSdotDHV2Gg, shape: Sh::GroupGroup, acc: VA::Sd, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09838, val: 0xc1d00008, code: Code::SmeSdotDHV2Gi, shape: Sh::GroupIdx, acc: VA::Sd, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0x400, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1601400, code: Code::SmeSdotDHV2Gs, shape: Sh::GroupSingle, acc: VA::Sd, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1e11400, code: Code::SmeSdotDHV4Gg, shape: Sh::GroupGroup, acc: VA::Sd, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09878, val: 0xc1d08008, code: Code::SmeSdotDHV4Gi, shape: Sh::GroupIdx, acc: VA::Sd, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0x400, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1701400, code: Code::SmeSdotDHV4Gs, shape: Sh::GroupSingle, acc: VA::Sd, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c38, val: 0xc1a01400, code: Code::SmeSdotSBV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sb, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1501020, code: Code::SmeSdotSBV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1201400, code: Code::SmeSdotSBV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1a11400, code: Code::SmeSdotSBV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1509020, code: Code::SmeSdotSBV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1301400, code: Code::SmeSdotSBV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c38, val: 0xc1e01408, code: Code::SmeSdotSHV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1501000, code: Code::SmeSdotSHV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1601408, code: Code::SmeSdotSHV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1e11408, code: Code::SmeSdotSHV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1509000, code: Code::SmeSdotSHV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1701408, code: Code::SmeSdotSHV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c38, val: 0xc1e01818, code: Code::SmeSubDDV2Gg, shape: Sh::GroupGroup, acc: VA::Sd, src: VA::Sd, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c38, val: 0xc1e01c18, code: Code::SmeSubDDV2Go, shape: Sh::GroupOnly, acc: VA::Sd, src: VA::Sd, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1601818, code: Code::SmeSubDDV2Gs, shape: Sh::GroupSingle, acc: VA::Sd, src: VA::Sd, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1e11818, code: Code::SmeSubDDV4Gg, shape: Sh::GroupGroup, acc: VA::Sd, src: VA::Sd, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c78, val: 0xc1e11c18, code: Code::SmeSubDDV4Go, shape: Sh::GroupOnly, acc: VA::Sd, src: VA::Sd, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1701818, code: Code::SmeSubDDV4Gs, shape: Sh::GroupSingle, acc: VA::Sd, src: VA::Sd, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c38, val: 0xc1a01818, code: Code::SmeSubSSV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Ss, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c38, val: 0xc1a01c18, code: Code::SmeSubSSV2Go, shape: Sh::GroupOnly, acc: VA::Ss, src: VA::Ss, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1201818, code: Code::SmeSubSSV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Ss, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1a11818, code: Code::SmeSubSSV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Ss, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c78, val: 0xc1a11c18, code: Code::SmeSubSSV4Go, shape: Sh::GroupOnly, acc: VA::Ss, src: VA::Ss, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1301818, code: Code::SmeSubSSV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Ss, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1501038, code: Code::SmeSudotSBV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1201418, code: Code::SmeSudotSBV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1509038, code: Code::SmeSudotSBV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1301418, code: Code::SmeSudotSBV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1508038, code: Code::SmeSuvdotSBV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09878, val: 0xc1d08808, code: Code::SmeSvdotDHV4Gi, shape: Sh::GroupIdx, acc: VA::Sd, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0x400, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1500020, code: Code::SmeSvdotSHV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c38, val: 0xc1e01410, code: Code::SmeUdotDHV2Gg, shape: Sh::GroupGroup, acc: VA::Sd, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09838, val: 0xc1d00018, code: Code::SmeUdotDHV2Gi, shape: Sh::GroupIdx, acc: VA::Sd, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0x400, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1601410, code: Code::SmeUdotDHV2Gs, shape: Sh::GroupSingle, acc: VA::Sd, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1e11410, code: Code::SmeUdotDHV4Gg, shape: Sh::GroupGroup, acc: VA::Sd, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09878, val: 0xc1d08018, code: Code::SmeUdotDHV4Gi, shape: Sh::GroupIdx, acc: VA::Sd, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0x400, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1701410, code: Code::SmeUdotDHV4Gs, shape: Sh::GroupSingle, acc: VA::Sd, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c38, val: 0xc1a01410, code: Code::SmeUdotSBV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sb, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1501030, code: Code::SmeUdotSBV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1201410, code: Code::SmeUdotSBV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1a11410, code: Code::SmeUdotSBV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1509030, code: Code::SmeUdotSBV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1301410, code: Code::SmeUdotSBV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c38, val: 0xc1e01418, code: Code::SmeUdotSHV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1501010, code: Code::SmeUdotSHV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1601418, code: Code::SmeUdotSHV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1e11418, code: Code::SmeUdotSHV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1509010, code: Code::SmeUdotSHV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1701418, code: Code::SmeUdotSHV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c38, val: 0xc1a01408, code: Code::SmeUsdotSBV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sb, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1501028, code: Code::SmeUsdotSBV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1201408, code: Code::SmeUsdotSBV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1a11408, code: Code::SmeUsdotSBV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1509028, code: Code::SmeUsdotSBV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1301408, code: Code::SmeUsdotSBV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1508028, code: Code::SmeUsvdotSBV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09878, val: 0xc1d08818, code: Code::SmeUvdotDHV4Gi, shape: Sh::GroupIdx, acc: VA::Sd, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0x400, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1500030, code: Code::SmeUvdotSHV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xffe19c38, val: 0xc1a01030, code: Code::SmeFdotSBV2Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sb, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x1e0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09038, val: 0xc1500038, code: Code::SmeFdotSBV2Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1201018, code: Code::SmeFdotSBV2Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffe39c78, val: 0xc1a11030, code: Code::SmeFdotSBV4Gg, shape: Sh::GroupGroup, acc: VA::Ss, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x1c0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1508008, code: Code::SmeFdotSBV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09c18, val: 0xc1301018, code: Code::SmeFdotSBV4Gs, shape: Sh::GroupSingle, acc: VA::Ss, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x3e0, zm: 0xf0000, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09030, val: 0xc1d01020, code: Code::SmeFvdotHBV2Gi, shape: Sh::GroupIdx, acc: VA::Sh, src: VA::Sb, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0xf0000, idx: 0xc08, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1508020, code: Code::SmeSvdotSBV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
    F { mask: 0xfff09078, val: 0xc1508030, code: Code::SmeUvdotSBV4Gi, shape: Sh::GroupIdx, acc: VA::Ss, src: VA::Sb, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0xf0000, idx: 0xc00, za: 0x0, zk: 0x0 },
];

#[cfg(test)]
mod tests {
    use crate::features::FeatureSet;
    use crate::format::{BufSink, FmtFormatter, Formatter};
    use crate::{Decoder, DecoderOptions};

    /// Decode `word` and render it with the default UAL formatter.
    fn render(word: u32, buf: &mut [u8]) -> &str {
        let bytes = word.to_le_bytes();
        let mut dec = Decoder::new(&bytes, 0x1000, DecoderOptions::default());
        let insn = dec.decode();
        let n = {
            let mut sink = BufSink::new(buf);
            FmtFormatter::new().format(&insn, &mut sink);
            sink.len()
        };
        core::str::from_utf8(&buf[..n]).unwrap_or("")
    }

    #[track_caller]
    fn check(word: u32, expected: &str) {
        let mut buf = [0u8; 128];
        assert_eq!(render(word, &mut buf), expected, "word={word:#010x}");
    }

    /// Decode `word`, re-encode it, and require the exact original word back.
    #[track_caller]
    fn rt(word: u32) {
        let bytes = word.to_le_bytes();
        let mut dec = Decoder::new(&bytes, 0x1000, DecoderOptions::default());
        let insn = dec.decode();
        assert!(!insn.is_invalid(), "word {word:#010x} did not decode");
        let got = insn
            .encode()
            .unwrap_or_else(|e| panic!("encode of {word:#010x} ({:?}) failed: {e:?}", insn.code()));
        assert_eq!(got, word, "round-trip mismatch for {word:#010x} (code={:?})", insn.code());
    }

    #[test]
    fn mlall_long_long() {
        // Single (indexed / non-indexed), vgx2, vgx4 — `.b` sources into za.s.
        check(0xC1000000, "smlall  za.s[w8, 0:3], z0.b, z0.b[0]");
        check(0xC1200400, "smlall  za.s[w8, 0:3], z0.b, z0.b");
        check(0xC1100000, "smlall  za.s[w8, 0:3, vgx2], { z0.b, z1.b }, z0.b[0]");
        check(0xC1108000, "smlall  za.s[w8, 0:3, vgx4], { z0.b - z3.b }, z0.b[0]");
        // The i16i64 `.h` → za.d long-long form.
        check(0xC1800000, "smlall  za.d[w8, 0:3], z0.h, z0.h[0]");
    }

    #[test]
    fn mlal_and_fmlal_widening() {
        check(0xC1C01000, "smlal   za.s[w8, 0:1], z0.h, z0.h[0]");
        check(0xC1801000, "fmlal   za.s[w8, 0:1], z0.h, z0.h[0]");
        check(0xC1400000, "fmlall  za.s[w8, 0:3], z0.b, z0.b[0]");
        // FP8 `.b` → za.h widening FMLAL.
        check(0xC1200804, "fmlal   za.h[w8, 0:1, vgx2], { z0.b, z1.b }, z0.b");
    }

    #[test]
    fn fmla_into_za() {
        check(0xC1A01008, "fmla    za.h[w8, 0, vgx2], { z0.h, z1.h }, { z0.h, z1.h }");
        check(0xC1A11800, "fmla    za.s[w8, 0, vgx4], { z0.s - z3.s }, { z0.s - z3.s }");
        check(0xC1201C00, "fmla    za.h[w8, 0, vgx2], { z0.h, z1.h }, z0.h");
        check(0xC1101000, "fmla    za.h[w8, 0, vgx2], { z0.h, z1.h }, z0.h[0]");
        check(0xC1D00000, "fmla    za.d[w8, 0, vgx2], { z0.d, z1.d }, z0.d[0]");
    }

    #[test]
    fn tmopa_outer_products() {
        check(0x80400000, "ftmopa  za0.s, { z0.s, z1.s }, z0.s, z20[0]");
        check(0x80408000, "stmopa  za0.s, { z0.b, z1.b }, z0.b, z20[0]");
        check(0x81400008, "ftmopa  za0.h, { z0.h, z1.h }, z0.h, z20[0]");
        check(0x81608000, "utmopa  za0.s, { z0.b, z1.b }, z0.b, z20[0]");
        check(0x80608000, "sutmopa za0.s, { z0.b, z1.b }, z0.b, z20[0]");
    }

    #[test]
    fn round_trip_representatives() {
        for &w in &[
            0xC1000000u32, 0xC1200400, 0xC1100000, 0xC1108000, 0xC1800000, 0xC1C01000, 0xC1801000,
            0xC1400000, 0xC1200804, 0xC1A01008, 0xC1A11800, 0xC1201C00, 0xC1101000, 0xC1D00000,
            0x80400000, 0x80408000, 0x81400008, 0x81608000, 0x80608000, 0x81408008,
        ] {
            rt(w);
        }
    }

    #[test]
    fn za_dot_add_sub_render() {
        // GAP example words render exactly as LLVM 21 (mnemonic padded to 8).
        check(0xc12015d2, "udot    za.s[w8, 2, vgx2], { z14.b, z15.b }, z0.b");
        check(0xc1203560, "sdot    za.s[w9, 0, vgx2], { z11.b, z12.b }, z0.b");
        check(0xc1201429, "usdot   za.s[w8, 1, vgx2], { z1.b, z2.b }, z0.b");
        check(0xc120163d, "sudot   za.s[w8, 5, vgx2], { z17.b, z18.b }, z0.b");
        check(0xc150020c, "fvdot   za.s[w8, 4, vgx2], { z16.h, z17.h }, z0.h[0]");
        check(0xc1500976, "uvdot   za.s[w8, 6, vgx2], { z10.h, z11.h }, z0.h[2]");
        check(0xc15006a7, "svdot   za.s[w8, 7, vgx2], { z20.h, z21.h }, z0.h[1]");
        check(0xc150051c, "bfvdot  za.s[w8, 4, vgx2], { z8.h, z9.h }, z0.h[1]");
        check(0xc1508839, "suvdot  za.s[w8, 1, vgx4], { z0.b - z3.b }, z0.b[2]");
        check(0xc1508128, "usvdot  za.s[w8, 0, vgx4], { z8.b - z11.b }, z0.b[0]");
        check(0xc1203976, "add     za.s[w9, 6, vgx2], { z11.s, z12.s }, z0.s");
        check(0xc1201898, "sub     za.s[w8, 0, vgx2], { z4.s, z5.s }, z0.s");
        check(0xc110bc46, "fdot    za.h[w9, 6, vgx4], { z0.b - z3.b }, z0.b[6]");
        check(0xc1201094, "bfdot   za.s[w8, 4, vgx2], { z4.h, z5.h }, z0.h");
        check(0xc1d00e86, "fvdotb  za.s[w8, 6, vgx4], { z20.b, z21.b }, z0.b[2]");
        check(0xc1d00f5f, "fvdott  za.s[w8, 7, vgx4], { z26.b, z27.b }, z0.b[3]");
    }

    #[test]
    fn za_dot_add_sub_round_trip() {
        // Every canonical form + every GAP example must round-trip exactly.
        for &w in &[
            0xc1a01030, 0xc1500038, 0xc1201018, 0xc1a11030, 0xc1508008, 0xc1301018, 0xc1d01020, 0xc1508020, 0xc1508030,
            0xc12015d2, 0xc1203560, 0xc1201429, 0xc120163d, 0xc150020c, 0xc1500976, 0xc15006a7, 0xc150051c,
            0xc1508839, 0xc1508128, 0xc1203976, 0xc1201898, 0xc110bc46, 0xc1201094, 0xc1d00e86, 0xc1d00f5f,
            0xc1e01810, 0xc1e01c10, 0xc1601810, 0xc1e11810, 0xc1e11c10, 0xc1701810, 0xc1a01810, 0xc1a01c10,
            0xc1201810, 0xc1a11810, 0xc1a11c10, 0xc1301810, 0xc1a01010, 0xc1501018, 0xc1201010, 0xc1a11010,
            0xc1509018, 0xc1301010, 0xc1500018, 0xc1e01c00, 0xc1e11c00, 0xc1a41c00, 0xc1a51c00, 0xc1a01c00,
            0xc1a11c00, 0xc1a01020, 0xc1d00020, 0xc1201008, 0xc1a11020, 0xc1109040, 0xc1301008, 0xc1a01000,
            0xc1501008, 0xc1201000, 0xc1a11000, 0xc1509008, 0xc1301000, 0xc1e01c08, 0xc1e11c08, 0xc1a41c08,
            0xc1a51c08, 0xc1a01c08, 0xc1a11c08, 0xc1500008, 0xc1d00800, 0xc1d00810, 0xc1e01400, 0xc1d00008,
            0xc1601400, 0xc1e11400, 0xc1d08008, 0xc1701400, 0xc1a01400, 0xc1501020, 0xc1201400, 0xc1a11400,
            0xc1509020, 0xc1301400, 0xc1e01408, 0xc1501000, 0xc1601408, 0xc1e11408, 0xc1509000, 0xc1701408,
            0xc1e01818, 0xc1e01c18, 0xc1601818, 0xc1e11818, 0xc1e11c18, 0xc1701818, 0xc1a01818, 0xc1a01c18,
            0xc1201818, 0xc1a11818, 0xc1a11c18, 0xc1301818, 0xc1501038, 0xc1201418, 0xc1509038, 0xc1301418,
            0xc1508038, 0xc1d08808, 0xc1500020, 0xc1e01410, 0xc1d00018, 0xc1601410, 0xc1e11410, 0xc1d08018,
            0xc1701410, 0xc1a01410, 0xc1501030, 0xc1201410, 0xc1a11410, 0xc1509030, 0xc1301410, 0xc1e01418,
            0xc1501010, 0xc1601418, 0xc1e11418, 0xc1509010, 0xc1701418, 0xc1a01408, 0xc1501028, 0xc1201408,
            0xc1a11408, 0xc1509028, 0xc1301408, 0xc1508028, 0xc1d08818, 0xc1500030,
        ] {
            rt(w);
        }
    }

    #[test]
    fn feature_gate_off_leaves_invalid() {
        // With FEAT_SME2 not accepted, the multi-vector forms must not decode.
        let opts = DecoderOptions { features: FeatureSet::BASE };
        let bytes = 0xC1000000u32.to_le_bytes();
        let mut dec = Decoder::new(&bytes, 0x1000, opts);
        assert!(dec.decode().is_invalid());
    }
}
