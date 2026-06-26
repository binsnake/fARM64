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

use crate::decode::bits::{bit, bits};
use crate::enums::{ExtendType, VectorArrangement as VA};
use crate::features::{Feature, FeatureSet};
use crate::instruction::Instruction;
use crate::mnemonic::Code;
use crate::operand::{Operand, SliceIndicator, SveMemMode};
use crate::register::{gp_register, sve_register, Register, RegWidth};

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
        tile: 0,
        slice: SliceIndicator::None,
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
        stride: 1,
        lane: None,
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
///
/// The `0xC1` region also holds the SME2 multi-vector ALU family (`SEL`, the
/// `S/U/F CLAMP`s, and `ZIP`/`UZP`); those use a separate, conflict-free
/// [`AluForm`] table and are tried only when no multiply-into-ZA [`Form`]
/// matches (the two key sets are disjoint, so the order is immaterial). The
/// multi-vector saturating-rounding shift-right-narrow family
/// ([`decode_narrow_shift`]) shares the `word<15:11> == 11011` slot with `SDOT`/
/// `FDOT`/... but is distinguished by its size field (`word<23:21>`), so it is
/// tried first with a strict size validation that rejects the neighbours.
pub fn decode_mul(word: u32, features: FeatureSet, out: &mut Instruction) {
    if decode_luti6(word, features, out) {
        // matched the SME2 multi-vector LUTI6 (FEAT_LUT), stride-4 or consecutive
    } else if decode_narrow_shift(word, out) {
        // matched the shift-right-narrow family (4-vector source)
    } else if decode_narrow_shift2(word, out) {
        // matched the shift-right-narrow family (2-vector source)
    } else if decode_mvm_alu(word, features, out) {
        // matched the multi-vector × multi-vector in-place ALU family
    } else if decode_mvs_alu(word, features, out) {
        // matched the multi-vector × single-vector in-place ALU family
    } else if decode_unpk(word, out) {
        // matched the multi-vector UUNPK/SUNPK
    } else if decode_fp_cvt(word, features, out) {
        // matched the multi-vector FP convert (FCVT/FCVTN/BFCVT/BFCVTN/FCVTL)
    } else if decode_fp_cvt2(word, features, out) {
        // matched the FP8 narrow/widen convert + FP round / int-convert families
    } else if decode_cvt_narrow(word, out) {
        // matched the multi-vector saturating extract-narrow family
    } else if let Some(f) = lookup(word) {
        // The Q4 BF16 `BFADD`/`BFSUB` ZA-array accumulate rows require
        // FEAT_SME_B16B16 (the `<22>=1` siblings of the `FADD`/`FSUB` `.h` forms).
        if matches!(
            f.code,
            Code::SmeBfaddV2Go | Code::SmeBfaddV4Go | Code::SmeBfsubV2Go | Code::SmeBfsubV4Go
        ) && !features.has(Feature::SmeB16b16)
        {
            return;
        }
        build(f, word, out);
    } else if let Some(f) = alu_lookup(word) {
        // The L3 BF16 `BFMUL` rows are the `size == 00` case of the FMUL slot and
        // require FEAT_SME_B16B16 (matching the function-decoded `bf*` multi×single
        // ALU forms).
        if matches!(
            f.code,
            Code::SmeBfmulMV2 | Code::SmeBfmulMV4 | Code::SmeBfmulMVS2 | Code::SmeBfmulMVS4
        ) && !features.has(Feature::SmeB16b16)
        {
            return;
        }
        build_alu(f, word, out);
    }
}

/// Decode the SME2 multi-vector × single-vector **in-place** ALU family:
/// `<op> { Zdn..Zdn+vg-1 }, { Zdn..Zdn+vg-1 }, Zm`.
///
/// Slot key: `word<31:24> == 0xC1`, `word<15:11> == 10100` (`word<10>` selects the
/// `sqdmulh` sub-table from the `smax`/... sub-table). `word<11>` selects
/// vgx2(0)/vgx4(1); the destination/first-source share the group field
/// `word<4:1>` (vgx2, stride 2) / `word<4:2>` (vgx4, stride 4). The single
/// multiplier `Zm = word<19:16>` (`z0..z15`). The sub-opcode is `word<9:5>` and
/// `word<0>` selects signed/unsigned (or max/min). Element size is `word<23:22>`
/// (`.b` re-types the floating-point ops to their BF16 `bf*` siblings).
///
/// Returns `true` (and fills `out`) on a match, `false` otherwise.
fn decode_mvs_alu(word: u32, features: FeatureSet, out: &mut Instruction) -> bool {
    // Slot: `word<31:24> == 0xC1`, `word<21> == 1`, `word<20> == 0`,
    // `word<15:12> == 1010` (`word<11>` is the vgx2/vgx4 selector and `word<10>`
    // the sub-table selector, both decoded below). The single multiplier `Zm` is a
    // `z0..z15` register (`word<19:16>`), so `word<20>` is RES0; both `word<21>`
    // and `word<20>` are part of the key (not masked away) to avoid over-decoding
    // the reserved `word<21> == 0` / `word<20> == 1` slots.
    if word & 0xff30_f000 != 0xc120_a000 {
        return false;
    }
    let vg: u8 = if bit(word, 11) == 0 { 2 } else { 4 };
    // Destination/first-source group: vgx2 base `word<4:1>` (stride 2), vgx4 base
    // `word<4:2>` (stride 4). `word<0>` is the signed/unsigned (or max/min)
    // selector for the ops that have one, so the only RES0 low bit is `word<1>`
    // (set for vgx4, where the base starts at `word<2>`).
    let (zdn, lo_reserved) = if vg == 2 {
        (bits(word, 1, 4) * 2, 0)
    } else {
        (bits(word, 2, 3) * 4, bit(word, 1))
    };
    let table = bit(word, 10);
    let sub = bits(word, 5, 5);
    let b0 = bit(word, 0);
    let size = bits(word, 22, 2);
    let zm = bits(word, 16, 4);

    // (Code, element-arrangement policy). The integer ops take all four sizes; the
    // `f*`/`bf*` ops take `.h`/`.s`/`.d` (and the `.b == BF16` re-type).
    enum Pol {
        Int,
        Fp,
    }
    let (code, pol): (Code, Pol) = if table == 1 {
        // Sub-table 1: only `SQDMULH` (sub 0, integer, no `word<0>` selector).
        if sub != 0 || b0 != 0 {
            return false;
        }
        (if vg == 2 { Code::SmeSqdmulhMVS2 } else { Code::SmeSqdmulhMVS4 }, Pol::Int)
    } else {
        match sub {
            0 => (
                match (b0, vg) {
                    (0, 2) => Code::SmeSmaxMVS2,
                    (0, _) => Code::SmeSmaxMVS4,
                    (_, 2) => Code::SmeUmaxMVS2,
                    (_, _) => Code::SmeUmaxMVS4,
                },
                Pol::Int,
            ),
            1 => (
                match (b0, vg) {
                    (0, 2) => Code::SmeSminMVS2,
                    (0, _) => Code::SmeSminMVS4,
                    (_, 2) => Code::SmeUminMVS2,
                    (_, _) => Code::SmeUminMVS4,
                },
                Pol::Int,
            ),
            8 => (
                match (b0, vg) {
                    (0, 2) => Code::SmeFmaxMVS2,
                    (0, _) => Code::SmeFmaxMVS4,
                    (_, 2) => Code::SmeFminMVS2,
                    (_, _) => Code::SmeFminMVS4,
                },
                Pol::Fp,
            ),
            9 => (
                match (b0, vg) {
                    (0, 2) => Code::SmeFmaxnmMVS2,
                    (0, _) => Code::SmeFmaxnmMVS4,
                    (_, 2) => Code::SmeFminnmMVS2,
                    (_, _) => Code::SmeFminnmMVS4,
                },
                Pol::Fp,
            ),
            12 => {
                // FSCALE has no `word<0>` selector.
                if b0 != 0 {
                    return false;
                }
                (if vg == 2 { Code::SmeFscaleMVS2 } else { Code::SmeFscaleMVS4 }, Pol::Fp)
            }
            17 => (
                match (b0, vg) {
                    (0, 2) => Code::SmeSrshlMVS2,
                    (0, _) => Code::SmeSrshlMVS4,
                    (_, 2) => Code::SmeUrshlMVS2,
                    (_, _) => Code::SmeUrshlMVS4,
                },
                Pol::Int,
            ),
            24 => {
                // ADD has no `word<0>` selector.
                if b0 != 0 {
                    return false;
                }
                (if vg == 2 { Code::SmeAddMVS2 } else { Code::SmeAddMVS4 }, Pol::Int)
            }
            _ => return false,
        }
    };

    // The low bits of the group field (above the encodable base) are RES0.
    if lo_reserved != 0 {
        return false;
    }

    // Recover the element arrangement (and, for the `.b` floating-point case, the
    // BF16 re-typed `bf*` code).
    let (arr, code) = match pol {
        Pol::Int => (size_to_va(size), code),
        Pol::Fp => match size {
            0 => {
                // BF16 re-type (FEAT_SME_B16B16): `.b` size renders `.h`.
                if !features.has(Feature::SmeB16b16) {
                    return false;
                }
                let bf = match code {
                    Code::SmeFmaxMVS2 => Code::SmeBfmaxMVS2,
                    Code::SmeFmaxMVS4 => Code::SmeBfmaxMVS4,
                    Code::SmeFminMVS2 => Code::SmeBfminMVS2,
                    Code::SmeFminMVS4 => Code::SmeBfminMVS4,
                    Code::SmeFmaxnmMVS2 => Code::SmeBfmaxnmMVS2,
                    Code::SmeFmaxnmMVS4 => Code::SmeBfmaxnmMVS4,
                    Code::SmeFminnmMVS2 => Code::SmeBfminnmMVS2,
                    Code::SmeFminnmMVS4 => Code::SmeBfminnmMVS4,
                    Code::SmeFscaleMVS2 => Code::SmeBfscaleMVS2,
                    Code::SmeFscaleMVS4 => Code::SmeBfscaleMVS4,
                    _ => return false,
                };
                (VA::Sh, bf)
            }
            1 => (VA::Sh, code),
            2 => (VA::Ss, code),
            _ => (VA::Sd, code),
        },
    };

    if !features.has(Feature::Sme2) {
        return false;
    }

    out.set(code);
    out.push_operand(alu_group(zdn, vg, arr));
    out.push_operand(alu_group(zdn, vg, arr));
    out.push_operand(zsrc(zm, arr));
    true
}

/// Decode the SME2 multi-vector × **multi-vector** in-place ALU family:
/// `<op> { Zdn..Zdn+vg-1 }, { Zdn..Zdn+vg-1 }, { Zm..Zm+vg-1 }`.
///
/// This is the multi×MULTI sibling of [`decode_mvs_alu`]: same sub-opcode scheme
/// (so the `(table, sub, b0)` selectors match exactly), but it lives one opcode
/// slot higher (`word<15:11> == 10110` vs the multi×single `1010x`) and the second
/// source is a 2-/4-register group `Zm` rather than a `z0..z15` single.
///
/// Slot key: `word<31:24> == 0xC1`, `word<21> == 1`, `word<15:12> == 1011`.
/// `word<11>` selects vgx2(0)/vgx4(1); `word<10>` selects the `sqdmulh` sub-table.
/// The destination/first-source group is `word<4:1>` (vgx2, stride 2) /
/// `word<4:2>` (vgx4, stride 4); the second-source group `Zm` is `word<20:17>`
/// (vgx2, stride 2) / `word<20:18>` (vgx4, stride 4). `word<16>` is RES0 in vgx2;
/// `word<17>`/`word<1>` are RES0 in vgx4. The sub-opcode is `word<9:5>` and
/// `word<0>` selects signed/unsigned (or max/min); element size is `word<23:22>`
/// (`.b` re-types the floating-point ops to their BF16 `bf*` siblings).
///
/// The op set is the same as multi×single minus `ADD` (no multi×multi `add`) plus
/// `FAMAX`/`FAMIN` (sub 10), which exist only here. Returns `true` (and fills
/// `out`) on a match, `false` otherwise.
fn decode_mvm_alu(word: u32, features: FeatureSet, out: &mut Instruction) -> bool {
    // Slot: `word<31:24> == 0xC1`, `word<21> == 1`, `word<15:12> == 1011`. Both
    // `word<21>` and `word<20>` are *not* part of the key (word<20:17> is the Zm
    // group field); `word<11>` (vgx) and `word<10>` (sub-table) are decoded below.
    if word & 0xff20_f000 != 0xc120_b000 {
        return false;
    }
    let vg: u8 = if bit(word, 11) == 0 { 2 } else { 4 };
    // Destination/first-source group and second-source group, with the reserved
    // low/Zm bits. vgx2: zdn = word<4:1>*2 (word<0> is the selector), zm =
    // word<20:17>*2 (word<16> RES0). vgx4: zdn = word<4:2>*4 (word<1> RES0), zm =
    // word<20:18>*4 (word<17> RES0).
    let (zdn, zm, lo_reserved) = if vg == 2 {
        // vgx2: Zdn = word<4:1>*2 (word<0> = selector), Zm = word<20:17>*2
        // (word<16> RES0).
        (bits(word, 1, 4) * 2, bits(word, 17, 4) * 2, bit(word, 16))
    } else {
        // vgx4: Zdn = word<4:2>*4 (word<1> RES0), Zm = word<20:18>*4
        // (word<17:16> RES0).
        (bits(word, 2, 3) * 4, bits(word, 18, 3) * 4, bit(word, 1) | bits(word, 16, 2))
    };
    let table = bit(word, 10);
    let sub = bits(word, 5, 5);
    let b0 = bit(word, 0);
    let size = bits(word, 22, 2);

    // Operand-size policy: `Int` takes all four sizes; `Fp` rejects `.b` (size 00);
    // `Bf` is the BF16 `.b`→`.h` re-type (FEAT_SME_B16B16) and exists *only* at
    // size 00 — the non-`.b` floating-point `fmax`/`fmin`/`fmaxnm`/`fminnm`
    // multi×multi forms (sub 8/9) are decoded by the in-place `SME2_ALU_FORMS`
    // table, so this function decodes only their BF16 siblings (size 00) and
    // returns `false` for the FP sizes to fall through to that table.
    enum Pol {
        Int,
        Fp,
        Bf,
    }
    let (code, pol): (Code, Pol) = if table == 1 {
        // Sub-table 1: only `SQDMULH` (sub 0, integer, no `word<0>` selector).
        if sub != 0 || b0 != 0 {
            return false;
        }
        (if vg == 2 { Code::SmeSqdmulhMV2 } else { Code::SmeSqdmulhMV4 }, Pol::Int)
    } else {
        match sub {
            0 => (
                match (b0, vg) {
                    (0, 2) => Code::SmeSmaxMV2,
                    (0, _) => Code::SmeSmaxMV4,
                    (_, 2) => Code::SmeUmaxMV2,
                    (_, _) => Code::SmeUmaxMV4,
                },
                Pol::Int,
            ),
            1 => (
                match (b0, vg) {
                    (0, 2) => Code::SmeSminMV2,
                    (0, _) => Code::SmeSminMV4,
                    (_, 2) => Code::SmeUminMV2,
                    (_, _) => Code::SmeUminMV4,
                },
                Pol::Int,
            ),
            8 => (
                match (b0, vg) {
                    (0, 2) => Code::SmeBfmaxMV2,
                    (0, _) => Code::SmeBfmaxMV4,
                    (_, 2) => Code::SmeBfminMV2,
                    (_, _) => Code::SmeBfminMV4,
                },
                Pol::Bf,
            ),
            9 => (
                match (b0, vg) {
                    (0, 2) => Code::SmeBfmaxnmMV2,
                    (0, _) => Code::SmeBfmaxnmMV4,
                    (_, 2) => Code::SmeBfminnmMV2,
                    (_, _) => Code::SmeBfminnmMV4,
                },
                Pol::Bf,
            ),
            10 => (
                match (b0, vg) {
                    (0, 2) => Code::SmeFamaxMV2,
                    (0, _) => Code::SmeFamaxMV4,
                    (_, 2) => Code::SmeFaminMV2,
                    (_, _) => Code::SmeFaminMV4,
                },
                Pol::Fp,
            ),
            12 => {
                // FSCALE has no `word<0>` selector. The `.b` (size 00) case is the
                // BF16 `bfscale` (re-typed below).
                if b0 != 0 {
                    return false;
                }
                (if vg == 2 { Code::SmeFscaleMVMV2 } else { Code::SmeFscaleMVMV4 }, Pol::Fp)
            }
            17 => (
                match (b0, vg) {
                    (0, 2) => Code::SmeSrshlMV2,
                    (0, _) => Code::SmeSrshlMV4,
                    (_, 2) => Code::SmeUrshlMV2,
                    (_, _) => Code::SmeUrshlMV4,
                },
                Pol::Int,
            ),
            _ => return false,
        }
    };

    // The reserved low/Zm bits above the encodable bases must be zero.
    if lo_reserved != 0 {
        return false;
    }

    // Recover the element arrangement (and, for `fscale`'s `.b` case, the BF16
    // `bfscale` re-type).
    let (arr, code) = match pol {
        Pol::Int => (size_to_va(size), code),
        Pol::Bf => {
            // sub 8/9: BF16 only (size 00). The non-`.b` FP forms decode via
            // `SME2_ALU_FORMS`, so fall through for any other size.
            if size != 0 || !features.has(Feature::SmeB16b16) {
                return false;
            }
            (VA::Sh, code)
        }
        Pol::Fp => match size {
            0 => {
                // BF16 re-type (FEAT_SME_B16B16): `.b` size renders `.h`. Only
                // `fscale` has a BF16 sibling among the `Pol::Fp` ops; `famax`/
                // `famin` (sub 10) have none, so reject their `.b` encoding.
                if !features.has(Feature::SmeB16b16) {
                    return false;
                }
                let bf = match code {
                    Code::SmeFscaleMVMV2 => Code::SmeBfscaleMV2,
                    Code::SmeFscaleMVMV4 => Code::SmeBfscaleMV4,
                    _ => return false,
                };
                (VA::Sh, bf)
            }
            1 => (VA::Sh, code),
            2 => (VA::Ss, code),
            _ => (VA::Sd, code),
        },
    };

    if !features.has(Feature::Sme2) {
        return false;
    }

    out.set(code);
    out.push_operand(alu_group(zdn, vg, arr));
    out.push_operand(alu_group(zdn, vg, arr));
    out.push_operand(alu_group(zm, vg, arr));
    true
}

/// Decode the SME2 multi-vector unpack `UUNPK`/`SUNPK`, two-count or four-count.
/// The source element is half the destination element; `word<0>` selects
/// `SUNPK`(0)/`UUNPK`(1); size `word<23:22>` selects `.h`/`.s`/`.d` (`00`
/// unallocated). FEAT_SME2.
///
/// * vgx2 (`word<20:16> == 00101`): `{ Zd, Zd+1 }.<T>, Zn.<Tsrc>` — a 2-register
///   destination group (base `word<4:1>`, stride 2) and a single source `Zn =
///   word<9:5>`.
/// * vgx4 (`word<20:16> == 10101`): `{ Zd - Zd+3 }.<T>, { Zn, Zn+1 }.<Tsrc>` — a
///   4-register destination group (base `word<4:2>`, stride 4; `word<1>` RES0,
///   `word<0>` is the opcode) and a 2-register source group (base `word<9:6>`,
///   stride 2; `word<5>` RES0).
fn decode_unpk(word: u32, out: &mut Instruction) -> bool {
    // `word<21> == 1`, `word<19:16> == 0101`, `word<15:10> == 111000`. `word<20>`
    // is the vgx2(0)/vgx4(4) count selector, decoded below.
    if word & 0xff2f_fc00 != 0xc125_e000 {
        return false;
    }
    let (dst, src) = match bits(word, 22, 2) {
        0b01 => (VA::Sh, VA::Sb),
        0b10 => (VA::Ss, VA::Sh),
        0b11 => (VA::Sd, VA::Ss),
        _ => return false,
    };
    let unsigned = bit(word, 0) == 1;
    let code = if unsigned { Code::SmeUunpk } else { Code::SmeSunpk };
    if bit(word, 20) == 0 {
        // vgx2: 2-register destination group, single source.
        let zd = bits(word, 1, 4) * 2;
        let zn = bits(word, 5, 5);
        out.set(code);
        out.push_operand(alu_group(zd, 2, dst));
        out.push_operand(zsrc(zn, src));
    } else {
        // vgx4: 4-register destination group, 2-register source group. The dest
        // base is `word<4:2>` (stride 4); `word<1>` and `word<5>` are RES0
        // (`word<0>` is the `sunpk`/`uunpk` selector).
        if bit(word, 1) != 0 || bit(word, 5) != 0 {
            return false;
        }
        let zd = bits(word, 2, 3) * 4;
        let zn = bits(word, 6, 4) * 2;
        out.set(code);
        out.push_operand(zgroup(zd, 4, dst));
        out.push_operand(zgroup(zn, 2, src));
    }
    true
}

/// Decode the SME2 multi-vector **FP convert** between FP16/BF16 and FP32 (`FCVT`/
/// `FCVTN`/`BFCVT`/`BFCVTN` narrow, `FCVT`/`FCVTL` widen). All members pin
/// `word<31:24> == 0xC1`, `word<21> == 1`, `word<20:16> == 00000`,
/// `word<15:10> == 111000`; `word<23:22>` size picks the family:
///
/// | sz `<23:22>` | direction | `<bit>` | mnemonics | operands | feature |
/// |-|-|-|-|-|-|
/// | `00` | narrow | `<5>` | FCVT(0)/FCVTN(1) | `Zd.h, { Zn.s, Zn+1.s }` | SME2 (FCVTN: SME_F16F16) |
/// | `01` | narrow | `<5>` | BFCVT(0)/BFCVTN(1) | `Zd.h, { Zn.s, Zn+1.s }` | SME2 (BFCVTN: SME_F16F16) |
/// | `10` | widen | `<0>` | FCVT(0)/FCVTL(1) | `{ Zd.s, Zd+1.s }, Zn.h` | SME_F16F16 |
/// | `11` | — | — | reserved | — | — |
///
/// For the narrow forms the destination is a single `Zd = word<4:0>` `.h` register
/// and the source is a consecutive 2-register `.s` group with base `word<9:6> * 2`
/// (`word<5>` is the FCVT/FCVTN selector). For the widen form the destination is a
/// consecutive 2-register `.s` group with base `word<4:1> * 2` (`word<0>` is the
/// FCVT/FCVTL selector) and the source is a single `Zn = word<9:5>` `.h` register.
///
/// The interleaving variants (`FCVTN`/`BFCVTN`/`FCVTL`) and the whole widen
/// direction require FEAT_SME_F16F16; the plain narrow `FCVT`/`BFCVT` need only
/// FEAT_SME2. Returns `true` (and fills/leaves `out`) on a slot match.
fn decode_fp_cvt(word: u32, features: FeatureSet, out: &mut Instruction) -> bool {
    if word & 0xff3f_fc00 != 0xc120_e000 {
        return false;
    }
    let has_f16f16 = features.has(Feature::SmeF16f16);
    match bits(word, 22, 2) {
        // Narrow: FP32 group -> FP16/BF16 single. `<5>` picks the interleaving
        // variant (FCVTN/BFCVTN, FEAT_SME_F16F16); `word<5>==0` is FCVT/BFCVT.
        sz @ (0b00 | 0b01) => {
            let interleave = bit(word, 5) == 1;
            if interleave && !has_f16f16 {
                return true; // recognised; gated off.
            }
            let zd = bits(word, 0, 5);
            let zn = bits(word, 6, 4) * 2;
            let code = match (sz, interleave) {
                (0b00, false) => Code::SmeFcvtNarrow,
                (0b00, true) => Code::SmeFcvtnNarrowFp,
                (0b01, false) => Code::SmeBfcvtNarrow,
                _ => Code::SmeBfcvtnNarrowFp,
            };
            out.set(code);
            out.push_operand(zsrc(zd, VA::Sh));
            out.push_operand(zgroup(zn, 2, VA::Ss));
            true
        }
        // Widen: FP16 single -> FP32 group (FEAT_SME_F16F16). `<0>` picks FCVT/FCVTL.
        0b10 => {
            if !has_f16f16 {
                return true;
            }
            let zd = bits(word, 1, 4) * 2;
            let zn = bits(word, 5, 5);
            let code = if bit(word, 0) == 0 { Code::SmeFcvtWiden } else { Code::SmeFcvtlWiden };
            out.set(code);
            out.push_operand(zgroup(zd, 2, VA::Ss));
            out.push_operand(zsrc(zn, VA::Sh));
            true
        }
        // `<23:22> == 11` is reserved.
        _ => true,
    }
}

/// Decode the SME2 multi-vector FP8 convert + FP round / int-convert families that
/// share the `word<31:24> == 0xC1`, `word<15:10> == 111000` slot with
/// [`decode_fp_cvt`] but use a different `word<23:16>` opcode (so they are tried
/// after it). Three sub-families, selected by `word<23:16>`:
///
/// * **FP8 narrow** (`Zd.b, { Zn.. }`): `0x24` FCVT (2-reg `.h` src), `0x34`
///   FCVT(`<5>=0`)/FCVTN(`<5>=1`) (4-reg `.s` src), `0x64` BFCVT (2-reg `.h` src).
///   FEAT_SME_F8F16.
/// * **FP8 widen** (`{ Zd.h, Zd+1.h }, Zn.b`): `0x26` F1CVT, `0xa6` F2CVT, `0x66`
///   BF1CVT, `0xe6` BF2CVT; `word<0>` adds the long-`l` variant. FEAT_SME_F8F16.
/// * **FP round / int-convert** (`{ Zd.s.. }, { Zn.s.. }`, `.s`-only, vgx2/vgx4 by
///   `word<20>`): `0xa8/9/a/c` FRINTN/P/M/A, `0x22` SCVTF(`<5>=0`)/UCVTF(`<5>=1`),
///   `0x21` FCVTZS(`<5>=0`)/FCVTZU(`<5>=1`). FEAT_SME2.
///
/// Returns `true` (and fills/leaves `out`) on a slot+opcode match, `false`
/// otherwise (so the caller falls through to the multiply / ALU tables).
fn decode_fp_cvt2(word: u32, features: FeatureSet, out: &mut Instruction) -> bool {
    // Slot: `word<31:24> == 0xC1`, `word<15:10> == 111000`.
    if word & 0xff00_fc00 != 0xc100_e000 {
        return false;
    }
    let op = bits(word, 16, 8); // word<23:16>
    let zd = bits(word, 0, 5);
    let has_f8f16 = features.has(Feature::SmeF8f16);

    // --- FP8 narrow: Zd.b <- group. ---
    match op {
        0x24 | 0x64 => {
            // 2-register `.h` source group, base word<9:6>*2; word<5:0> RES0 except
            // the register-base bits below word<6>. word<5> must be 0.
            if bit(word, 5) != 0 {
                return false;
            }
            if !has_f8f16 {
                return true;
            }
            let zn = bits(word, 6, 4) * 2;
            out.set(if op == 0x24 { Code::SmeFcvtNarrowFp8 } else { Code::SmeBfcvtNarrowFp8 });
            out.push_operand(zsrc(zd, VA::Sb));
            out.push_operand(zgroup(zn, 2, VA::Sh));
            return true;
        }
        0x34 => {
            // 4-register `.s` source group, base word<9:7>*4; word<5> selects
            // FCVT(0)/FCVTN(1); word<6> RES0.
            if bit(word, 6) != 0 {
                return false;
            }
            if !has_f8f16 {
                return true;
            }
            let zn = bits(word, 7, 3) * 4;
            out.set(if bit(word, 5) == 0 { Code::SmeFcvtNarrowFp8 } else { Code::SmeFcvtnNarrowFp8 });
            out.push_operand(zsrc(zd, VA::Sb));
            out.push_operand(zgroup(zn, 4, VA::Ss));
            return true;
        }
        _ => {}
    }

    // --- FP8 widen: { Zd.h, Zd+1.h } <- Zn.b; word<0> = cvt/cvtl. ---
    let widen = match op {
        0x26 => Some((Code::SmeF1cvtWiden, Code::SmeF1cvtlWiden)),
        0xa6 => Some((Code::SmeF2cvtWiden, Code::SmeF2cvtlWiden)),
        0x66 => Some((Code::SmeBf1cvtWiden, Code::SmeBf1cvtlWiden)),
        0xe6 => Some((Code::SmeBf2cvtWiden, Code::SmeBf2cvtlWiden)),
        _ => None,
    };
    if let Some((cvt, cvtl)) = widen {
        // Dest 2-register `.h` group base word<4:1>*2; src single `.b` word<9:5>.
        if !has_f8f16 {
            return true;
        }
        let zdg = bits(word, 1, 4) * 2;
        let zn = bits(word, 5, 5);
        out.set(if bit(word, 0) == 0 { cvt } else { cvtl });
        out.push_operand(zgroup(zdg, 2, VA::Sh));
        out.push_operand(zsrc(zn, VA::Sb));
        return true;
    }

    // --- FP round / int-convert: { Zd.s.. } <- { Zn.s.. }, vgx2/vgx4 (word<20>). ---
    // The opcode minus the vgx4 bit (`word<20>`); reject any other size/opcode.
    let vg: u8 = if bit(word, 20) == 0 { 2 } else { 4 };
    let base_op = op & !0x10; // clear word<20>
    let usel = bit(word, 5);
    let code = match base_op {
        0xa8 => Code::SmeFrintn,
        0xa9 => Code::SmeFrintp,
        0xaa => Code::SmeFrintm,
        0xac => Code::SmeFrinta,
        0x22 => {
            if usel == 0 {
                Code::SmeScvtf
            } else {
                Code::SmeUcvtf
            }
        }
        0x21 => {
            if usel == 0 {
                Code::SmeFcvtzs
            } else {
                Code::SmeFcvtzu
            }
        }
        _ => return false,
    };
    // For frint, word<5> is RES0; for scvtf/cvt-int it is the signed selector.
    if matches!(code, Code::SmeFrintn | Code::SmeFrintp | Code::SmeFrintm | Code::SmeFrinta)
        && usel != 0
    {
        return false;
    }
    if !features.has(Feature::Sme2) {
        return true;
    }
    let (zd_base, zn_base) = if vg == 2 {
        // word<6> RES0 in vgx2 (the src group base is word<9:6>*2, so word<6> is the
        // low base bit — not reserved). word<1> is the low dest base bit. The only
        // RES0 low bits are word<0> and the unused selector bits, already handled.
        (bits(word, 1, 4) * 2, bits(word, 6, 4) * 2)
    } else {
        // vgx4: dest base word<4:2>*4 (word<1> RES0), src base word<9:7>*4
        // (word<6> RES0).
        if bit(word, 1) != 0 || bit(word, 6) != 0 {
            return false;
        }
        (bits(word, 2, 3) * 4, bits(word, 7, 3) * 4)
    };
    // word<0> RES0 for all of these.
    if bit(word, 0) != 0 {
        return false;
    }
    out.set(code);
    out.push_operand(zgroup(zd_base, vg, VA::Ss));
    out.push_operand(zgroup(zn_base, vg, VA::Ss));
    true
}

/// Decode the SME2 multi-vector saturating extract-narrow family: `<op> Zd.<th>,
/// { Zn.. }.<ts>` — a single destination and a 2- or 4-register consecutive source
/// group whose element is twice the destination element.
///
/// Slot: `word<31:24> == 0xC1`, `word<21> == 1`, `word<19:16> == 0011`,
/// `word<15:10> == 111000`; `word<20>` selects the count (`0` → 2-register
/// source, `1` → 4-register source). For the 4-register form `word<6:5>` selects
/// the operation within the size-dependent signedness family (`sqcvt`/`uqcvt`/
/// `sqcvtn`/`uqcvtn` and the signed→unsigned `sqcvtu`/`sqcvtun`), the source group
/// base is `word<9:7> * 4`, and the size `word<23:22>` chooses the element widths
/// (`00`/`01` → `.b`/`.s`, `10`/`11` → `.h`/`.d`) and the signed (`word<22> == 0`)
/// vs signed→unsigned (`word<22> == 1`) family. For the 2-register form only the
/// `.h`/`.s` `sqcvt`/`uqcvt`/`sqcvtu` forms exist: `word<5>` selects signed(0)/
/// unsigned(1), the source group base is `word<9:6> * 2`, and `word<23:22>` is
/// `00` (`sqcvt`/`uqcvt`) or `01` (`sqcvtu`). FEAT_SME2.
fn decode_cvt_narrow(word: u32, out: &mut Instruction) -> bool {
    if word & 0xff2f_fc00 != 0xc123_e000 {
        return false;
    }
    let zd = bits(word, 0, 5);
    if bit(word, 20) == 0 {
        // 2-register source form: dst `.h`, src `.s`. `word<22> == 0` → `sqcvt`/
        // `uqcvt` (`word<5>`), `word<22> == 1` → `sqcvtu` (`word<5> == 0` only);
        // `word<23>` is RES0.
        if bit(word, 23) != 0 {
            return false;
        }
        let usel = bit(word, 5);
        let code = if bit(word, 22) == 0 {
            if usel == 0 {
                Code::SmeSqcvt
            } else {
                Code::SmeUqcvt
            }
        } else if usel == 0 {
            Code::SmeSqcvtu
        } else {
            return false;
        };
        let zn = bits(word, 6, 4) * 2;
        out.set(code);
        out.push_operand(zsrc(zd, VA::Sh));
        out.push_operand(zgroup(zn, 2, VA::Ss));
        return true;
    }
    let op = bits(word, 5, 2);
    // Destination element / source element from the size field: even size → `.b`/
    // `.s`, odd/high size → `.h`/`.d`.
    let (dst, src) = if bit(word, 23) == 0 {
        (VA::Sb, VA::Ss)
    } else {
        (VA::Sh, VA::Sd)
    };
    // `(size, word<6:5>)` → mnemonic. Sizes `00`/`10` (`word<22> == 0`) name the
    // signed family (`sqcvt`/`uqcvt`/`sqcvtn`/`uqcvtn`); `01`/`11` (`word<22> == 1`)
    // name the signed→unsigned family (`sqcvtu`/`sqcvtun`, only `word<5> == 0`).
    let code = if bit(word, 22) == 0 {
        match op {
            0 => Code::SmeSqcvt,
            1 => Code::SmeUqcvt,
            2 => Code::SmeSqcvtnNarrow,
            _ => Code::SmeUqcvtnNarrow,
        }
    } else {
        match op {
            0 => Code::SmeSqcvtu,
            2 => Code::SmeSqcvtunNarrow,
            _ => return false,
        }
    };
    let zn = bits(word, 7, 3) * 4;
    out.set(code);
    out.push_operand(zsrc(zd, dst));
    out.push_operand(zgroup(zn, 4, src));
    true
}

/// Decode the SME2 multi-vector `LUTI6` (FEAT_LUT): a four-register strided
/// destination group, a two-register consecutive source group, and a
/// two-register table-pair selector with a single-bit element index.
///
/// Shape: `{ Zd, Zd+4, Zd+8, Zd+12 }.H, { Zn, Zn+1 }.H, { Zt, Zt+1 }[<index>]`.
/// Fields: the destination group base is `(word<4> << 4) | word<1:0>` (a stride-4
/// group of four `.H` registers, so the base is in `{0..3, 16..19}` and
/// `word<3:2>` is RES0); `Zn = word<9:5>` (2-register consecutive, wraps); the
/// table-pair base is `word<20:16>` (a 2-register consecutive group rendered
/// without an element suffix); the element index is `word<22>` (1 bit, `word<23>`
/// RES0). The opcode pins `word<21> == 1` and `word<15:10> == 111111`.
///
/// Returns `true` (and fills `out`) on a match, `false` otherwise.
fn decode_luti6(word: u32, features: FeatureSet, out: &mut Instruction) -> bool {
    // Both LUTI6 forms share `word<31:24> == 0xC1`, `word<21> == 1`,
    // `word<23> == 0` (RES0), `word<15:12> == 1111`; they differ in
    // `word<11:10>`: `11` (`word<15:10> == 111111`) is the stride-4 destination,
    // `01` (`word<15:10> == 111101`) the 4-register consecutive destination. Both
    // keep the `{ Zn, Zn+1 }` source pair and `{ Zt, Zt+1 }[index]` table pair.
    let strided = word & 0xff20_fc00 == 0xc120_fc00; // word<15:10> == 111111
    let consec = word & 0xff20_fc00 == 0xc120_f400; // word<15:10> == 111101
    if !(strided || consec) {
        return false;
    }
    if !features.has(Feature::Lut) {
        return false;
    }
    // `word<23>` is RES0; the element index is the single bit `word<22>`.
    if bit(word, 23) != 0 {
        return false;
    }
    // Validate the destination-base RES0 bits *before* touching `out`, so a
    // reserved encoding leaves the instruction untouched (the caller falls
    // through). The strided form's base packs `word<4>` (high) and `word<1:0>`
    // (low), with `word<3:2>` RES0; the consecutive form's base is `word<4:0>`
    // with the low two bits RES0 (a 4-register group steps by 4).
    let dest = if strided {
        if bits(word, 2, 2) != 0 {
            return false;
        }
        strided_group((bit(word, 4) << 4) | bits(word, 0, 2), 4, VA::Sh, 4)
    } else {
        if bits(word, 0, 2) != 0 {
            return false;
        }
        zgroup(bits(word, 0, 5), 4, VA::Sh)
    };
    let index = bit(word, 22) as u8;
    let zn = bits(word, 5, 5);
    let table = bits(word, 16, 5);
    out.set(if strided { Code::SmeLuti6 } else { Code::SmeLuti6Consec });
    out.push_operand(dest);
    // Source: 2-register consecutive group, `.h`.
    out.push_operand(zgroup(zn, 2, VA::Sh));
    // Table pair `{ Zt, Zt+1 }[index]` — no element suffix, bracketed index.
    out.push_operand(Operand::SveVecGroup {
        first: sve_register(table as u8),
        count: 2,
        arr: None,
        range: false,
        stride: 1,
        lane: Some(index),
    });
    true
}

/// Decode the SME2 multi-vector saturating rounding shift-right-narrow-by-
/// immediate family: `SQRSHR`/`UQRSHR`/`SQRSHRN`/`UQRSHRN`/`SQRSHRU`/`SQRSHRUN`.
///
/// Shape: a single destination `Zd.<b|h>`, a 4-register consecutive source group
/// `{ Zn.s - Zn+3.s }` (or `.d`), and a `#shift` immediate. The destination
/// element / source element / shift range come from `word<23:21>` (a `tsz`-style
/// "highest set bit" size selector): `011` → `.b`/`.s`, shift `1..32`; `101` →
/// `.h`/`.d`, shift `33..64`; `111` → `.h`/`.d`, shift `1..32`. The mnemonic is
/// chosen by `word<10>` (`n` interleave), `word<6>` (unsigned result) and
/// `word<5>` (unsigned input); `word<6> == word<5> == 1` is unallocated.
///
/// Returns `true` (and fills `out`) on a match, `false` otherwise (leaving `out`
/// untouched so the caller can fall through to the multiply / ALU tables).
fn decode_narrow_shift(word: u32, out: &mut Instruction) -> bool {
    // Family key: `word<31:24> == 0xC1` and `word<15:11> == 11011`.
    if word & 0xff00_f800 != 0xc100_d800 {
        return false;
    }
    let n = bit(word, 10);
    let uresult = bit(word, 6);
    let uinput = bit(word, 5);
    // Unsigned result is only defined for a signed input (`SQRSHRU`/`SQRSHRUN`).
    if uresult == 1 && uinput == 1 {
        return false;
    }
    // Size + shift amount (`tsz`-style); an out-of-set size rejects the word so a
    // neighbouring `SDOT`/`FDOT`/... (other sizes) falls through.
    let imm5 = bits(word, 16, 5);
    let (dst, src, shift) = match bits(word, 21, 3) {
        0b011 => (VA::Sb, VA::Ss, 32 - imm5),
        0b101 => (VA::Sh, VA::Sd, 64 - imm5),
        0b111 => (VA::Sh, VA::Sd, 32 - imm5),
        _ => return false,
    };
    let code = match (uresult, uinput, n) {
        (0, 0, 0) => Code::SmeSqrshr,
        (0, 1, 0) => Code::SmeUqrshr,
        (0, 0, 1) => Code::SmeSqrshrn,
        (0, 1, 1) => Code::SmeUqrshrn,
        (1, 0, 0) => Code::SmeSqrshru,
        _ /* (1, 0, 1) */ => Code::SmeSqrshrun,
    };
    let zd = bits(word, 0, 5);
    // Source 4-register consecutive group, base = `word<9:7> * 4`.
    let zn = bits(word, 7, 3) * 4;
    out.set(code);
    out.push_operand(zsrc(zd, dst));
    out.push_operand(zgroup(zn, 4, src));
    out.push_operand(Operand::ShiftAmount(shift as u8));
    true
}

/// Decode the SME2 multi-vector saturating rounding shift-right-narrow with a
/// **2-vector** source: `SQRSHR`/`UQRSHR`/`SQRSHRU` `Zd.h, { Zn, Zn+1 }.s,
/// #shift`.
///
/// Slot key: `word<31:24> == 0xC1`, `word<15:11> == 11010`, `word<10> == 1`,
/// `word<23:21> == 111` (the only allocated size — destination `.h`, source `.s`).
/// `word<5>` is the unsigned input (`SQRSHR`→`UQRSHR`), `word<20>` the unsigned
/// result (`SQRSHR`→`SQRSHRU`); `word<5> == word<20> == 1` is unallocated. The
/// shift is `16 - word<19:16>` (`#1`..`#16`). The source 2-register consecutive
/// group base is `word<9:6> * 2`; `Zd = word<4:0>`. FEAT_SME2.
///
/// Returns `true` (and fills `out`) on a match, `false` otherwise.
fn decode_narrow_shift2(word: u32, out: &mut Instruction) -> bool {
    // Family key: `word<31:24> == 0xC1`, `word<15:11> == 11010`, `word<10> == 1`,
    // `word<23:21> == 111`.
    if word & 0xffe0_fc00 != 0xc1e0_d400 {
        return false;
    }
    let uinput = bit(word, 5);
    let uresult = bit(word, 20);
    // `SQRSHR`(0,0) / `UQRSHR`(uinput) / `SQRSHRU`(uresult); both set is
    // unallocated.
    let code = match (uresult, uinput) {
        (0, 0) => Code::SmeSqrshrV2,
        (0, 1) => Code::SmeUqrshrV2,
        (1, 0) => Code::SmeSqrshruV2,
        _ => return false,
    };
    let shift = 16 - bits(word, 16, 4);
    let zd = bits(word, 0, 5);
    // Source 2-register consecutive group, base = `word<9:6> * 2`.
    let zn = bits(word, 6, 4) * 2;
    out.set(code);
    out.push_operand(zsrc(zd, VA::Sh));
    out.push_operand(zgroup(zn, 2, VA::Ss));
    out.push_operand(Operand::ShiftAmount(shift as u8));
    true
}

/// Encode a matched SME2 [`Code`] by scattering its operand fields back into the
/// form's opcode template. Returns `None` if `code` is not an SME2 form.
pub fn form_for_code(code: Code) -> Option<&'static Form> {
    SME2_FORMS.iter().find(|f| f.code == code)
}

// ===========================================================================
// SME2 / SVE2.1 multi-vector ALU: SEL (predicate-as-counter), S/U/F CLAMP,
// ZIP/UZP. Quadrant `110`, `word<24> == 1` (the `0xC1` region), carved by an
// exact `(mask, val)` key set that is disjoint from `SME2_FORMS`.
// ===========================================================================

/// Operand layout of an SME2 multi-vector ALU form.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AluSh {
    /// `{ Zd.. }, PNg, { Zn.. }, { Zm.. }` — `SEL` (predicate-as-counter, two
    /// vector groups in and out).
    SelGroup,
    /// `{ Zd.. }, Zn.<T>, Zm.<T>` — `CLAMP` / 2-register `ZIP`/`UZP` (a vector
    /// group destination and two single-vector sources).
    TwoSingle,
    /// `{ Zd.. }, { Zn.. }` — 4-register `ZIP`/`UZP` (a vector group in and out).
    ZipGroup,
    /// `{ Zd.. }, { Zn.. }, { Zm.. }` — three vector groups, no predicate
    /// (SME2/SVE2 multi-vector `FMUL`).
    GroupGroup3,
    /// `{ Zd.. }, { Zn.. }, Zm.<T>` — two vector groups and a single multiplier
    /// (SME2 multi-vector × single-multiplier `FMUL`, `Zm` in `z0..z15`).
    GroupGroupSingle,
}

/// How a form's element-size suffix is recovered from the encoding.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AluArr {
    /// `word<23:22>` → `.b`/`.h`/`.s`/`.d`.
    Bhsd,
    /// `word<23:22>` → `.h`/`.s`/`.d` (floating-point clamp: `.b` is invalid).
    Fp,
    /// Always `.h` (BFloat16 clamp; the size field is fixed to `00` by the key).
    BfH,
    /// `word<23:22>` → `.b`/`.h`/`.s`/`.d`, or `.q` when `word<10> == 1` (the
    /// 2-register `ZIP`/`UZP` `Q` form; valid only with `word<23:22> == 00`).
    Zip2,
    /// `word<23:22>` → `.b`/`.h`/`.s`/`.d`, or `.q` when `word<16> == 1` (the
    /// 4-register `ZIP`/`UZP` `Q` form; valid only with `word<23:22> == 00`).
    Zip4,
}

/// One SME2 multi-vector ALU encoding row: an exact `(mask, val)` opcode key plus
/// the operand layout, element-size policy, multi-vector count and per-field
/// bitmasks (gathered with [`group_base`]/[`pext`], re-scattered with [`pdep`]).
pub struct AluForm {
    /// Opcode mask: the bits that must equal [`val`](AluForm::val) to match.
    pub mask: u32,
    /// Opcode value the masked word must equal.
    pub val: u32,
    /// The [`Code`] this row decodes to.
    pub code: Code,
    /// Operand layout.
    pub shape: AluSh,
    /// Element-size recovery policy.
    pub arr: AluArr,
    /// Multi-vector count of the register groups (`2` or `4`).
    pub vg: u8,
    /// `Zd` (destination group base) field bits.
    pub zd: u32,
    /// `PNg` predicate-as-counter field bits (`SEL` only; `0` otherwise).
    pub pn: u32,
    /// `Zn` (first source) field bits.
    pub zn: u32,
    /// `Zm` (second source) field bits (`0` for the 4-register `ZIP`/`UZP`).
    pub zm: u32,
}

/// Look up the [`AluForm`] matching `word`, if any.
#[inline]
fn alu_lookup(word: u32) -> Option<&'static AluForm> {
    SME2_ALU_FORMS.iter().find(|f| (word & f.mask) == f.val)
}

/// Encode-side lookup: the [`AluForm`] for a multi-vector ALU [`Code`], if any.
pub fn alu_form_for_code(code: Code) -> Option<&'static AluForm> {
    SME2_ALU_FORMS.iter().find(|f| f.code == code)
}

/// `size<23:22>` → element arrangement (`.b`/`.h`/`.s`/`.d`).
#[inline]
fn size_to_va(size: u32) -> VA {
    match size & 3 {
        0 => VA::Sb,
        1 => VA::Sh,
        2 => VA::Ss,
        _ => VA::Sd,
    }
}

/// Recover a form's element arrangement from `word`, or `None` if the encoding is
/// not a valid element size for this form (e.g. floating-point `.b`, or a `.q`
/// form with a non-zero size field).
#[inline]
pub fn alu_arrangement(f: &AluForm, word: u32) -> Option<VA> {
    let size = bits(word, 22, 2);
    match f.arr {
        AluArr::Bhsd => Some(size_to_va(size)),
        AluArr::Fp => match size {
            1 => Some(VA::Sh),
            2 => Some(VA::Ss),
            3 => Some(VA::Sd),
            _ => None,
        },
        AluArr::BfH => Some(VA::Sh),
        AluArr::Zip2 | AluArr::Zip4 => {
            let qbit = if f.arr == AluArr::Zip2 { bit(word, 10) } else { bit(word, 16) };
            if qbit == 1 {
                if size == 0 {
                    Some(VA::Sq)
                } else {
                    None
                }
            } else {
                Some(size_to_va(size))
            }
        }
    }
}

/// `{ Z<first> .. }` consecutive multi-vector group of `count` registers.
#[inline]
fn alu_group(first: u32, count: u8, arr: VA) -> Operand {
    Operand::SveVecGroup {
        first: sve_register(first as u8),
        count,
        arr: Some(arr),
        range: count == 4 && (first + 3) < 32,
        stride: 1,
        lane: None,
    }
}

/// `{ Z<first>, Z<first+stride>, .. }` *strided* multi-vector group (the SME2
/// non-consecutive load/store lists). `stride` is `8` for a 2-register group and
/// `4` for a 4-register group; these always render as a comma list.
#[inline]
fn strided_group(first: u32, count: u8, arr: VA, stride: u8) -> Operand {
    Operand::SveVecGroup {
        first: sve_register(first as u8),
        count,
        arr: Some(arr),
        range: false,
        stride,
        lane: None,
    }
}

/// `pn8`..`pn15` predicate-as-counter from a 3-bit `PNg` field, with optional
/// `/z` zeroing.
#[inline]
fn pn_counter(v: u32, zeroing: bool) -> Operand {
    Operand::PredCounter {
        reg: pn_register(v),
        zeroing,
        arr: None,
    }
}

/// Map a 3-bit `PNg` field (`0..=7`) to the underlying predicate `P8`..`P15`.
#[inline]
fn pn_register(v: u32) -> Register {
    match v & 7 {
        0 => Register::P8,
        1 => Register::P9,
        2 => Register::P10,
        3 => Register::P11,
        4 => Register::P12,
        5 => Register::P13,
        6 => Register::P14,
        _ => Register::P15,
    }
}

/// Build the operands for a matched [`AluForm`] from `word`.
fn build_alu(f: &AluForm, word: u32, out: &mut Instruction) {
    // Recover the element size first: an invalid size leaves the instruction
    // `Invalid` (the form's `(mask,val)` matched but the size field does not name
    // a legal arrangement for this family).
    let arr = match alu_arrangement(f, word) {
        Some(a) => a,
        None => return,
    };
    out.set(f.code);
    match f.shape {
        AluSh::SelGroup => {
            out.push_operand(alu_group(group_base(word, f.zd), f.vg, arr));
            out.push_operand(pn_counter(pext(word, f.pn), false));
            out.push_operand(alu_group(group_base(word, f.zn), f.vg, arr));
            out.push_operand(alu_group(group_base(word, f.zm), f.vg, arr));
        }
        AluSh::TwoSingle => {
            out.push_operand(alu_group(group_base(word, f.zd), f.vg, arr));
            out.push_operand(zsrc(pext(word, f.zn), arr));
            out.push_operand(zsrc(pext(word, f.zm), arr));
        }
        AluSh::ZipGroup => {
            out.push_operand(alu_group(group_base(word, f.zd), f.vg, arr));
            out.push_operand(alu_group(group_base(word, f.zn), f.vg, arr));
        }
        AluSh::GroupGroup3 => {
            out.push_operand(alu_group(group_base(word, f.zd), f.vg, arr));
            out.push_operand(alu_group(group_base(word, f.zn), f.vg, arr));
            out.push_operand(alu_group(group_base(word, f.zm), f.vg, arr));
        }
        AluSh::GroupGroupSingle => {
            out.push_operand(alu_group(group_base(word, f.zd), f.vg, arr));
            out.push_operand(alu_group(group_base(word, f.zn), f.vg, arr));
            // Single multiplier `Zm` is a plain `z0..z15` register (4-bit field).
            out.push_operand(zsrc(pext(word, f.zm), arr));
        }
    }
}

// Short alias for the ALU table literal.
use AluForm as A;

/// The SME2 / SVE2.1 multi-vector ALU encoding table. Conflict-free with
/// [`SME2_FORMS`] and within itself; differentially validated against LLVM 21.
#[rustfmt::skip]
pub static SME2_ALU_FORMS: &[AluForm] = &[
    // SEL (predicate-as-counter): { Zd }, PNg, { Zn }, { Zm }.
    A { mask: 0xff21e021, val: 0xc1208000, code: Code::SmeSelMV2, shape: AluSh::SelGroup, arr: AluArr::Bhsd, vg: 2, zd: 0x1e, pn: 0x1c00, zn: 0x3c0, zm: 0x1e0000 },
    A { mask: 0xff23e063, val: 0xc1218000, code: Code::SmeSelMV4, shape: AluSh::SelGroup, arr: AluArr::Bhsd, vg: 4, zd: 0x1c, pn: 0x1c00, zn: 0x380, zm: 0x1c0000 },
    // S/U CLAMP: { Zd }, Zn, Zm. (bit0 selects S(0)/U(1).)
    A { mask: 0xff20fc01, val: 0xc120c400, code: Code::SmeSclampMV2, shape: AluSh::TwoSingle, arr: AluArr::Bhsd, vg: 2, zd: 0x1e, pn: 0x0, zn: 0x3e0, zm: 0x1f0000 },
    A { mask: 0xff20fc01, val: 0xc120c401, code: Code::SmeUclampMV2, shape: AluSh::TwoSingle, arr: AluArr::Bhsd, vg: 2, zd: 0x1e, pn: 0x0, zn: 0x3e0, zm: 0x1f0000 },
    A { mask: 0xff20fc03, val: 0xc120cc00, code: Code::SmeSclampMV4, shape: AluSh::TwoSingle, arr: AluArr::Bhsd, vg: 4, zd: 0x1c, pn: 0x0, zn: 0x3e0, zm: 0x1f0000 },
    A { mask: 0xff20fc03, val: 0xc120cc01, code: Code::SmeUclampMV4, shape: AluSh::TwoSingle, arr: AluArr::Bhsd, vg: 4, zd: 0x1c, pn: 0x0, zn: 0x3e0, zm: 0x1f0000 },
    // BF CLAMP: { Zd }, Zn, Zm. The `.h` (BFloat16) clamp is the `size == 00`
    // case of the FCLAMP opcode slot, so its key fixes the size field and must
    // precede the FCLAMP rows (which leave the size field free).
    A { mask: 0xffe0fc01, val: 0xc120c000, code: Code::SmeBfclampMV2, shape: AluSh::TwoSingle, arr: AluArr::BfH, vg: 2, zd: 0x1e, pn: 0x0, zn: 0x3e0, zm: 0x1f0000 },
    A { mask: 0xffe0fc03, val: 0xc120c800, code: Code::SmeBfclampMV4, shape: AluSh::TwoSingle, arr: AluArr::BfH, vg: 4, zd: 0x1c, pn: 0x0, zn: 0x3e0, zm: 0x1f0000 },
    // F CLAMP: { Zd }, Zn, Zm. (Floating-point `.h`/`.s`/`.d` at size 01/10/11;
    // bit10 == 0 distinguishes it from the integer clamps. size == 00 is the
    // BFCLAMP rows above.)
    A { mask: 0xff20fc01, val: 0xc120c000, code: Code::SmeFclampMV2, shape: AluSh::TwoSingle, arr: AluArr::Fp, vg: 2, zd: 0x1e, pn: 0x0, zn: 0x3e0, zm: 0x1f0000 },
    A { mask: 0xff20fc03, val: 0xc120c800, code: Code::SmeFclampMV4, shape: AluSh::TwoSingle, arr: AluArr::Fp, vg: 4, zd: 0x1c, pn: 0x0, zn: 0x3e0, zm: 0x1f0000 },
    // ZIP/UZP, two registers: { Zd }, Zn, Zm. (bit0 selects ZIP(0)/UZP(1);
    // bit10 selects the `.q` form.)
    A { mask: 0xff20f801, val: 0xc120d000, code: Code::SmeZipMV2, shape: AluSh::TwoSingle, arr: AluArr::Zip2, vg: 2, zd: 0x1e, pn: 0x0, zn: 0x3e0, zm: 0x1f0000 },
    A { mask: 0xff20f801, val: 0xc120d001, code: Code::SmeUzpMV2, shape: AluSh::TwoSingle, arr: AluArr::Zip2, vg: 2, zd: 0x1e, pn: 0x0, zn: 0x3e0, zm: 0x1f0000 },
    // ZIP/UZP, four registers: { Zd }, { Zn }. (bit1 selects ZIP(0)/UZP(1);
    // bit16 selects the `.q` form.)
    A { mask: 0xff3efc63, val: 0xc136e000, code: Code::SmeZipMV4, shape: AluSh::ZipGroup, arr: AluArr::Zip4, vg: 4, zd: 0x1c, pn: 0x0, zn: 0x380, zm: 0x0 },
    A { mask: 0xff3efc63, val: 0xc136e002, code: Code::SmeUzpMV4, shape: AluSh::ZipGroup, arr: AluArr::Zip4, vg: 4, zd: 0x1c, pn: 0x0, zn: 0x380, zm: 0x0 },
    // SME2 BF16 multi-vector BFMUL: the `size == 00` case of the FMUL opcode slot,
    // rendered `.h` (FEAT_SME_B16B16). These rows pin the size field to `00` and
    // must precede the `AluArr::Fp` FMUL rows below (which reject `.b` by leaving
    // the instruction Invalid for size 00). { Zd }, { Zn }, { Zm } (multi×multi).
    A { mask: 0xffe1fc21, val: 0xc120e400, code: Code::SmeBfmulMV2, shape: AluSh::GroupGroup3, arr: AluArr::BfH, vg: 2, zd: 0x1e, pn: 0x0, zn: 0x3c0, zm: 0x1e0000 },
    A { mask: 0xffe3fc63, val: 0xc121e400, code: Code::SmeBfmulMV4, shape: AluSh::GroupGroup3, arr: AluArr::BfH, vg: 4, zd: 0x1c, pn: 0x0, zn: 0x380, zm: 0x1c0000 },
    // BFMUL multi×single (`size == 00` of the FMUL multi×single slot).
    A { mask: 0xffe1fc21, val: 0xc120e800, code: Code::SmeBfmulMVS2, shape: AluSh::GroupGroupSingle, arr: AluArr::BfH, vg: 2, zd: 0x1e, pn: 0x0, zn: 0x3c0, zm: 0x1e0000 },
    A { mask: 0xffe1fc63, val: 0xc121e800, code: Code::SmeBfmulMVS4, shape: AluSh::GroupGroupSingle, arr: AluArr::BfH, vg: 4, zd: 0x1c, pn: 0x0, zn: 0x380, zm: 0x1e0000 },
    // SME2/SVE2 multi-vector FMUL: { Zd }, { Zn }, { Zm } (three vector groups,
    // no predicate). `<15:10> == 111001`, `<21> == 1`, `<16>` selects vgx2(0)/
    // vgx4(1). `AluArr::Fp` rejects `.b` (size 00 is the BF16 BFMUL neighbour).
    A { mask: 0xff21fc21, val: 0xc120e400, code: Code::SmeFmulMV2, shape: AluSh::GroupGroup3, arr: AluArr::Fp, vg: 2, zd: 0x1e, pn: 0x0, zn: 0x3c0, zm: 0x1e0000 },
    A { mask: 0xff23fc63, val: 0xc121e400, code: Code::SmeFmulMV4, shape: AluSh::GroupGroup3, arr: AluArr::Fp, vg: 4, zd: 0x1c, pn: 0x0, zn: 0x380, zm: 0x1c0000 },
    // SME2 multi-vector FP min/max (multi & multi, in-place): { Zdn }, { Zdn },
    // { Zm }. The destination and first source share the `<4:1>`/`<4:2>` group
    // field (so `zn == zd`); `<5>` selects the `nm` (number) variant and `<0>`
    // selects min(1)/max(0). `<9:6> == 0100` opcode marker. `AluArr::Fp` rejects
    // `.b` (size 00). vgx4 narrows the group fields and pins `<1>`/`<17>` to 0.
    A { mask: 0xff21ffe1, val: 0xc120b100, code: Code::SmeFmaxMV2,   shape: AluSh::GroupGroup3, arr: AluArr::Fp, vg: 2, zd: 0x1e, pn: 0x0, zn: 0x1e, zm: 0x1e0000 },
    A { mask: 0xff21ffe1, val: 0xc120b101, code: Code::SmeFminMV2,   shape: AluSh::GroupGroup3, arr: AluArr::Fp, vg: 2, zd: 0x1e, pn: 0x0, zn: 0x1e, zm: 0x1e0000 },
    A { mask: 0xff21ffe1, val: 0xc120b120, code: Code::SmeFmaxnmMV2, shape: AluSh::GroupGroup3, arr: AluArr::Fp, vg: 2, zd: 0x1e, pn: 0x0, zn: 0x1e, zm: 0x1e0000 },
    A { mask: 0xff21ffe1, val: 0xc120b121, code: Code::SmeFminnmMV2, shape: AluSh::GroupGroup3, arr: AluArr::Fp, vg: 2, zd: 0x1e, pn: 0x0, zn: 0x1e, zm: 0x1e0000 },
    A { mask: 0xff23ffe3, val: 0xc120b900, code: Code::SmeFmaxMV4,   shape: AluSh::GroupGroup3, arr: AluArr::Fp, vg: 4, zd: 0x1c, pn: 0x0, zn: 0x1c, zm: 0x1c0000 },
    A { mask: 0xff23ffe3, val: 0xc120b901, code: Code::SmeFminMV4,   shape: AluSh::GroupGroup3, arr: AluArr::Fp, vg: 4, zd: 0x1c, pn: 0x0, zn: 0x1c, zm: 0x1c0000 },
    A { mask: 0xff23ffe3, val: 0xc120b920, code: Code::SmeFmaxnmMV4, shape: AluSh::GroupGroup3, arr: AluArr::Fp, vg: 4, zd: 0x1c, pn: 0x0, zn: 0x1c, zm: 0x1c0000 },
    A { mask: 0xff23ffe3, val: 0xc120b921, code: Code::SmeFminnmMV4, shape: AluSh::GroupGroup3, arr: AluArr::Fp, vg: 4, zd: 0x1c, pn: 0x0, zn: 0x1c, zm: 0x1c0000 },
    // SME2 multi-vector × single-multiplier FMUL: { Zd }, { Zn }, Zm. The single
    // multiplier `Zm` is a `z0..z15` register at `<20:17>` (4-bit) in both vgx2
    // and vgx4. `<5> == 0` and `<0> == 0` fixed; vgx4 pins `<6>`/`<1>`.
    A { mask: 0xff21fc21, val: 0xc120e800, code: Code::SmeFmulMVS2, shape: AluSh::GroupGroupSingle, arr: AluArr::Fp, vg: 2, zd: 0x1e, pn: 0x0, zn: 0x3c0, zm: 0x1e0000 },
    A { mask: 0xff21fc63, val: 0xc121e800, code: Code::SmeFmulMVS4, shape: AluSh::GroupGroupSingle, arr: AluArr::Fp, vg: 4, zd: 0x1c, pn: 0x0, zn: 0x380, zm: 0x1e0000 },
];

// ===========================================================================
// SME2 / SVE2.1 contiguous multi-vector load/store (predicate-as-counter):
// LD1{B,H,W,D} / LDNT1{B,H,W,D} / ST1{B,H,W,D} / STNT1{B,H,W,D}. Quadrant
// `101` (`0xA0`), `word<23> == 0` (the integer outer products set `word<23>`,
// so this carve does not regress them). Single-vector SVE LD1*/ST1* live in
// the SVE group (`op0 == 0b0010`, the `0xA5`/`0xE4` regions) and are untouched.
// ===========================================================================

/// Decode an SME2 contiguous *or* strided multi-vector load/store
/// (`word<31:29> == 101`, `word<23> == 0`). `word<24>` selects the strided family
/// (the non-consecutive register lists). Leaves the instruction `Invalid` for
/// words outside the handled `LD1*/LDNT1*/ST1*/STNT1*` set.
pub fn decode_mem(word: u32, out: &mut Instruction) {
    let strided = bit(word, 24) == 1;
    let is_imm = bit(word, 22) == 1;
    let is_store = bit(word, 21) == 1;
    let msz = bits(word, 13, 2);
    let count: u8 = if bit(word, 15) == 1 { 4 } else { 2 };
    let pn = bits(word, 10, 3);
    let rn = bits(word, 5, 5);
    let arr = size_to_va(msz);

    // Data register group + nontemporal flag. The two families pack these
    // differently:
    //   * consecutive (`word<24> == 0`): base in `word<4:1>` (vgx2, stride-2) or
    //     `word<4:2>` (vgx4, stride-4); `NT = word<0>`.
    //   * strided (`word<24> == 1`): base = `word<4>:word<2:0>` (vgx2) or
    //     `word<4>:word<1:0>` (vgx4) — a `{z0..7,z16..23}` / `{z0..3,z16..19}`
    //     base whose group steps by 8 / 4; `NT = word<3>`.
    let (group, is_nt) = if strided {
        let is_nt = bit(word, 3) == 1;
        if count == 4 {
            // vgx4 leaves bit<2> reserved (must be zero).
            if bit(word, 2) != 0 {
                return;
            }
            let base = (bit(word, 4) << 4) | bits(word, 0, 2);
            (strided_group(base, 4, arr, 4), is_nt)
        } else {
            let base = (bit(word, 4) << 4) | bits(word, 0, 3);
            (strided_group(base, 2, arr, 8), is_nt)
        }
    } else {
        let is_nt = bit(word, 0) == 1;
        // vgx2 packs the base in bits<4:1> (stride 2), vgx4 in bits<4:2> (stride
        // 4); bit<0> is the nontemporal flag. vgx4 leaves bit<1> reserved (must be
        // zero); reject a stray set bit so the accepted set matches LLVM exactly.
        let (zt_mask, lo_reserved) = if count == 4 { (0x1cu32, 0x2u32) } else { (0x1eu32, 0x0u32) };
        if word & lo_reserved != 0 {
            return;
        }
        (alu_group(group_base(word, zt_mask), count, arr), is_nt)
    };

    let code = match (is_store, is_nt, msz) {
        (false, false, 0) => Code::SmeLd1bMV,
        (false, false, 1) => Code::SmeLd1hMV,
        (false, false, 2) => Code::SmeLd1wMV,
        (false, false, 3) => Code::SmeLd1dMV,
        (false, true, 0) => Code::SmeLdnt1bMV,
        (false, true, 1) => Code::SmeLdnt1hMV,
        (false, true, 2) => Code::SmeLdnt1wMV,
        (false, true, 3) => Code::SmeLdnt1dMV,
        (true, false, 0) => Code::SmeSt1bMV,
        (true, false, 1) => Code::SmeSt1hMV,
        (true, false, 2) => Code::SmeSt1wMV,
        (true, false, 3) => Code::SmeSt1dMV,
        (true, true, 0) => Code::SmeStnt1bMV,
        (true, true, 1) => Code::SmeStnt1hMV,
        (true, true, 2) => Code::SmeStnt1wMV,
        _ => Code::SmeStnt1dMV,
    };

    let mem = if is_imm {
        // `[Xn{, #imm, MUL VL}]` — a 4-bit signed imm in units of the group
        // count; the displayed offset is `imm4 * count`. `word<20>` is reserved
        // (zero) in the immediate form (it is the high `Rm` bit of the scalar +
        // scalar form); reject a stray set bit to match LLVM exactly.
        if bit(word, 20) != 0 {
            return;
        }
        let imm4 = sign_extend4(bits(word, 16, 4));
        Operand::SveMem {
            base: gp_register(true, RegWidth::X64, rn as u8),
            offset: Register::None,
            arr: None,
            extend: ExtendType::Uxtx,
            imm: imm4 * count as i32,
            amount: 0,
            mode: SveMemMode::ScalarImmMulVl,
        }
    } else {
        // `[Xn, Xm{, LSL #msz}]` — scalar base + scalar index scaled by the
        // element size. The shift is shown only for the scaled element sizes.
        let rm = bits(word, 16, 5);
        let shift = if msz == 0 { 0 } else { 0x80 | msz as u8 };
        Operand::MemExt {
            base: gp_register(true, RegWidth::X64, rn as u8),
            index: gp_register(false, RegWidth::X64, rm as u8),
            extend: ExtendType::Uxtx,
            shift,
        }
    };

    out.set(code);
    out.push_operand(group);
    out.push_operand(pn_counter(pn, !is_store));
    out.push_operand(mem);
}

/// Sign-extend a 4-bit field to `i32`.
#[inline]
fn sign_extend4(v: u32) -> i32 {
    ((v as i32) << 28) >> 28
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
    // Q4: SME2 BF16 ZA-array accumulate `BFADD`/`BFSUB` — the `<22>=1` (BF16)
    // siblings of the `FADD`/`FSUB` `.h` GroupOnly forms (FEAT_SME_B16B16, gated in
    // `decode_mul`). `za.h[Ws, off, vgxN], { Zn.. }`.
    F { mask: 0xffff9c38, val: 0xc1e41c00, code: Code::SmeBfaddV2Go, shape: Sh::GroupOnly, acc: VA::Sh, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c78, val: 0xc1e51c00, code: Code::SmeBfaddV4Go, shape: Sh::GroupOnly, acc: VA::Sh, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c38, val: 0xc1e41c08, code: Code::SmeBfsubV2Go, shape: Sh::GroupOnly, acc: VA::Sh, src: VA::Sh, span: 1, vg: 2, ws: 0x6000, off: 0x7, zn: 0x3c0, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
    F { mask: 0xffff9c78, val: 0xc1e51c08, code: Code::SmeBfsubV4Go, shape: Sh::GroupOnly, acc: VA::Sh, src: VA::Sh, span: 1, vg: 4, ws: 0x6000, off: 0x7, zn: 0x380, zm: 0x0, idx: 0x0, za: 0x0, zk: 0x0 },
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
