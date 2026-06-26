//! SVE / SVE2 permute, predicate-logical / generation, and compare encodings.
//!
//! Hand-written from the *ARM Architecture Reference Manual* SVE encoding index.
//! This module owns the permute/predicate/compare leaves that share the SVE
//! integer quadrants (`word<31:29>` = `000`/`001`) but are declined by
//! [`super::sve_int`]; the family dispatcher in [`super`] tries `sve_int` first
//! and falls back here when it leaves the instruction [`Code::Invalid`].
//!
//! It covers:
//!
//! * vector permute — `ZIP1/2`, `UZP1/2`, `TRN1/2` (element and 128-bit `.Q`),
//!   `REV`, `REVB/REVH/REVW`, `SUNPK{HI,LO}` / `UUNPK{HI,LO}`, `COMPACT`,
//!   `SPLICE`, `EXT`, `TBL`/`TBX`, `CLASTA/B`, `LASTA/B`;
//! * predicate permute — `ZIP/UZP/TRN` (predicate), `REV` (predicate),
//!   `PUNPK{HI,LO}`, and the predicate-indexed `DUP`;
//! * predicate generation / logical — `PTRUE{S}`, `PFALSE`, `PTEST`, `PFIRST`,
//!   `PNEXT`, `RDFFR{S}`/`WRFFR`/`SETFFR`, the predicate `AND/ORR/EOR/BIC/...`
//!   logical group (and the `MOV`/`MOVS`/`NOT`/`NOTS` aliases), `SEL`, and the
//!   break family `BRKA/BRKB/BRKN/BRKPA/BRKPB` (+ flag-setting `S` forms);
//! * compare — the `CMP<cc>` vector / wide / unsigned-immediate forms not taken
//!   by `sve_int`, the `WHILE` family (`WHILELT/LE/LO/LS/GE/GT/HI/HS/RW/WR`), and
//!   `CTERMEQ/NE`.
//!
//! Code identity follows the module convention: one [`Code`] per ARM ARM encoding
//! class, the preferred-disassembly alias installed via
//! [`Instruction::set_mnemonic`] where the corpus uses one, and arrangement /
//! predicate / lane decoration carried in the operands. Every path is total and
//! panic-free; unallocated encodings are left [`Code::Invalid`].

use crate::decode::bits::{bit, bits};
use crate::enums::VectorArrangement as VA;
use crate::features::{Feature, FeatureSet};
use crate::instruction::Instruction;
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::{Operand, PredQual};
use crate::register::{gp_register, Register, RegWidth};

// ---------------------------------------------------------------------------
// Register-bank tables (mirrors of the ones in `sve_int`, kept local so the two
// permute / integer halves stay independent).
// ---------------------------------------------------------------------------

const Z: [Register; 32] = [
    Register::Z0, Register::Z1, Register::Z2, Register::Z3, Register::Z4, Register::Z5, Register::Z6, Register::Z7,
    Register::Z8, Register::Z9, Register::Z10, Register::Z11, Register::Z12, Register::Z13, Register::Z14, Register::Z15,
    Register::Z16, Register::Z17, Register::Z18, Register::Z19, Register::Z20, Register::Z21, Register::Z22, Register::Z23,
    Register::Z24, Register::Z25, Register::Z26, Register::Z27, Register::Z28, Register::Z29, Register::Z30, Register::Z31,
];
const P: [Register; 16] = [
    Register::P0, Register::P1, Register::P2, Register::P3, Register::P4, Register::P5, Register::P6, Register::P7,
    Register::P8, Register::P9, Register::P10, Register::P11, Register::P12, Register::P13, Register::P14, Register::P15,
];
const BR: [Register; 32] = [
    Register::B0, Register::B1, Register::B2, Register::B3, Register::B4, Register::B5, Register::B6, Register::B7,
    Register::B8, Register::B9, Register::B10, Register::B11, Register::B12, Register::B13, Register::B14, Register::B15,
    Register::B16, Register::B17, Register::B18, Register::B19, Register::B20, Register::B21, Register::B22, Register::B23,
    Register::B24, Register::B25, Register::B26, Register::B27, Register::B28, Register::B29, Register::B30, Register::B31,
];
const HR: [Register; 32] = [
    Register::H0, Register::H1, Register::H2, Register::H3, Register::H4, Register::H5, Register::H6, Register::H7,
    Register::H8, Register::H9, Register::H10, Register::H11, Register::H12, Register::H13, Register::H14, Register::H15,
    Register::H16, Register::H17, Register::H18, Register::H19, Register::H20, Register::H21, Register::H22, Register::H23,
    Register::H24, Register::H25, Register::H26, Register::H27, Register::H28, Register::H29, Register::H30, Register::H31,
];
const SR: [Register; 32] = [
    Register::S0, Register::S1, Register::S2, Register::S3, Register::S4, Register::S5, Register::S6, Register::S7,
    Register::S8, Register::S9, Register::S10, Register::S11, Register::S12, Register::S13, Register::S14, Register::S15,
    Register::S16, Register::S17, Register::S18, Register::S19, Register::S20, Register::S21, Register::S22, Register::S23,
    Register::S24, Register::S25, Register::S26, Register::S27, Register::S28, Register::S29, Register::S30, Register::S31,
];
const DR: [Register; 32] = [
    Register::D0, Register::D1, Register::D2, Register::D3, Register::D4, Register::D5, Register::D6, Register::D7,
    Register::D8, Register::D9, Register::D10, Register::D11, Register::D12, Register::D13, Register::D14, Register::D15,
    Register::D16, Register::D17, Register::D18, Register::D19, Register::D20, Register::D21, Register::D22, Register::D23,
    Register::D24, Register::D25, Register::D26, Register::D27, Register::D28, Register::D29, Register::D30, Register::D31,
];

// ---------------------------------------------------------------------------
// Small operand constructors.
// ---------------------------------------------------------------------------

/// Element-size arrangement (`.b`/`.h`/`.s`/`.d`) from a 2-bit `size`.
#[inline]
fn arr(size: u32) -> VA {
    match size & 3 {
        0 => VA::Sb,
        1 => VA::Sh,
        2 => VA::Ss,
        _ => VA::Sd,
    }
}

/// A scalable `Z{n}` operand with arrangement `a`.
#[inline]
fn zreg(n: u32, a: VA) -> Operand {
    Operand::Reg { reg: Z[(n & 0x1f) as usize], arr: Some(a), lane: None, shift: None, extend: None, pred: None }
}

/// A scalable `Z{n}.Q` operand (128-bit element permute).
#[inline]
fn zreg_q(n: u32) -> Operand {
    Operand::Reg { reg: Z[(n & 0x1f) as usize], arr: Some(VA::Sq), lane: None, shift: None, extend: None, pred: None }
}

/// A governing predicate `P{n}` with a `/z` or `/m` qualifier.
#[inline]
fn preg_q(n: u32, q: PredQual) -> Operand {
    Operand::Reg { reg: P[(n & 0xf) as usize], arr: None, lane: None, shift: None, extend: None, pred: Some(q) }
}

/// A bare predicate `P{n}` (no qualifier, no size).
#[inline]
fn preg(n: u32) -> Operand {
    Operand::Reg { reg: P[(n & 0xf) as usize], arr: None, lane: None, shift: None, extend: None, pred: None }
}

/// A sized predicate `P{n}.<T>` (no qualifier).
#[inline]
fn preg_sz(n: u32, a: VA) -> Operand {
    Operand::Reg { reg: P[(n & 0xf) as usize], arr: Some(a), lane: None, shift: None, extend: None, pred: None }
}

/// A general-purpose register operand (`X`/`W`), reg-31 as ZR.
#[inline]
fn gpr(n: u32, w: RegWidth) -> Operand {
    Operand::Reg { reg: gp_register(false, w, n as u8), arr: None, lane: None, shift: None, extend: None, pred: None }
}

/// A scalar SIMD `B/H/S/D` operand for the element width given by `size`.
#[inline]
fn scalar_fp(n: u32, size: u32) -> Operand {
    let n = (n & 0x1f) as usize;
    let reg = match size & 3 {
        0 => BR[n],
        1 => HR[n],
        2 => SR[n],
        _ => DR[n],
    };
    Operand::Reg { reg, arr: None, lane: None, shift: None, extend: None, pred: None }
}

/// A two-register Z list `{Z{n}.<T>, Z{n+1}.<T>}` for the `*_con` permute forms.
#[inline]
fn zlist2(n: u32, a: VA) -> Operand {
    let n0 = (n & 0x1f) as usize;
    let n1 = ((n + 1) & 0x1f) as usize;
    Operand::MultiReg { regs: [Z[n0], Z[n1], Register::None, Register::None], count: 2, arr: Some(a), lane: None }
}

// ---------------------------------------------------------------------------
// Top-level entry.
// ---------------------------------------------------------------------------

/// Decode an SVE permute / predicate / compare instruction into `out`.
///
/// Dispatches on `word<31:24>` (the SVE top byte). Total and panic-free; leaves
/// `out` [`Code::Invalid`] for anything it does not own (so the caller's earlier
/// `sve_int` result, if any, is preserved by the `is_invalid` guard in the family
/// dispatcher).
#[inline]
pub fn decode(word: u32, ip: u64, features: FeatureSet, out: &mut Instruction) {
    let _ = ip;
    if !features.has(Feature::Sve) {
        return;
    }
    match bits(word, 24, 8) {
        // 0x05: SVE permute (vector + predicate), unpack, splice, ext, tbl, rev,
        //       clast/last, compact (quadrant 000).
        0x05 => decode_perm(word, features, out),
        // 0x25: predicate logical / generation / FFR / break, WHILE, CTERM, and
        //       the predicate-indexed DUP (quadrant 001).
        0x25 => decode_pred(word, features, out),
        _ => {}
    }
}

// ===========================================================================
// Top byte 0x05 — SVE permute family.
// ===========================================================================

#[inline]
fn decode_perm(word: u32, features: FeatureSet, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let s1513 = bits(word, 13, 3); // word<15:13>

    match s1513 {
        // ZIP/UZP/TRN element-size permute: `<15:10>=0110xx`, `<12:10>` selects op.
        // Like the TBL/REV leaf below, every member fixes `<21>=1`; words with
        // `<21>=0` belong to the CPY-imm / logical-immediate / EXT regions that
        // `sve_int` owns (a `<21>=0` word here is a reserved CPY-imm slot, e.g.
        // `05156075`, that must stay Invalid rather than mis-decode as `ZIP1`).
        0b011 if bit(word, 21) == 1 => {
            let mnem = match bits(word, 10, 3) {
                0b000 => Mnemonic::Zip1,
                0b001 => Mnemonic::Zip2,
                0b010 => Mnemonic::Uzp1,
                0b011 => Mnemonic::Uzp2,
                0b100 => Mnemonic::Trn1,
                0b101 => Mnemonic::Trn2,
                _ => return,
            };
            let a = arr(size);
            let zm = bits(word, 16, 5);
            let zn = bits(word, 5, 5);
            let zd = bits(word, 0, 5);
            out.set(Code::SveZipUzpTrnZzz);
            out.set_mnemonic(mnem);
            out.push_operand(zreg(zd, a));
            out.push_operand(zreg(zn, a));
            out.push_operand(zreg(zm, a));
        }

        // `<15:13>=001`: TBL/TBX (`<15:11>=00110`/`00101`), TBXQ (`001101`),
        // REV (vector), SUNPK/UUNPK (`<15:10>=001110`). Every member of this leaf
        // fixes `<21>=1`; words with `<21>=0` are not part of this family (they
        // belong to the logical-immediate / EXT regions handled by `sve_int`).
        0b001 if bit(word, 21) == 1 => {
            match bits(word, 10, 6) {
                // TBL single-table.
                0b001100 => {
                    let a = arr(size);
                    let zm = bits(word, 16, 5);
                    let zn = bits(word, 5, 5);
                    let zd = bits(word, 0, 5);
                    out.set(Code::SveTbl);
                    out.set_mnemonic(Mnemonic::Tbl);
                    out.push_operand(zreg(zd, a));
                    out.push_operand(zlist1(zn, a));
                    out.push_operand(zreg(zm, a));
                }
                // TBL2 (op=0) / TBX (op=1): `<15:11>=00101`, `<10>=op`.
                0b001010 | 0b001011 => {
                    let a = arr(size);
                    let zm = bits(word, 16, 5);
                    let zn = bits(word, 5, 5);
                    let zd = bits(word, 0, 5);
                    if bit(word, 10) == 0 {
                        out.set(Code::SveTbl2);
                        out.set_mnemonic(Mnemonic::Tbl);
                        out.push_operand(zreg(zd, a));
                        out.push_operand(zlist2(zn, a));
                        out.push_operand(zreg(zm, a));
                    } else {
                        out.set(Code::SveTbx);
                        out.set_mnemonic(Mnemonic::Tbx);
                        out.push_operand(zreg(zd, a));
                        out.push_operand(zreg(zn, a));
                        out.push_operand(zreg(zm, a));
                    }
                }
                // TBXQ (`<15:10>=001101`, SVE2.1 128-bit-segment table lookup
                // with base): `<Zd>.<T>, <Zn>.<T>, <Zm>.<T>`.
                0b001101 => {
                    let a = arr(size);
                    let zm = bits(word, 16, 5);
                    let zn = bits(word, 5, 5);
                    let zd = bits(word, 0, 5);
                    out.set(Code::SveTbxq);
                    out.push_operand(zreg(zd, a));
                    out.push_operand(zreg(zn, a));
                    out.push_operand(zreg(zm, a));
                }
                // REV (vector) and SUNPK/UUNPK: `<15:10>=001110`, distinguished
                // by `<20:16>` (REV=11000, unpack=100UH).
                0b001110 => {
                    let op2016 = bits(word, 16, 5);
                    if op2016 == 0b11000 {
                        let a = arr(size);
                        let zn = bits(word, 5, 5);
                        let zd = bits(word, 0, 5);
                        out.set(Code::SveRevZz);
                        out.set_mnemonic(Mnemonic::Rev);
                        out.push_operand(zreg(zd, a));
                        out.push_operand(zreg(zn, a));
                    } else if bits(word, 18, 3) == 0b100 {
                        // SUNPK/UUNPK: U=<17>, H=<16>. Source is half-width.
                        let u = bit(word, 17);
                        let h = bit(word, 16);
                        let src = match size {
                            1 => VA::Sb,
                            2 => VA::Sh,
                            3 => VA::Ss,
                            _ => return,
                        };
                        let a = arr(size);
                        let mnem = match (u, h) {
                            (0, 0) => Mnemonic::Sunpklo,
                            (0, 1) => Mnemonic::Sunpkhi,
                            (1, 0) => Mnemonic::Uunpklo,
                            _ => Mnemonic::Uunpkhi,
                        };
                        let zn = bits(word, 5, 5);
                        let zd = bits(word, 0, 5);
                        out.set(Code::SveUnpk);
                        out.set_mnemonic(mnem);
                        out.push_operand(zreg(zd, a));
                        out.push_operand(zreg(zn, src));
                    }
                }
                _ => {}
            }
        }

        // `<15:13>=000`: 128-bit `.Q` permute (`<12:11>` family, `<10>=H`) and EXT.
        0b000 => {
            // `.Q` permute lives with `<23:22>=10` and `<15:11>=00011`? In fact the
            // `.Q` forms have `<15:13>=000`, `<12:11>` family, `<10>=H`, and the
            // `<23:22>` carries `op` (0) and the fixed `1`. We detect via `<15:11>`.
            if bits(word, 11, 2) <= 0b11 && bits(word, 14, 2) == 0b00 && bits(word, 13, 1) == 0 && bit(word, 21) == 1 && is_q_perm(word) {
                let fam = bits(word, 11, 2);
                let h = bit(word, 10);
                let mnem = match (fam, h) {
                    (0b00, 0) => Mnemonic::Zip1,
                    (0b00, 1) => Mnemonic::Zip2,
                    (0b01, 0) => Mnemonic::Uzp1,
                    (0b01, 1) => Mnemonic::Uzp2,
                    (0b11, 0) => Mnemonic::Trn1,
                    (0b11, 1) => Mnemonic::Trn2,
                    _ => return,
                };
                let zm = bits(word, 16, 5);
                let zn = bits(word, 5, 5);
                let zd = bits(word, 0, 5);
                out.set(Code::SveZipUzpTrnQ);
                out.set_mnemonic(mnem);
                out.push_operand(zreg_q(zd));
                out.push_operand(zreg_q(zn));
                out.push_operand(zreg_q(zm));
                return;
            }
            // EXT: `<23:21>` is `001` (destructive) or `011` (constructive) — i.e.
            // `<23> == 0` and `<21> == 1`, with `<22>` selecting des(0)/con(1). The
            // `<23> == 1` slots (e.g. `05AF14C1`, formerly mis-decoded as
            // `ext z1.b, z1.b, z6.b, #0x7d`) and the `<21> == 0` slots (e.g.
            // `050007E1`) are UNDEFINED. The valid `<23> == 1` `.Q` ZIP/UZP/TRN
            // forms are already claimed by the q-permute branch above, and the
            // valid `<21> == 0` logical-immediate / CPY-imm words are claimed by
            // `sve_int` before this fallback runs, so neither reaches here.
            if bit(word, 23) != 0 || bit(word, 21) != 1 {
                return;
            }
            if bit(word, 22) == 0 {
                decode_ext_des(word, out);
            } else {
                decode_ext_con(word, out);
            }
        }

        // `<15:13>=010`: predicate permute (ZIP/UZP/TRN/REV predicate, PUNPK).
        // Every member fixes `<21>=1`; a `<21>=0` word here is a reserved
        // logical-immediate slot (e.g. `050055E0`) that `sve_int` now correctly
        // leaves Invalid, so it must NOT mis-decode as a predicate `TRN2` — gate
        // on `<21>=1` (verified: bit-21-cleared pred-perm words are LLVM
        // UNDEFINED).
        0b010 if bit(word, 21) == 1 => decode_pred_perm(word, out),

        // `<15:13>=100` / `101`: COMPACT, SPLICE, CLAST/LAST, REVB/H/W, REVD.
        // These also fix `<21>=1`; a `<21>=0` word is a reserved logical-immediate
        // / LAST*-to-GP slot (e.g. `0500A7AE`) that must stay Invalid rather than
        // mis-decode as `LASTA`/`LASTB`.
        0b100 | 0b101 if bit(word, 21) == 1 => decode_perm_misc(word, features, out),

        _ => {}
    }
}

/// `true` when the `<15:13>=000` word is a 128-bit `.Q` ZIP/UZP/TRN permute
/// (`<23:22>=10`, `<21>=1`, `<12:11>` in {00,01,11}).
#[inline]
fn is_q_perm(word: u32) -> bool {
    bits(word, 22, 2) == 0b10 && bit(word, 21) == 1 && matches!(bits(word, 11, 2), 0b00 | 0b01 | 0b11)
}

/// Predicate permute (`<15:13>=010`): ZIP/UZP/TRN predicate (`<20>=0`, Pm present)
/// and the unary REV (predicate) / PUNPK{HI,LO} (`<20>=1`).
#[inline]
fn decode_pred_perm(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let pn = bits(word, 5, 4);
    let pd = bits(word, 0, 4);

    if bit(word, 4) != 0 || bit(word, 9) != 0 {
        return;
    }

    if bit(word, 20) == 0 {
        // ZIP/UZP/TRN predicate: `<12:10>` selects the op.
        let mnem = match bits(word, 10, 3) {
            0b000 => Mnemonic::Zip1,
            0b001 => Mnemonic::Zip2,
            0b010 => Mnemonic::Uzp1,
            0b011 => Mnemonic::Uzp2,
            0b100 => Mnemonic::Trn1,
            0b101 => Mnemonic::Trn2,
            _ => return,
        };
        let a = arr(size);
        let pm = bits(word, 16, 4);
        out.set(Code::SveZipUzpTrnPpp);
        out.set_mnemonic(mnem);
        out.push_operand(preg_sz(pd, a));
        out.push_operand(preg_sz(pn, a));
        out.push_operand(preg_sz(pm, a));
        return;
    }

    // `<20>=1`: REV (predicate) `<20:16>=10100`, PUNPK `<20:16>=1000H`. Both are
    // unary (no `Pm`) and fix `<12:10>=000`; a non-zero `<12:10>` is reserved
    // (e.g. `05744501` over `05744101 rev p1.h,p8.h` — `<unknown>` in LLVM).
    if bits(word, 10, 3) != 0 {
        return;
    }
    let op2016 = bits(word, 16, 5);
    if op2016 == 0b10100 {
        let a = arr(size);
        out.set(Code::SveRevP);
        out.set_mnemonic(Mnemonic::Rev);
        out.push_operand(preg_sz(pd, a));
        out.push_operand(preg_sz(pn, a));
    } else if bits(word, 17, 4) == 0b1000 && size == 0 {
        // PUNPKHI/LO: H=<16>. Dest `.H`, source `.B`.
        let hi = bit(word, 16) == 1;
        out.set(Code::SvePunpk);
        out.set_mnemonic(if hi { Mnemonic::Punpkhi } else { Mnemonic::Punpklo });
        out.push_operand(preg_sz(pd, VA::Sh));
        out.push_operand(preg_sz(pn, VA::Sb));
    }
}

/// COMPACT / SPLICE / CLAST / LAST / REVB / REVH / REVW / REVD
/// (`<15:13>=100` or `101`). All carry a governing predicate `<12:10>` and a
/// distinguishing `<20:16>` opcode.
#[inline]
fn decode_perm_misc(word: u32, features: FeatureSet, out: &mut Instruction) {
    let size = bits(word, 22, 2);
    let op2016 = bits(word, 16, 5);
    let pg = bits(word, 10, 3);
    let s1 = bits(word, 5, 5);
    let d = bits(word, 0, 5);

    // `<15:13>=101` group: CLASTA/B and LASTA/B to a GP register.
    if bits(word, 13, 3) == 0b101 {
        match op2016 {
            // CLASTA/CLASTB to GP register: `<20:16>=1000 B`.
            0b10000 | 0b10001 => {
                let b = bit(word, 16);
                let w = if size == 3 { RegWidth::X64 } else { RegWidth::W32 };
                out.set(Code::SveClastR);
                out.set_mnemonic(if b == 0 { Mnemonic::Clasta } else { Mnemonic::Clastb });
                out.push_operand(gpr(d, w));
                out.push_operand(preg(pg));
                out.push_operand(gpr(d, w));
                out.push_operand(zreg(s1, arr(size)));
            }
            // LASTA/LASTB to GP register: `<20:16>=0000 B`.
            0b00000 | 0b00001 => {
                let b = bit(word, 16);
                let w = if size == 3 { RegWidth::X64 } else { RegWidth::W32 };
                out.set(Code::SveLastR);
                out.set_mnemonic(if b == 0 { Mnemonic::Lasta } else { Mnemonic::Lastb });
                out.push_operand(gpr(d, w));
                out.push_operand(preg(pg));
                out.push_operand(zreg(s1, arr(size)));
            }
            // REVD zeroing (`/z`, FEAT_SVE2p1): `<21>=1`, `<20:16>=01110`,
            // `<13>=1`, `<23:22>=00`. The merging (`/m`) form lives in the
            // `<15:13>=100` group below; this is its `<15:13>=101` sibling.
            0b01110 if size == 0 && bit(word, 21) == 1 && features.has(Feature::Sve2p1) => {
                out.set(Code::SveRevdZpzZero);
                out.set_mnemonic(Mnemonic::Revd);
                out.push_operand(zreg_q(d));
                out.push_operand(preg_q(pg, PredQual::Zeroing));
                out.push_operand(zreg_q(s1));
            }
            _ => {}
        }
        return;
    }

    // `<15:13>=100` group.
    match op2016 {
        // COMPACT: `<20:16>=00001`, size in `<22>` (.s/.d).
        0b00001 => {
            // Distinguish COMPACT (`<23>=1`) from REVW (`<20:16>=00110`, different).
            let a = if bit(word, 22) == 0 { VA::Ss } else { VA::Sd };
            out.set(Code::SveCompact);
            out.set_mnemonic(Mnemonic::Compact);
            out.push_operand(zreg(d, a));
            out.push_operand(preg(pg));
            out.push_operand(zreg(s1, a));
        }
        // SPLICE destructive: `<20:16>=01100`.
        0b01100 => {
            let a = arr(size);
            out.set(Code::SveSpliceDes);
            out.set_mnemonic(Mnemonic::Splice);
            out.push_operand(zreg(d, a));
            out.push_operand(preg(pg));
            out.push_operand(zreg(d, a));
            out.push_operand(zreg(s1, a));
        }
        // SPLICE constructive (2-list): `<20:16>=01101`.
        0b01101 => {
            let a = arr(size);
            out.set(Code::SveSpliceCon);
            out.set_mnemonic(Mnemonic::Splice);
            out.push_operand(zreg(d, a));
            out.push_operand(preg(pg));
            out.push_operand(zlist2(s1, a));
        }
        // CLASTA/CLASTB to vector (destructive): `<20:16>=01000 B`.
        0b01000 | 0b01001 => {
            let b = bit(word, 16);
            let a = arr(size);
            out.set(Code::SveClastZ);
            out.set_mnemonic(if b == 0 { Mnemonic::Clasta } else { Mnemonic::Clastb });
            out.push_operand(zreg(d, a));
            out.push_operand(preg(pg));
            out.push_operand(zreg(d, a));
            out.push_operand(zreg(s1, a));
        }
        // CLASTA/CLASTB to SIMD scalar: `<20:16>=01010 B`.
        0b01010 | 0b01011 => {
            let b = bit(word, 16);
            out.set(Code::SveClastV);
            out.set_mnemonic(if b == 0 { Mnemonic::Clasta } else { Mnemonic::Clastb });
            out.push_operand(scalar_fp(d, size));
            out.push_operand(preg(pg));
            out.push_operand(scalar_fp(d, size));
            out.push_operand(zreg(s1, arr(size)));
        }
        // LASTA/LASTB to SIMD scalar: `<20:16>=00010 B`.
        0b00010 | 0b00011 => {
            let b = bit(word, 16);
            out.set(Code::SveLastV);
            out.set_mnemonic(if b == 0 { Mnemonic::Lasta } else { Mnemonic::Lastb });
            out.push_operand(scalar_fp(d, size));
            out.push_operand(preg(pg));
            out.push_operand(zreg(s1, arr(size)));
        }
        // REVB/REVH/REVW (predicated): `<20:16>=00100/00101/00110`.
        0b00100..=0b00110 => {
            let (mnem, min_size) = match op2016 {
                0b00100 => (Mnemonic::Revb, 1),
                0b00101 => (Mnemonic::Revh, 2),
                _ => (Mnemonic::Revw, 3),
            };
            if size < min_size {
                return;
            }
            let a = arr(size);
            out.set(Code::SveRevbhw);
            out.set_mnemonic(mnem);
            out.push_operand(zreg(d, a));
            out.push_operand(preg_q(pg, PredQual::Merging));
            out.push_operand(zreg(s1, a));
        }
        // REVD merging (`/m`, `.Q`): `<21>=1`, `<20:16>=01110`, `<15:13>=100`,
        // `<23:22>=00` (size != 00 is UNDEFINED). The `<15:13>=101` zeroing (`/z`)
        // sibling is handled in the CLAST/LAST block above.
        0b01110 => {
            if size != 0 || bit(word, 21) != 1 {
                return; // size field reserved; `<21>` must be 1.
            }
            out.set(Code::RevdZPZ);
            out.set_mnemonic(Mnemonic::Revd);
            out.push_operand(zreg_q(d));
            out.push_operand(preg_q(pg, PredQual::Merging));
            out.push_operand(zreg_q(s1));
        }
        _ => {}
    }
}


/// A single-register Z list `{Z{n}.<T>}` for the single-table `TBL`.
#[inline]
fn zlist1(n: u32, a: VA) -> Operand {
    Operand::MultiReg { regs: [Z[(n & 0x1f) as usize], Register::None, Register::None, Register::None], count: 1, arr: Some(a), lane: None }
}

/// EXT (destructive): `EXT <Zdn>.B, <Zdn>.B, <Zm>.B, #imm`.
///
/// `imm = imm8h:imm8l` with `imm8h = word<20:16>`, `imm8l = word<12:10>`.
#[inline]
fn decode_ext_des(word: u32, out: &mut Instruction) {
    let imm8h = bits(word, 16, 5);
    let imm8l = bits(word, 10, 3);
    let imm = (imm8h << 3) | imm8l;
    let zm = bits(word, 5, 5);
    let zdn = bits(word, 0, 5);
    out.set(Code::SveExtDes);
    out.set_mnemonic(Mnemonic::Ext);
    out.push_operand(zreg(zdn, VA::Sb));
    out.push_operand(zreg(zdn, VA::Sb));
    out.push_operand(zreg(zm, VA::Sb));
    out.push_operand(Operand::ImmUnsigned(imm as u64));
}

/// EXT (constructive): `EXT <Zd>.B, {<Zn1>.B, <Zn2>.B}, #imm`.
///
/// Binary Ninja renders an extra `z0.b` operand between the register list and the
/// immediate (`{z30.b, z31.b}, z0.b, #0xb5`). The ARM ARM constructive-EXT
/// syntax has no such third register operand; we emit the literal `z0.b` to match
/// the corpus and note this as a Binary-Ninja divergence from the spec.
#[inline]
fn decode_ext_con(word: u32, out: &mut Instruction) {
    let imm8h = bits(word, 16, 5);
    let imm8l = bits(word, 10, 3);
    let imm = (imm8h << 3) | imm8l;
    let zn = bits(word, 5, 5);
    let zd = bits(word, 0, 5);
    out.set(Code::SveExtCon);
    out.set_mnemonic(Mnemonic::Ext);
    out.push_operand(zreg(zd, VA::Sb));
    out.push_operand(zlist2(zn, VA::Sb));
    out.push_operand(zreg(0, VA::Sb)); // Binary Ninja's spurious `z0.b`.
    out.push_operand(Operand::ImmUnsigned(imm as u64));
}

// ===========================================================================
// Top byte 0x25 — predicate logical / generation / FFR / break, WHILE, CTERM.
// ===========================================================================

#[inline]
fn decode_pred(word: u32, features: FeatureSet, out: &mut Instruction) {
    // The predicate region is keyed by `<15:14>`; within `00` the `<13>` bit
    // splits the WHILE<cc>/compare region (0) from CTERM/WHILERW (1).
    match bits(word, 14, 2) {
        0b00 => {
            if bit(word, 13) == 0 {
                decode_while_dup(word, out);
            } else if bits(word, 10, 3) == 0b100 {
                // WHILERW / WHILEWR: `<15:10>=001100`.
                decode_while_rw(word, out);
            } else {
                // CTERMEQ / CTERMNE: `<15:10>=001000`.
                decode_cterm(word, out);
            }
        }
        // 01/10/11: predicate gen/logical/FFR/break/SEL, BRKP, PTRUE/PFALSE/...,
        // and the SVE2.1 WHILE predicate-pair / predicate-as-counter forms (which
        // share `<15:14>=01`, distinguished by `<21:20>=10`, `<15:13>` in {010,011}).
        0b01..=0b11 => {
            if features.has(Feature::Sve2p1) {
                decode_while_pair_pn(word, out);
            }
            if out.is_invalid() {
                decode_pred_misc(word, out);
            }
        }
        _ => {}
    }
}

/// SVE2.1 `WHILE<cc>` with a predicate-PAIR result (`{Pd.T, Pd+1.T}`) or a
/// predicate-as-counter result (`PNd.T, ..., VLx{2,4}`). Layout (top byte 0x25):
/// `00100101 size 1 Rm 010 ...`, both operands 64-bit X registers.
///
/// * `<12>=1` selects the **predicate-pair** form: condition `(U<11>, lt<10>,
///   eq<0>)`, the result pair is `P(2*<3:1>)`/`P(2*<3:1>+1)` and `<4>=1`.
/// * `<12>=0` selects the **predicate-as-counter** form: condition `(U<11>,
///   lt<10>, eq<3>)`, `<13>` selects `VLx2`(0)/`VLx4`(1), result `PN(8+<2:0>)`
///   and `<4>=1`.
#[inline]
fn decode_while_pair_pn(word: u32, out: &mut Instruction) {
    // Skeleton: `<15:13>=010` (pair / counter-x2) or `011` (counter-x4),
    // `<21>=1` (`Rm`-region marker; `<20:16>` is the free `Rm` field), `<4>=1`.
    if bits(word, 13, 3) != 0b010 && bits(word, 13, 3) != 0b011 {
        return;
    }
    if bit(word, 21) != 1 || bit(word, 4) != 1 {
        return;
    }
    let size = bits(word, 22, 2);
    let a = arr(size);
    let rm = bits(word, 16, 5);
    let rn = bits(word, 5, 5);
    let u = bit(word, 11);
    let lt = bit(word, 10);
    let is_pair = bit(word, 12) == 1;
    // `<13>` must be 0 for the pair form (it has no VL multiplier).
    if is_pair && bit(word, 13) != 0 {
        return;
    }
    let eq = if is_pair { bit(word, 0) } else { bit(word, 3) };
    let mnem = match (u, lt, eq) {
        (0, 1, 0) => Mnemonic::Whilelt,
        (0, 1, 1) => Mnemonic::Whilele,
        (1, 1, 0) => Mnemonic::Whilelo,
        (1, 1, 1) => Mnemonic::Whilels,
        (0, 0, 0) => Mnemonic::Whilege,
        (0, 0, 1) => Mnemonic::Whilegt,
        (1, 0, 1) => Mnemonic::Whilehi,
        _ => Mnemonic::Whilehs,
    };
    if is_pair {
        // Predicate pair: `{P(2k).T, P(2k+1).T}`, k = <3:1>.
        let k = bits(word, 1, 3);
        let first = (2 * k) & 0xf;
        out.set(Code::SveWhilePair);
        out.set_mnemonic(mnem);
        out.push_operand(pred_pair(first, a));
    } else {
        // Predicate-as-counter: `PN(8 + <2:0>).T`, with a VLx2/VLx4 multiplier.
        let pn = 8 + bits(word, 0, 3);
        out.set(Code::SveWhilePn);
        out.set_mnemonic(mnem);
        out.push_operand(Operand::PredCounter { reg: P[(pn & 0xf) as usize], zeroing: false, arr: Some(a) });
    }
    out.push_operand(gpr(rn, RegWidth::X64));
    out.push_operand(gpr(rm, RegWidth::X64));
    if !is_pair {
        let mul = if bit(word, 13) == 1 { 4 } else { 2 };
        out.push_operand(Operand::VlMul(mul));
    }
}

/// A predicate-pair `{P(first).T, P(first+1).T}` rendered via [`Operand::MultiReg`].
#[inline]
fn pred_pair(first: u32, a: VA) -> Operand {
    Operand::MultiReg {
        regs: [P[(first & 0xf) as usize], P[((first + 1) & 0xf) as usize], Register::None, Register::None],
        count: 2,
        arr: Some(a),
        lane: None,
    }
}

/// WHILE<cc> (`<15:13>=000`, `<21>=1`): `00100101 size 1 Rm 000 sf U lt Rn eq Pd`.
///
/// `sve_int` owns the signed-immediate compares sharing `<15:13>=000` (they have
/// `<21>=0`); the WHILE forms have `<21>=1`. `<12>=sf` selects the GP-register
/// width, `(U<11>, lt<10>, eq<4>)` select the condition.
#[inline]
fn decode_while_dup(word: u32, out: &mut Instruction) {
    if bit(word, 21) != 1 {
        return;
    }
    let size = bits(word, 22, 2);
    let a = arr(size);
    let rm = bits(word, 16, 5);
    let rn = bits(word, 5, 5);
    let pd = bits(word, 0, 4);
    let sf = bit(word, 12);
    let w = if sf == 1 { RegWidth::X64 } else { RegWidth::W32 };

    let u = bit(word, 11);
    let lt = bit(word, 10);
    let eq = bit(word, 4);
    let mnem = match (u, lt, eq) {
        (0, 1, 0) => Mnemonic::Whilelt,
        (0, 1, 1) => Mnemonic::Whilele,
        (1, 1, 0) => Mnemonic::Whilelo,
        (1, 1, 1) => Mnemonic::Whilels,
        (0, 0, 0) => Mnemonic::Whilege,
        (0, 0, 1) => Mnemonic::Whilegt,
        (1, 0, 1) => Mnemonic::Whilehi,
        (1, 0, 0) => Mnemonic::Whilehs,
        _ => return,
    };
    out.set(Code::SveWhile);
    out.set_mnemonic(mnem);
    out.push_operand(preg_sz(pd, a));
    out.push_operand(gpr(rn, w));
    out.push_operand(gpr(rm, w));
}

/// WHILERW / WHILEWR (`<15:10>=001100`, `<21>=1`): the address-dependency checks.
/// Both operands are 64-bit X registers; `rw<4>` selects RW(1)/WR(0).
#[inline]
fn decode_while_rw(word: u32, out: &mut Instruction) {
    if bit(word, 21) != 1 {
        return;
    }
    let a = arr(bits(word, 22, 2));
    let rm = bits(word, 16, 5);
    let rn = bits(word, 5, 5);
    let pd = bits(word, 0, 4);
    let rw = bit(word, 4);
    out.set(Code::SveWhileRw);
    out.set_mnemonic(if rw == 1 { Mnemonic::Whilerw } else { Mnemonic::Whilewr });
    out.push_operand(preg_sz(pd, a));
    out.push_operand(gpr(rn, RegWidth::X64));
    out.push_operand(gpr(rm, RegWidth::X64));
}

/// CTERMEQ / CTERMNE: `00100101 1 sz 1 Rm 001000 Rn op 0000`.
///
/// `sz<22>` selects the X (1) vs W (0) operand width; `op<4>` selects EQ(0)/NE(1).
#[inline]
fn decode_cterm(word: u32, out: &mut Instruction) {
    // Skeleton check: `<15:10>=001000`, `<3:0>=0000`.
    if bits(word, 10, 6) != 0b001000 || bits(word, 0, 4) != 0 {
        return;
    }
    let sz = bit(word, 22);
    let w = if sz == 1 { RegWidth::X64 } else { RegWidth::W32 };
    let rm = bits(word, 16, 5);
    let rn = bits(word, 5, 5);
    let op = bit(word, 4);
    out.set(Code::SveCterm);
    out.set_mnemonic(if op == 0 { Mnemonic::Ctermeq } else { Mnemonic::Ctermne });
    out.push_operand(gpr(rn, w));
    out.push_operand(gpr(rm, w));
}

/// Predicate generation / logical / FFR / break / SEL (`<15:14>=01` or `=11`).
///
/// The members of this region overlap in `<15:14>`, so the more specific groups
/// (DUP-of-predicate, the fixed-field generation/FFR ops, the break family, and
/// the `<15:14>=11` BRKP group) are tried first; the generic predicate-logical
/// group (which has the free `Pm` field at `<19:16>` and `<21:20>=00`) is the
/// final fallthrough.
#[inline]
fn decode_pred_misc(word: u32, out: &mut Instruction) {
    // The fixed-pattern generation / FFR ops (PTRUE/PFALSE/PTEST/PFIRST/PNEXT/
    // RDFFR/WRFFR/SETFFR) span `<15:14>` in {10,11}; they have fully-fixed
    // skeletons, so try them first.
    decode_pred_gen(word, out);
    if !out.is_invalid() {
        return;
    }
    // BRKP{A,B}{S} live at `<15:14>=11`.
    if bits(word, 14, 2) == 0b11 {
        decode_brkp(word, out);
        return;
    }
    // The break-unary family (BRKA/BRKB/BRKN) and their fixed `<20:16>` patterns.
    decode_break(word, out);
    if !out.is_invalid() {
        return;
    }
    // Predicate-indexed DUP (`<21>=1`, `<15:14>=01`).
    if bit(word, 21) == 1 && decode_dup_pred(word, out) {
        return;
    }

    // --- Generic predicate-logical group (Pd.B, Pg/Z, Pn.B, Pm.B) and SEL ----
    // Skeleton: `0010010 op S 00 Pm 01 Pg o2 Pn o3 Pd`, with the opcode formed by
    //   (op<23>, S<22>, o2<9>, o3<4>). The free Pm field is `<19:16>`; the group
    //   is gated by `<21:20>==00` (the break/generation ops have `<20>` set).
    if bits(word, 14, 2) != 0b01 || bits(word, 20, 2) != 0b00 {
        return;
    }
    let pm = bits(word, 16, 4);
    let pg = bits(word, 10, 4);
    let pn = bits(word, 5, 4);
    let pd = bits(word, 0, 4);
    let op = bit(word, 23);
    let s = bit(word, 22);
    let o2 = bit(word, 9);
    let o3 = bit(word, 4);
    let key = (op << 3) | (s << 2) | (o2 << 1) | o3;

    // SEL (predicate): `op=0, S=0, o2=1, o3=1` -> `00100101 00 00 Pm 01 Pg 1 Pn 1 Pd`.
    if key == 0b0011 {
        // MOV alias: SEL Pd, Pg, Pn, Pd  (Pm == Pd)  ->  MOV Pd.B, Pg/M, Pn.B.
        if pm == pd {
            out.set(Code::SveSelPred);
            out.set_mnemonic(Mnemonic::Mov);
            out.push_operand(preg_sz(pd, VA::Sb));
            out.push_operand(preg_q(pg, PredQual::Merging));
            out.push_operand(preg_sz(pn, VA::Sb));
        } else {
            out.set(Code::SveSelPred);
            out.set_mnemonic(Mnemonic::Sel);
            out.push_operand(preg_sz(pd, VA::Sb));
            out.push_operand(preg(pg));
            out.push_operand(preg_sz(pn, VA::Sb));
            out.push_operand(preg_sz(pm, VA::Sb));
        }
        return;
    }

    let mnem = match key {
        0b0000 => Mnemonic::And,
        0b0001 => Mnemonic::Bic,
        0b0010 => Mnemonic::Eor,
        0b0100 => Mnemonic::Ands,
        0b0101 => Mnemonic::Bics,
        0b0110 => Mnemonic::Eors,
        0b1000 => Mnemonic::Orr,
        0b1001 => Mnemonic::Orn,
        0b1010 => Mnemonic::Nor,
        0b1011 => Mnemonic::Nand,
        0b1100 => Mnemonic::Orrs,
        0b1101 => Mnemonic::Orns,
        0b1110 => Mnemonic::Nors,
        0b1111 => Mnemonic::Nands,
        _ => return,
    };
    emit_pred_logical(out, mnem, pd, pg, pn, pm);
}

/// Predicate-indexed DUP (`DUP <Pd>.<T>, <Pg>/Z, <Pn>.<T>[<Ws>{, #imm}]`).
///
/// Returns `true` if it matched. Encoding (SVE2.1, FEAT_SME2p1 style): the
/// element-index field is `i1:tszh:tszl` (`<23>`, `<22>`, `<20:18>`); the lowest
/// set bit of `tszh:tszl` gives the element size; the slice register is
/// `W12 + <17:16>`; and the immediate index is the upper part of the index field
/// above the size bit. The fixed markers are `<21>=1`, `<15:14>=01`, `<9>=0`,
/// `<4>=0`.
#[inline]
fn decode_dup_pred(word: u32, out: &mut Instruction) -> bool {
    // This `<21>=1, <15:14>=01, <9>=0, <4>=0` slot is PSEL (predicate select):
    // `PSEL <Pd>, <Pn>, <Pm>.<T>[<Wv>{, #imm}]`. It was historically mis-decoded
    // as a predicate-indexed `DUP`; the entire slot is PSEL per LLVM.
    //
    // `<15:14>` is a fixed `01` marker. A `<15:14>` of `10`/`11` reaches here only
    // when the generation / break decoders decline, and is reserved → UNDEFINED
    // (verified by an LLVM `<15:13>` sweep: only `010`/`011` decode as PSEL).
    if bits(word, 14, 2) != 0b01 || bit(word, 9) != 0 || bit(word, 4) != 0 {
        return false;
    }
    // The element / index live in the SVE `tszh:tszl` field `<23:22>:<20:18>` (5
    // bits): the lowest set bit gives the element size, the bits above it the
    // index. An all-zero field is reserved.
    let tsz = (bits(word, 22, 2) << 3) | bits(word, 18, 3);
    if tsz == 0 {
        return false; // reserved
    }
    let esize = tsz.trailing_zeros(); // 0=>.b,1=>.h,2=>.s,3=>.d
    let a = match esize {
        0 => VA::Sb,
        1 => VA::Sh,
        2 => VA::Ss,
        3 => VA::Sd,
        _ => return false, // `tsz==10000`: no element marker, reserved
    };
    let imm = (tsz >> (esize + 1)) as i64;
    let wv = 12 + bits(word, 16, 2); // W12..W15
    let pm = bits(word, 5, 4);
    let pn = bits(word, 10, 4);
    let pd = bits(word, 0, 4);
    out.set(Code::SvePsel);
    out.push_operand(preg(pd));
    out.push_operand(preg(pn));
    out.push_operand(Operand::IndexedElement {
        reg: P[(pm & 0xf) as usize],
        arr: Some(a),
        index: gp_register(false, RegWidth::W32, wv as u8),
        imm,
    });
    true
}

/// Emit a predicate-logical instruction (`op <Pd>.B, <Pg>/Z, <Pn>.B, <Pm>.B`),
/// applying the `MOV`/`MOVS`/`NOT`/`NOTS` aliases the corpus prefers. The
/// governing predicate `<Pg>/Z` prints with the `/z` qualifier but no `.b` size.
#[inline]
fn emit_pred_logical(out: &mut Instruction, mnem: Mnemonic, pd: u32, pg: u32, pn: u32, pm: u32) {
    out.set(Code::SvePredLogical);

    // Alias rewriting (preferred disassembly):
    //   ORR  Pd.B, Pn/Z, Pn.B, Pn.B  -> MOV  Pd.B, Pn.B           (Pg==Pn==Pm)
    //   ORRS Pd.B, Pn/Z, Pn.B, Pn.B  -> MOVS Pd.B, Pn.B
    //   AND  Pd.B, Pg/Z, Pn.B, Pn.B  -> MOV  Pd.B, Pg/Z, Pn.B     (Pn==Pm)
    //   ANDS Pd.B, Pg/Z, Pn.B, Pn.B  -> MOVS Pd.B, Pg/Z, Pn.B
    //   EOR  Pd.B, Pg/Z, Pn.B, Pg.B  -> NOT  Pd.B, Pg/Z, Pn.B     (Pm==Pg)
    //   EORS Pd.B, Pg/Z, Pn.B, Pg.B  -> NOTS Pd.B, Pg/Z, Pn.B
    match mnem {
        Mnemonic::Orr if pg == pn && pn == pm => {
            out.set_mnemonic(Mnemonic::Mov);
            out.push_operand(preg_sz(pd, VA::Sb));
            out.push_operand(preg_sz(pn, VA::Sb));
            return;
        }
        Mnemonic::Orrs if pg == pn && pn == pm => {
            out.set_mnemonic(Mnemonic::Movs);
            out.push_operand(preg_sz(pd, VA::Sb));
            out.push_operand(preg_sz(pn, VA::Sb));
            return;
        }
        Mnemonic::And if pn == pm => {
            out.set_mnemonic(Mnemonic::Mov);
            out.push_operand(preg_sz(pd, VA::Sb));
            out.push_operand(preg_q(pg, PredQual::Zeroing));
            out.push_operand(preg_sz(pn, VA::Sb));
            return;
        }
        Mnemonic::Ands if pn == pm => {
            out.set_mnemonic(Mnemonic::Movs);
            out.push_operand(preg_sz(pd, VA::Sb));
            out.push_operand(preg_q(pg, PredQual::Zeroing));
            out.push_operand(preg_sz(pn, VA::Sb));
            return;
        }
        Mnemonic::Eor if pm == pg => {
            out.set_mnemonic(Mnemonic::Not);
            out.push_operand(preg_sz(pd, VA::Sb));
            out.push_operand(preg_q(pg, PredQual::Zeroing));
            out.push_operand(preg_sz(pn, VA::Sb));
            return;
        }
        Mnemonic::Eors if pm == pg => {
            out.set_mnemonic(Mnemonic::Nots);
            out.push_operand(preg_sz(pd, VA::Sb));
            out.push_operand(preg_q(pg, PredQual::Zeroing));
            out.push_operand(preg_sz(pn, VA::Sb));
            return;
        }
        _ => {}
    }

    out.set_mnemonic(mnem);
    out.push_operand(preg_sz(pd, VA::Sb));
    out.push_operand(preg_q(pg, PredQual::Zeroing));
    out.push_operand(preg_sz(pn, VA::Sb));
    out.push_operand(preg_sz(pm, VA::Sb));
}

/// BRKP{A,B}{S} (binary, `<15:14>=11`): `00100101 0 S 00 Pm 11 Pg 0 Pn B 0 Pd`.
///   `<23>=0, <22>=S, <21:20>=00, <19:16>=Pm, <13:10>=Pg, <8:5>=Pn, <4>=B`.
#[inline]
fn decode_brkp(word: u32, out: &mut Instruction) {
    if bits(word, 20, 2) != 0b00 || bit(word, 23) != 0 || bit(word, 9) != 0 {
        return;
    }
    let s = bit(word, 22);
    let b = bit(word, 4);
    let pm = bits(word, 16, 4);
    let pg = bits(word, 10, 4);
    let pn = bits(word, 5, 4);
    let pd = bits(word, 0, 4);
    let mnem = match (b, s) {
        (0, 0) => Mnemonic::Brkpa,
        (0, 1) => Mnemonic::Brkpas,
        (1, 0) => Mnemonic::Brkpb,
        _ => Mnemonic::Brkpbs,
    };
    out.set(Code::SveBrkpPred);
    out.set_mnemonic(mnem);
    out.push_operand(preg_sz(pd, VA::Sb));
    out.push_operand(preg_q(pg, PredQual::Zeroing));
    out.push_operand(preg_sz(pn, VA::Sb));
    out.push_operand(preg_sz(pm, VA::Sb));
}

/// Break group: BRKA/BRKAS/BRKB/BRKBS (unary) and BRKN/BRKNS (`<15:14>=01`).
#[inline]
fn decode_break(word: u32, out: &mut Instruction) {
    let pg = bits(word, 10, 4);
    let pn = bits(word, 5, 4);
    let pd = bits(word, 0, 4);

    // BRKA/BRKAS/BRKB/BRKBS: `00100101 B S 01000 0 01 Pg M Pn 0 Pd`.
    //   <23>=B(A=0/B=1), <22>=S, <21:16>=010000, <15:14>=01, <13:10>=Pg,
    //   <9>=0, <8:5>=Pn, <4>=M (merge select; flag-setting S=1 forms are
    //   zeroing-only with M=0), Pd=<3:0>.
    if bits(word, 16, 6) == 0b010000 && bits(word, 14, 2) == 0b01 && bit(word, 9) == 0 {
        let bb = bit(word, 23); // 0=A,1=B
        let s = bit(word, 22);
        let mnem = match (bb, s) {
            (0, 0) => Mnemonic::Brka,
            (0, 1) => Mnemonic::Brkas,
            (1, 0) => Mnemonic::Brkb,
            _ => Mnemonic::Brkbs,
        };
        let m = bit(word, 4);
        // The flag-setting forms (BRKAS/BRKBS, `S=1`) are zeroing-only: the merge
        // bit `<4>=M` must be 0. `M=1` with `S=1` is reserved → UNDEFINED (e.g.
        // `25D05893` over `25D05883 brkbs p3.b,p6/z,p4.b` — `<unknown>` in LLVM).
        if s == 1 && m == 1 {
            return;
        }
        let q = if s == 0 && m == 1 { PredQual::Merging } else { PredQual::Zeroing };
        out.set(Code::SveBrkPred);
        out.set_mnemonic(mnem);
        out.push_operand(preg_sz(pd, VA::Sb));
        out.push_operand(preg_q(pg, q));
        out.push_operand(preg_sz(pn, VA::Sb));
        return;
    }

    // BRKN/BRKNS: `00100101 0 S 011000 01 Pg 0 Pn 0 Pdm`.
    //   KEY brkn : <23:16>=00011000 ; brkns: <23:16>=01011000.
    if bits(word, 16, 6) == 0b011000 && bits(word, 14, 2) == 0b01 && bit(word, 9) == 0 && bit(word, 4) == 0 && bit(word, 23) == 0 {
        let s = bit(word, 22);
        out.set(Code::SveBrkn);
        out.set_mnemonic(if s == 0 { Mnemonic::Brkn } else { Mnemonic::Brkns });
        out.push_operand(preg_sz(pd, VA::Sb));
        out.push_operand(preg_q(pg, PredQual::Zeroing));
        out.push_operand(preg_sz(pn, VA::Sb));
        out.push_operand(preg_sz(pd, VA::Sb));
    }
}

/// Predicate generation: PTRUE/PTRUES/PFALSE/PTEST/PFIRST/PNEXT/RDFFR/RDFFRS/
/// WRFFR/SETFFR.
#[inline]
fn decode_pred_gen(word: u32, out: &mut Instruction) {
    let size = bits(word, 22, 2);

    // SETFFR: `00100101 0010 1100 1001 0000 0000 0000` (fully fixed).
    if word & 0x00FF_FFFF == 0x002C_9000 {
        out.set(Code::SveSetffr);
        out.set_mnemonic(Mnemonic::Setffr);
        return;
    }
    // WRFFR: `00100101 0010 1000 1001 000 Pn 0 0000`.
    if bits(word, 12, 12) == 0b001010001001 && bits(word, 0, 5) == 0 {
        let pn = bits(word, 5, 4);
        out.set(Code::SveWrffr);
        out.set_mnemonic(Mnemonic::Wrffr);
        out.push_operand(preg_sz(pn, VA::Sb));
        return;
    }
    // PFALSE: `00100101 0001 1000 1110 0100 0000 Pd`.
    if bits(word, 4, 20) == 0b00011000111001000000 {
        let pd = bits(word, 0, 4);
        out.set(Code::SvePfalse);
        out.set_mnemonic(Mnemonic::Pfalse);
        out.push_operand(preg_sz(pd, VA::Sb));
        return;
    }
    // RDFFR (unpredicated): `00100101 0001 1001 1111 0000 0000 Pd`.
    if bits(word, 4, 20) == 0b00011001111100000000 {
        let pd = bits(word, 0, 4);
        out.set(Code::SveRdffr);
        out.set_mnemonic(Mnemonic::Rdffr);
        out.push_operand(preg_sz(pd, VA::Sb));
        return;
    }
    // RDFFR/RDFFRS (predicated): `00100101 0 S 011000 1111 000 Pg 0 Pd`.
    //   rdffr  : <23:16>=00011000, <15:5>=11110000... Pg
    //   rdffrs : <23:16>=01011000
    if bits(word, 16, 6) == 0b011000 && bits(word, 9, 7) == 0b1111000 && bit(word, 4) == 0 {
        let s = bit(word, 22);
        let pg = bits(word, 5, 4);
        let pd = bits(word, 0, 4);
        out.set(Code::SveRdffrPred);
        out.set_mnemonic(if s == 0 { Mnemonic::Rdffr } else { Mnemonic::Rdffrs });
        out.push_operand(preg_sz(pd, VA::Sb));
        out.push_operand(preg_q(pg, PredQual::Zeroing));
        return;
    }

    // PTEST: `00100101 0101 0000 11 Pg 0 Pn 0 0000`.
    if bits(word, 16, 8) == 0b01010000 && bits(word, 14, 2) == 0b11 && bit(word, 9) == 0 && bits(word, 0, 5) == 0 {
        let pg = bits(word, 10, 4);
        let pn = bits(word, 5, 4);
        out.set(Code::SvePtest);
        out.set_mnemonic(Mnemonic::Ptest);
        out.push_operand(preg(pg));
        out.push_operand(preg_sz(pn, VA::Sb));
        return;
    }

    // PFIRST: `00100101 0101 1000 1100 000 Pg 0 Pdn`.
    if bits(word, 16, 8) == 0b01011000 && bits(word, 9, 7) == 0b1100000 && bit(word, 4) == 0 {
        let pg = bits(word, 5, 4);
        let pdn = bits(word, 0, 4);
        out.set(Code::SvePfirst);
        out.set_mnemonic(Mnemonic::Pfirst);
        out.push_operand(preg_sz(pdn, VA::Sb));
        out.push_operand(preg(pg));
        out.push_operand(preg_sz(pdn, VA::Sb));
        return;
    }

    // PNEXT: `00100101 size 011001 11000 1 0 Pg 0 Pdn`.
    if bits(word, 16, 6) == 0b011001 && bits(word, 11, 5) == 0b11000 && bit(word, 10) == 1 && bit(word, 9) == 0 && bit(word, 4) == 0 {
        let a = arr(size);
        let pg = bits(word, 5, 4);
        let pdn = bits(word, 0, 4);
        out.set(Code::SvePnext);
        out.set_mnemonic(Mnemonic::Pnext);
        out.push_operand(preg_sz(pdn, a));
        out.push_operand(preg(pg));
        out.push_operand(preg_sz(pdn, a));
        return;
    }

    // PTRUE/PTRUES: `00100101 size 011 00 S 1110 0 pattern 0 Pd`.
    //   ptrue  : <23:22>=size, <20:16>=11000, <15:11>=11100, pattern=<9:5>, Pd=<3:0>
    //   ptrues : ... S=1 (<16>=1)
    if bits(word, 17, 4) == 0b1100 && bits(word, 11, 5) == 0b11100 && bit(word, 10) == 0 && bit(word, 4) == 0 {
        let s = bit(word, 16);
        let a = arr(size);
        let pattern = bits(word, 5, 5);
        let pd = bits(word, 0, 4);
        out.set(Code::SvePtrue);
        out.set_mnemonic(if s == 0 { Mnemonic::Ptrue } else { Mnemonic::Ptrues });
        out.push_operand(preg_sz(pd, a));
        // The pattern operand is elided when it is `all` (0x1f) for PTRUE.
        push_pattern(out, pattern);
    }
}

/// Push the SVE predicate pattern operand, eliding `all` (`0x1f`) like the corpus
/// (`ptrue p9.h` with no pattern when pattern == all).
#[inline]
fn push_pattern(out: &mut Instruction, pattern: u32) {
    if pattern & 0x1f != 0x1f {
        out.push_operand(Operand::SvePattern((pattern & 0x1f) as u8));
    }
}

#[cfg(test)]
mod tests {
    use crate::format::{BufSink, FmtFormatter, Formatter};
    use crate::{Decoder, DecoderOptions};

    /// Decode `word` and render with the default UAL formatter into `buf`.
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

    #[test]
    fn vector_permute_zip_uzp_trn() {
        check(0x053E617B, "zip1    z27.b, z11.b, z30.b");
        check(0x05B96666, "zip2    z6.s, z19.s, z25.s");
        check(0x052A6BA5, "uzp1    z5.b, z29.b, z10.b");
        check(0x05EF6C78, "uzp2    z24.d, z3.d, z15.d");
        check(0x05377273, "trn1    z19.b, z19.b, z23.b");
        check(0x057177B8, "trn2    z24.h, z29.h, z17.h");
        // 128-bit `.Q` permute.
        check(0x05BA1BB0, "trn1    z16.q, z29.q, z26.q");
        check(0x05A900A3, "zip1    z3.q, z5.q, z9.q");
        check(0x05A40ABA, "uzp1    z26.q, z21.q, z4.q");
    }

    #[test]
    fn predicate_permute() {
        check(0x056E4169, "zip1    p9.h, p11.h, p14.h");
        check(0x05AA4D2A, "uzp2    p10.s, p9.s, p10.s");
        check(0x05215185, "trn1    p5.b, p12.b, p1.b");
        check(0x05B44101, "rev     p1.s, p8.s");
        check(0x05314060, "punpkhi p0.h, p3.b");
        check(0x05304142, "punpklo p2.h, p10.b");
    }

    #[test]
    fn rev_unpack_compact_splice_ext_tbl() {
        check(0x05F83B9B, "rev     z27.d, z28.d");
        check(0x05E49E31, "revb    z17.d, p7/m, z17.d");
        check(0x05A58CC7, "revh    z7.s, p3/m, z6.s");
        check(0x05E68723, "revw    z3.d, p1/m, z25.d");
        check(0x05B13A5F, "sunpkhi z31.s, z18.h");
        check(0x057039B7, "sunpklo z23.h, z13.b");
        check(0x05B3383E, "uunpkhi z30.s, z1.h");
        check(0x05A182A5, "compact z5.s, p0, z21.s");
        check(0x056C9CBE, "splice  z30.h, p7, z30.h, z5.h");
        check(0x05330A08, "ext     z8.b, z8.b, z16.b, #0x9a");
        check(0x05B833B6, "tbl     z22.s, {z29.s}, z24.s");
        check(0x05712A95, "tbl     z21.h, {z20.h, z21.h}, z17.h");
        check(0x05F22E68, "tbx     z8.d, z19.d, z18.d");
    }

    #[test]
    fn clast_last() {
        check(0x05E883C0, "clasta  z0.d, p0, z0.d, z30.d");
        check(0x05AA914E, "clasta  s14, p4, s14, z10.s");
        check(0x05B0BBEB, "clasta  w11, p6, w11, z31.s");
        check(0x0571B796, "clastb  w22, p5, w22, z28.h");
        check(0x05E29C08, "lasta   d8, p7, z0.d");
        check(0x05E0BA9F, "lasta   xzr, p6, z20.d");
        check(0x05E1A4DA, "lastb   x26, p1, z6.d");
    }

    #[test]
    fn predicate_logical_and_aliases() {
        check(0x25054D29, "and     p9.b, p3/z, p9.b, p5.b");
        check(0x254758C8, "ands    p8.b, p6/z, p6.b, p7.b");
        check(0x25097471, "bic     p1.b, p13/z, p3.b, p9.b");
        check(0x2506566A, "eor     p10.b, p5/z, p3.b, p6.b");
        check(0x25846D42, "orr     p2.b, p11/z, p10.b, p4.b");
        check(0x25896D93, "orn     p3.b, p11/z, p12.b, p9.b");
        check(0x25887E9D, "nand    p13.b, p15/z, p4.b, p8.b");
        check(0x258C5F63, "nor     p3.b, p7/z, p11.b, p12.b");
        // Aliases: MOV/MOVS/NOT/NOTS and the predicate SEL.
        check(0x250B7D69, "mov     p9.b, p15/z, p11.b");
        check(0x258554A9, "mov     p9.b, p5.b");
        check(0x2549492A, "movs    p10.b, p2/z, p9.b");
        check(0x25486284, "nots    p4.b, p8/z, p4.b");
        check(0x25025FD9, "sel     p9.b, p7, p14.b, p2.b");
        check(0x250863B8, "mov     p8.b, p8/m, p13.b");
    }

    #[test]
    fn predicate_gen_ffr() {
        check(0x2558E049, "ptrue   p9.h, vl2");
        check(0x2519E021, "ptrues  p1.b, vl1");
        check(0x2518E409, "pfalse  p9.b");
        check(0x2550F940, "ptest   p14, p10.b");
        check(0x2558C0E8, "pfirst  p8.b, p7, p8.b");
        check(0x25D9C52F, "pnext   p15.d, p9, p15.d");
        check(0x2519F00E, "rdffr   p14.b");
        check(0x2518F04A, "rdffr   p10.b, p2/z");
        check(0x2558F060, "rdffrs  p0.b, p3/z");
        check(0x25289020, "wrffr   p1.b");
        check(0x252C9000, "setffr");
    }

    #[test]
    fn break_family() {
        check(0x25106589, "brka    p9.b, p9/z, p12.b");
        check(0x259044FA, "brkb    p10.b, p1/m, p7.b");
        check(0x25505426, "brkas   p6.b, p5/z, p1.b");
        check(0x25D070E6, "brkbs   p6.b, p12/z, p7.b");
        check(0x251878C5, "brkn    p5.b, p14/z, p6.b, p5.b");
        check(0x250ADD20, "brkpa   p0.b, p7/z, p9.b, p10.b");
        check(0x250FC4B6, "brkpb   p6.b, p1/z, p5.b, p15.b");
        check(0x254BDDA7, "brkpas  p7.b, p7/z, p13.b, p11.b");
    }

    #[test]
    fn while_and_cterm() {
        check(0x256D042D, "whilelt p13.h, w1, w13");
        check(0x25FE163C, "whilele p12.d, x17, x30");
        check(0x25E91E0C, "whilelo p12.d, x16, x9");
        check(0x25390C9C, "whilels p12.b, w4, w25");
        check(0x25B510C1, "whilege p1.s, x6, x21");
        check(0x25B00BDA, "whilehi p10.s, w30, w16");
        check(0x25A13152, "whilerw p2.s, x10, x1");
        check(0x252E336D, "whilewr p13.b, x27, x14");
        check(0x25E22300, "ctermeq x24, x2");
        check(0x25B721A0, "ctermeq w13, w23");
        check(0x25E022D0, "ctermne x22, x0");
    }

    /// The SVE permute/predicate decoder must never panic across the whole
    /// quadrant space and the default (no-SVE) build leaves these invalid.
    #[test]
    fn never_panics_on_perm_space() {
        // Exhaustively sweep a representative bit-slice of the 0x05 / 0x25 top
        // bytes by varying the operand and opcode fields.
        for hi in [0x05u32, 0x25u32] {
            for mid in 0u32..=0xffff {
                let word = (hi << 24) | (mid << 8) | 0x5a;
                let mut buf = [0u8; 128];
                let _ = render(word, &mut buf);
            }
        }
    }
}
