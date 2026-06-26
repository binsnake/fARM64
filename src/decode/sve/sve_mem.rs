//! SVE / SVE2 memory (load / store / prefetch) encodings.
//!
//! Hand-written from the *ARM Architecture Reference Manual* SVE encoding index
//! and validated against the differential corpus (100% parity on the SVE memory
//! groups). This module owns the SVE contiguous / gather / scatter loads and
//! stores, the replicating (`LD1RQ`/`LD1RO`) and broadcast (`LD1R*`) loads, the
//! first-fault (`LDFF1*`) / non-fault (`LDNF1*`) loads, the non-temporal
//! (`LDNT1*`/`STNT1*`) forms (including the SVE2 vector-base gather/scatter
//! variants), the structured (`LD2-4`/`ST2-4`) forms, the prefetches
//! (`PRFB`/`PRFH`/`PRFW`/`PRFD`), and the vector/predicate register transfers
//! (`LDR`/`STR`).
//!
//! Dispatch key (after the SVE group selected on `word<31:29>`): the top byte
//! `word<31:24>` is one of `0x84/0x85` (32-bit-element gather + the `.s`/.d`
//! contiguous loads sharing the quadrant, LD1R*, LDR), `0xA4/0xA5` (contiguous
//! loads, replicating, structured), `0xC4/0xC5` (64-bit-element gather + PRF),
//! `0xE4/0xE5` (contiguous/structured stores + scatter + STR). Inside each, the
//! memory size `msz = word<24:23>`, the form bits `word<22:21>` and the sub-op
//! `word<15:13>` select the precise encoding; the addressing-mode rendering is
//! carried by [`Operand::SveMem`] (and [`Operand::MemExt`] for scalar+scalar).
//!
//! Code identity follows the established convention: one [`Code`] per family +
//! addressing [`Form`] (`Sve<Mnem><Form>`); the [`Mnemonic`] is carried by the
//! code. Every path is total and panic-free; unallocated encodings are left
//! [`Code::Invalid`].

use crate::decode::bits::{bit, bits, sign_extend};
use crate::decode::ldst::prefetch_op_sve;
use crate::enums::{ExtendType, VectorArrangement as VA};
use crate::features::{Feature, FeatureSet};
use crate::instruction::Instruction;
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::{Operand, PredQual, SveMemMode};
use crate::register::{gp_register, Register, RegWidth};

// ---------------------------------------------------------------------------
// Register-bank tables.
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

/// The SVE addressing [`Operand::SveMem`] / [`Code`] form selector.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Form {
    /// Scalar base + immediate (`MUL VL` or plain), incl. `base`-only.
    Imm,
    /// Scalar + scalar (`[Xn, Xm{, lsl #amt}]`).
    Ss,
    /// 32-bit-element gather/scatter (scalar+vec or vec+imm).
    G32,
    /// 64-bit-element gather/scatter (scalar+vec or vec+imm).
    G64,
    /// Vector + immediate (`[Zn.T{, #imm}]`).
    Vi,
    /// Vector + scalar (`[Zn.T, Xm]`, SVE2 `LDNT1`/`STNT1`).
    Vs,
}

/// The [`Code`] for a (mnemonic, addressing form) pair. Generated from the
/// same family table that produced the `Sve*` code rows in `mnemonic.rs`;
/// unallocated pairs fall back to [`Code::Invalid`].
fn code_for(m: Mnemonic, form: Form) -> Code {
    match (m, form) {
        (Mnemonic::Ld1b, Form::Imm) => Code::SveLd1bImm,
        (Mnemonic::Ld1b, Form::Ss) => Code::SveLd1bSs,
        (Mnemonic::Ld1b, Form::G32) => Code::SveLd1bG32,
        (Mnemonic::Ld1b, Form::G64) => Code::SveLd1bG64,
        (Mnemonic::Ld1b, Form::Vi) => Code::SveLd1bVi,
        (Mnemonic::Ld1h, Form::Imm) => Code::SveLd1hImm,
        (Mnemonic::Ld1h, Form::Ss) => Code::SveLd1hSs,
        (Mnemonic::Ld1h, Form::G32) => Code::SveLd1hG32,
        (Mnemonic::Ld1h, Form::G64) => Code::SveLd1hG64,
        (Mnemonic::Ld1h, Form::Vi) => Code::SveLd1hVi,
        (Mnemonic::Ld1w, Form::Imm) => Code::SveLd1wImm,
        (Mnemonic::Ld1w, Form::Ss) => Code::SveLd1wSs,
        (Mnemonic::Ld1w, Form::G32) => Code::SveLd1wG32,
        (Mnemonic::Ld1w, Form::G64) => Code::SveLd1wG64,
        (Mnemonic::Ld1w, Form::Vi) => Code::SveLd1wVi,
        (Mnemonic::Ld1d, Form::Imm) => Code::SveLd1dImm,
        (Mnemonic::Ld1d, Form::Ss) => Code::SveLd1dSs,
        (Mnemonic::Ld1d, Form::G32) => Code::SveLd1dG32,
        (Mnemonic::Ld1d, Form::G64) => Code::SveLd1dG64,
        (Mnemonic::Ld1d, Form::Vi) => Code::SveLd1dVi,
        (Mnemonic::Ld1sb, Form::Imm) => Code::SveLd1sbImm,
        (Mnemonic::Ld1sb, Form::Ss) => Code::SveLd1sbSs,
        (Mnemonic::Ld1sb, Form::G32) => Code::SveLd1sbG32,
        (Mnemonic::Ld1sb, Form::G64) => Code::SveLd1sbG64,
        (Mnemonic::Ld1sb, Form::Vi) => Code::SveLd1sbVi,
        (Mnemonic::Ld1sh, Form::Imm) => Code::SveLd1shImm,
        (Mnemonic::Ld1sh, Form::Ss) => Code::SveLd1shSs,
        (Mnemonic::Ld1sh, Form::G32) => Code::SveLd1shG32,
        (Mnemonic::Ld1sh, Form::G64) => Code::SveLd1shG64,
        (Mnemonic::Ld1sh, Form::Vi) => Code::SveLd1shVi,
        (Mnemonic::Ld1sw, Form::Imm) => Code::SveLd1swImm,
        (Mnemonic::Ld1sw, Form::Ss) => Code::SveLd1swSs,
        (Mnemonic::Ld1sw, Form::G32) => Code::SveLd1swG32,
        (Mnemonic::Ld1sw, Form::G64) => Code::SveLd1swG64,
        (Mnemonic::Ld1sw, Form::Vi) => Code::SveLd1swVi,
        (Mnemonic::Ldff1b, Form::Ss) => Code::SveLdff1bSs,
        (Mnemonic::Ldff1b, Form::G32) => Code::SveLdff1bG32,
        (Mnemonic::Ldff1b, Form::G64) => Code::SveLdff1bG64,
        (Mnemonic::Ldff1b, Form::Vi) => Code::SveLdff1bVi,
        (Mnemonic::Ldff1h, Form::Ss) => Code::SveLdff1hSs,
        (Mnemonic::Ldff1h, Form::G32) => Code::SveLdff1hG32,
        (Mnemonic::Ldff1h, Form::G64) => Code::SveLdff1hG64,
        (Mnemonic::Ldff1h, Form::Vi) => Code::SveLdff1hVi,
        (Mnemonic::Ldff1w, Form::Ss) => Code::SveLdff1wSs,
        (Mnemonic::Ldff1w, Form::G32) => Code::SveLdff1wG32,
        (Mnemonic::Ldff1w, Form::G64) => Code::SveLdff1wG64,
        (Mnemonic::Ldff1w, Form::Vi) => Code::SveLdff1wVi,
        (Mnemonic::Ldff1d, Form::Ss) => Code::SveLdff1dSs,
        (Mnemonic::Ldff1d, Form::G32) => Code::SveLdff1dG32,
        (Mnemonic::Ldff1d, Form::G64) => Code::SveLdff1dG64,
        (Mnemonic::Ldff1d, Form::Vi) => Code::SveLdff1dVi,
        (Mnemonic::Ldff1sb, Form::Ss) => Code::SveLdff1sbSs,
        (Mnemonic::Ldff1sb, Form::G32) => Code::SveLdff1sbG32,
        (Mnemonic::Ldff1sb, Form::G64) => Code::SveLdff1sbG64,
        (Mnemonic::Ldff1sb, Form::Vi) => Code::SveLdff1sbVi,
        (Mnemonic::Ldff1sh, Form::Ss) => Code::SveLdff1shSs,
        (Mnemonic::Ldff1sh, Form::G32) => Code::SveLdff1shG32,
        (Mnemonic::Ldff1sh, Form::G64) => Code::SveLdff1shG64,
        (Mnemonic::Ldff1sh, Form::Vi) => Code::SveLdff1shVi,
        (Mnemonic::Ldff1sw, Form::Ss) => Code::SveLdff1swSs,
        (Mnemonic::Ldff1sw, Form::G32) => Code::SveLdff1swG32,
        (Mnemonic::Ldff1sw, Form::G64) => Code::SveLdff1swG64,
        (Mnemonic::Ldff1sw, Form::Vi) => Code::SveLdff1swVi,
        (Mnemonic::Ldnf1b, Form::Imm) => Code::SveLdnf1bImm,
        (Mnemonic::Ldnf1h, Form::Imm) => Code::SveLdnf1hImm,
        (Mnemonic::Ldnf1w, Form::Imm) => Code::SveLdnf1wImm,
        (Mnemonic::Ldnf1d, Form::Imm) => Code::SveLdnf1dImm,
        (Mnemonic::Ldnf1sb, Form::Imm) => Code::SveLdnf1sbImm,
        (Mnemonic::Ldnf1sh, Form::Imm) => Code::SveLdnf1shImm,
        (Mnemonic::Ldnf1sw, Form::Imm) => Code::SveLdnf1swImm,
        (Mnemonic::Ldnt1b, Form::Imm) => Code::SveLdnt1bImm,
        (Mnemonic::Ldnt1b, Form::Ss) => Code::SveLdnt1bSs,
        (Mnemonic::Ldnt1b, Form::Vs) => Code::SveLdnt1bVs,
        (Mnemonic::Ldnt1h, Form::Imm) => Code::SveLdnt1hImm,
        (Mnemonic::Ldnt1h, Form::Ss) => Code::SveLdnt1hSs,
        (Mnemonic::Ldnt1h, Form::Vs) => Code::SveLdnt1hVs,
        (Mnemonic::Ldnt1w, Form::Imm) => Code::SveLdnt1wImm,
        (Mnemonic::Ldnt1w, Form::Ss) => Code::SveLdnt1wSs,
        (Mnemonic::Ldnt1w, Form::Vs) => Code::SveLdnt1wVs,
        (Mnemonic::Ldnt1d, Form::Imm) => Code::SveLdnt1dImm,
        (Mnemonic::Ldnt1d, Form::Ss) => Code::SveLdnt1dSs,
        (Mnemonic::Ldnt1d, Form::Vs) => Code::SveLdnt1dVs,
        (Mnemonic::Ldnt1sb, Form::Imm) => Code::SveLdnt1sbImm,
        (Mnemonic::Ldnt1sb, Form::Ss) => Code::SveLdnt1sbSs,
        (Mnemonic::Ldnt1sb, Form::Vs) => Code::SveLdnt1sbVs,
        (Mnemonic::Ldnt1sh, Form::Imm) => Code::SveLdnt1shImm,
        (Mnemonic::Ldnt1sh, Form::Ss) => Code::SveLdnt1shSs,
        (Mnemonic::Ldnt1sh, Form::Vs) => Code::SveLdnt1shVs,
        (Mnemonic::Ldnt1sw, Form::Imm) => Code::SveLdnt1swImm,
        (Mnemonic::Ldnt1sw, Form::Ss) => Code::SveLdnt1swSs,
        (Mnemonic::Ldnt1sw, Form::Vs) => Code::SveLdnt1swVs,
        (Mnemonic::Ld1rb, Form::Imm) => Code::SveLd1rbImm,
        (Mnemonic::Ld1rh, Form::Imm) => Code::SveLd1rhImm,
        (Mnemonic::Ld1rw, Form::Imm) => Code::SveLd1rwImm,
        (Mnemonic::Ld1rd, Form::Imm) => Code::SveLd1rdImm,
        (Mnemonic::Ld1rsb, Form::Imm) => Code::SveLd1rsbImm,
        (Mnemonic::Ld1rsh, Form::Imm) => Code::SveLd1rshImm,
        (Mnemonic::Ld1rsw, Form::Imm) => Code::SveLd1rswImm,
        (Mnemonic::Ld1rqb, Form::Imm) => Code::SveLd1rqbImm,
        (Mnemonic::Ld1rqb, Form::Ss) => Code::SveLd1rqbSs,
        (Mnemonic::Ld1rqh, Form::Imm) => Code::SveLd1rqhImm,
        (Mnemonic::Ld1rqh, Form::Ss) => Code::SveLd1rqhSs,
        (Mnemonic::Ld1rqw, Form::Imm) => Code::SveLd1rqwImm,
        (Mnemonic::Ld1rqw, Form::Ss) => Code::SveLd1rqwSs,
        (Mnemonic::Ld1rqd, Form::Imm) => Code::SveLd1rqdImm,
        (Mnemonic::Ld1rqd, Form::Ss) => Code::SveLd1rqdSs,
        (Mnemonic::Ld1rob, Form::Imm) => Code::SveLd1robImm,
        (Mnemonic::Ld1rob, Form::Ss) => Code::SveLd1robSs,
        (Mnemonic::Ld1roh, Form::Imm) => Code::SveLd1rohImm,
        (Mnemonic::Ld1roh, Form::Ss) => Code::SveLd1rohSs,
        (Mnemonic::Ld1row, Form::Imm) => Code::SveLd1rowImm,
        (Mnemonic::Ld1row, Form::Ss) => Code::SveLd1rowSs,
        (Mnemonic::Ld1rod, Form::Imm) => Code::SveLd1rodImm,
        (Mnemonic::Ld1rod, Form::Ss) => Code::SveLd1rodSs,
        (Mnemonic::Ld2b, Form::Imm) => Code::SveLd2bImm,
        (Mnemonic::Ld2b, Form::Ss) => Code::SveLd2bSs,
        (Mnemonic::Ld2h, Form::Imm) => Code::SveLd2hImm,
        (Mnemonic::Ld2h, Form::Ss) => Code::SveLd2hSs,
        (Mnemonic::Ld2w, Form::Imm) => Code::SveLd2wImm,
        (Mnemonic::Ld2w, Form::Ss) => Code::SveLd2wSs,
        (Mnemonic::Ld2d, Form::Imm) => Code::SveLd2dImm,
        (Mnemonic::Ld2d, Form::Ss) => Code::SveLd2dSs,
        (Mnemonic::Ld3b, Form::Imm) => Code::SveLd3bImm,
        (Mnemonic::Ld3b, Form::Ss) => Code::SveLd3bSs,
        (Mnemonic::Ld3h, Form::Imm) => Code::SveLd3hImm,
        (Mnemonic::Ld3h, Form::Ss) => Code::SveLd3hSs,
        (Mnemonic::Ld3w, Form::Imm) => Code::SveLd3wImm,
        (Mnemonic::Ld3w, Form::Ss) => Code::SveLd3wSs,
        (Mnemonic::Ld3d, Form::Imm) => Code::SveLd3dImm,
        (Mnemonic::Ld3d, Form::Ss) => Code::SveLd3dSs,
        (Mnemonic::Ld4b, Form::Imm) => Code::SveLd4bImm,
        (Mnemonic::Ld4b, Form::Ss) => Code::SveLd4bSs,
        (Mnemonic::Ld4h, Form::Imm) => Code::SveLd4hImm,
        (Mnemonic::Ld4h, Form::Ss) => Code::SveLd4hSs,
        (Mnemonic::Ld4w, Form::Imm) => Code::SveLd4wImm,
        (Mnemonic::Ld4w, Form::Ss) => Code::SveLd4wSs,
        (Mnemonic::Ld4d, Form::Imm) => Code::SveLd4dImm,
        (Mnemonic::Ld4d, Form::Ss) => Code::SveLd4dSs,
        (Mnemonic::St1b, Form::Imm) => Code::SveSt1bImm,
        (Mnemonic::St1b, Form::Ss) => Code::SveSt1bSs,
        (Mnemonic::St1b, Form::G32) => Code::SveSt1bG32,
        (Mnemonic::St1b, Form::G64) => Code::SveSt1bG64,
        (Mnemonic::St1b, Form::Vi) => Code::SveSt1bVi,
        (Mnemonic::St1h, Form::Imm) => Code::SveSt1hImm,
        (Mnemonic::St1h, Form::Ss) => Code::SveSt1hSs,
        (Mnemonic::St1h, Form::G32) => Code::SveSt1hG32,
        (Mnemonic::St1h, Form::G64) => Code::SveSt1hG64,
        (Mnemonic::St1h, Form::Vi) => Code::SveSt1hVi,
        (Mnemonic::St1w, Form::Imm) => Code::SveSt1wImm,
        (Mnemonic::St1w, Form::Ss) => Code::SveSt1wSs,
        (Mnemonic::St1w, Form::G32) => Code::SveSt1wG32,
        (Mnemonic::St1w, Form::G64) => Code::SveSt1wG64,
        (Mnemonic::St1w, Form::Vi) => Code::SveSt1wVi,
        (Mnemonic::St1d, Form::Imm) => Code::SveSt1dImm,
        (Mnemonic::St1d, Form::Ss) => Code::SveSt1dSs,
        (Mnemonic::St1d, Form::G32) => Code::SveSt1dG32,
        (Mnemonic::St1d, Form::G64) => Code::SveSt1dG64,
        (Mnemonic::St1d, Form::Vi) => Code::SveSt1dVi,
        (Mnemonic::St2b, Form::Imm) => Code::SveSt2bImm,
        (Mnemonic::St2b, Form::Ss) => Code::SveSt2bSs,
        (Mnemonic::St2h, Form::Imm) => Code::SveSt2hImm,
        (Mnemonic::St2h, Form::Ss) => Code::SveSt2hSs,
        (Mnemonic::St2w, Form::Imm) => Code::SveSt2wImm,
        (Mnemonic::St2w, Form::Ss) => Code::SveSt2wSs,
        (Mnemonic::St2d, Form::Imm) => Code::SveSt2dImm,
        (Mnemonic::St2d, Form::Ss) => Code::SveSt2dSs,
        (Mnemonic::St3b, Form::Imm) => Code::SveSt3bImm,
        (Mnemonic::St3b, Form::Ss) => Code::SveSt3bSs,
        (Mnemonic::St3h, Form::Imm) => Code::SveSt3hImm,
        (Mnemonic::St3h, Form::Ss) => Code::SveSt3hSs,
        (Mnemonic::St3w, Form::Imm) => Code::SveSt3wImm,
        (Mnemonic::St3w, Form::Ss) => Code::SveSt3wSs,
        (Mnemonic::St3d, Form::Imm) => Code::SveSt3dImm,
        (Mnemonic::St3d, Form::Ss) => Code::SveSt3dSs,
        (Mnemonic::St4b, Form::Imm) => Code::SveSt4bImm,
        (Mnemonic::St4b, Form::Ss) => Code::SveSt4bSs,
        (Mnemonic::St4h, Form::Imm) => Code::SveSt4hImm,
        (Mnemonic::St4h, Form::Ss) => Code::SveSt4hSs,
        (Mnemonic::St4w, Form::Imm) => Code::SveSt4wImm,
        (Mnemonic::St4w, Form::Ss) => Code::SveSt4wSs,
        (Mnemonic::St4d, Form::Imm) => Code::SveSt4dImm,
        (Mnemonic::St4d, Form::Ss) => Code::SveSt4dSs,
        (Mnemonic::Stnt1b, Form::Imm) => Code::SveStnt1bImm,
        (Mnemonic::Stnt1b, Form::Ss) => Code::SveStnt1bSs,
        (Mnemonic::Stnt1b, Form::Vs) => Code::SveStnt1bVs,
        (Mnemonic::Stnt1h, Form::Imm) => Code::SveStnt1hImm,
        (Mnemonic::Stnt1h, Form::Ss) => Code::SveStnt1hSs,
        (Mnemonic::Stnt1h, Form::Vs) => Code::SveStnt1hVs,
        (Mnemonic::Stnt1w, Form::Imm) => Code::SveStnt1wImm,
        (Mnemonic::Stnt1w, Form::Ss) => Code::SveStnt1wSs,
        (Mnemonic::Stnt1w, Form::Vs) => Code::SveStnt1wVs,
        (Mnemonic::Stnt1d, Form::Imm) => Code::SveStnt1dImm,
        (Mnemonic::Stnt1d, Form::Ss) => Code::SveStnt1dSs,
        (Mnemonic::Stnt1d, Form::Vs) => Code::SveStnt1dVs,
        (Mnemonic::Prfb, Form::Imm) => Code::SvePrfbImm,
        (Mnemonic::Prfb, Form::Ss) => Code::SvePrfbSs,
        (Mnemonic::Prfb, Form::G32) => Code::SvePrfbG32,
        (Mnemonic::Prfb, Form::G64) => Code::SvePrfbG64,
        (Mnemonic::Prfb, Form::Vi) => Code::SvePrfbVi,
        (Mnemonic::Prfh, Form::Imm) => Code::SvePrfhImm,
        (Mnemonic::Prfh, Form::Ss) => Code::SvePrfhSs,
        (Mnemonic::Prfh, Form::G32) => Code::SvePrfhG32,
        (Mnemonic::Prfh, Form::G64) => Code::SvePrfhG64,
        (Mnemonic::Prfh, Form::Vi) => Code::SvePrfhVi,
        (Mnemonic::Prfw, Form::Imm) => Code::SvePrfwImm,
        (Mnemonic::Prfw, Form::Ss) => Code::SvePrfwSs,
        (Mnemonic::Prfw, Form::G32) => Code::SvePrfwG32,
        (Mnemonic::Prfw, Form::G64) => Code::SvePrfwG64,
        (Mnemonic::Prfw, Form::Vi) => Code::SvePrfwVi,
        (Mnemonic::Prfd, Form::Imm) => Code::SvePrfdImm,
        (Mnemonic::Prfd, Form::Ss) => Code::SvePrfdSs,
        (Mnemonic::Prfd, Form::G32) => Code::SvePrfdG32,
        (Mnemonic::Prfd, Form::G64) => Code::SvePrfdG64,
        (Mnemonic::Prfd, Form::Vi) => Code::SvePrfdVi,
        _ => Code::Invalid,
    }
}

// ---------------------------------------------------------------------------
// Small constructors.
// ---------------------------------------------------------------------------

/// Element arrangement (`.b`/.h`/.s`/.d`) from a 2-bit size code.
#[inline]
fn esz(size: u32) -> VA {
    match size & 3 { 0 => VA::Sb, 1 => VA::Sh, 2 => VA::Ss, _ => VA::Sd }
}

#[inline]
fn pg_z(n: u32) -> Operand {
    Operand::Reg { reg: P[(n & 0xf) as usize], arr: None, lane: None, shift: None, extend: None, pred: Some(PredQual::Zeroing) }
}
#[inline]
fn pg_plain(n: u32) -> Operand {
    Operand::Reg { reg: P[(n & 0xf) as usize], arr: None, lane: None, shift: None, extend: None, pred: None }
}
#[inline]
fn zlist(zt: u32, nreg: u8, a: VA) -> Operand {
    let mut regs = [Register::None; 4];
    let n = nreg.min(4);
    let mut i = 0u32;
    while i < n as u32 { regs[i as usize] = Z[((zt + i) & 0x1f) as usize]; i += 1; }
    Operand::MultiReg { regs, count: n, arr: Some(a), lane: None }
}
#[inline]
fn zbare(n: u32) -> Operand {
    Operand::Reg { reg: Z[(n & 0x1f) as usize], arr: None, lane: None, shift: None, extend: None, pred: None }
}
#[inline]
fn pbare(n: u32) -> Operand {
    Operand::Reg { reg: P[(n & 0xf) as usize], arr: None, lane: None, shift: None, extend: None, pred: None }
}

// --- SVE addressing operands ---------------------------------------------

#[inline]
fn m_mulvl(rn: u32, imm: i32) -> Operand {
    Operand::SveMem { base: gp_register(true, RegWidth::X64, rn as u8), offset: Register::None, arr: None, extend: ExtendType::Uxtx, imm, amount: 0, mode: SveMemMode::ScalarImmMulVl }
}
#[inline]
fn m_imm_hex(rn: u32, imm: i32) -> Operand {
    Operand::SveMem { base: gp_register(true, RegWidth::X64, rn as u8), offset: Register::None, arr: None, extend: ExtendType::Uxtx, imm, amount: 0, mode: SveMemMode::ScalarImm }
}
#[inline]
fn m_imm_dec(rn: u32, imm: i32) -> Operand {
    Operand::SveMem { base: gp_register(true, RegWidth::X64, rn as u8), offset: Register::None, arr: None, extend: ExtendType::Uxtx, imm, amount: 0, mode: SveMemMode::ScalarImmDec }
}
/// `[Xn, Xm{, lsl #amt}]` via [`Operand::MemExt`] (S-bit packed so the
/// formatter shows `lsl #amt` iff `amt != 0`).
#[inline]
fn m_ss(rn: u32, rm: u32, amt: u8) -> Operand {
    let shift = if amt != 0 { 0x80 | amt } else { 0 };
    Operand::MemExt { base: gp_register(true, RegWidth::X64, rn as u8), index: gp_register(false, RegWidth::X64, rm as u8), extend: ExtendType::Uxtx, shift }
}
/// `[Xn, Zm.T{, <mod> #amt}]`; `amount == 0xFF` suppresses the `#amt`.
#[inline]
fn m_xz(rn: u32, zm: u32, a: VA, ext: ExtendType, amount: u8) -> Operand {
    Operand::SveMem { base: gp_register(true, RegWidth::X64, rn as u8), offset: Z[(zm & 0x1f) as usize], arr: Some(a), extend: ext, imm: 0, amount, mode: SveMemMode::ScalarVec }
}
/// `[Zn.T{, #imm}]`.
#[inline]
fn m_vi(zn: u32, a: VA, imm: i32) -> Operand {
    Operand::SveMem { base: Z[(zn & 0x1f) as usize], offset: Register::None, arr: Some(a), extend: ExtendType::Uxtx, imm, amount: 0, mode: SveMemMode::VecImm }
}
/// `[Zn.T, Xm]`.
#[inline]
fn m_vs(zn: u32, a: VA, rm: u32) -> Operand {
    Operand::SveMem { base: Z[(zn & 0x1f) as usize], offset: gp_register(false, RegWidth::X64, rm as u8), arr: Some(a), extend: ExtendType::Uxtx, imm: 0, amount: 0, mode: SveMemMode::VecScalar }
}

/// Resolve the gather/scatter modifier + amount. `xs` selects `Sxtw`(1)/`Uxtw`(0)
/// for the 32-bit-unpacked offset; `packed` selects the 64-bit offset (`Uxtx` ->
/// `lsl`). The `#amt` (= `amt`) is shown only when `scaled && amt != 0`;
/// otherwise `0xFF` suppresses it (matching the corpus, which omits `#0x0`).
#[inline]
fn gmod(xs: u32, packed: bool, scaled: bool, amt: u32) -> (ExtendType, u8) {
    let ext = if packed { ExtendType::Uxtx } else if xs == 1 { ExtendType::Sxtw } else { ExtendType::Uxtw };
    let amount = if scaled && amt != 0 { amt as u8 } else { 0xFF };
    (ext, amount)
}

// ===========================================================================
// Mnemonic resolution helpers (string-free; map (size, kind) -> Mnemonic).
// ===========================================================================

/// Contiguous LOAD dtype table (`word<24:21>`) -> (mnemonic, element arrangement).
fn load_dtype(dt: u32) -> (Mnemonic, VA) {
    match dt & 0xf {
        0 => (Mnemonic::Ld1b, VA::Sb), 1 => (Mnemonic::Ld1b, VA::Sh), 2 => (Mnemonic::Ld1b, VA::Ss), 3 => (Mnemonic::Ld1b, VA::Sd),
        4 => (Mnemonic::Ld1sw, VA::Sd), 5 => (Mnemonic::Ld1h, VA::Sh), 6 => (Mnemonic::Ld1h, VA::Ss), 7 => (Mnemonic::Ld1h, VA::Sd),
        8 => (Mnemonic::Ld1sh, VA::Sd), 9 => (Mnemonic::Ld1sh, VA::Ss), 10 => (Mnemonic::Ld1w, VA::Ss), 11 => (Mnemonic::Ld1w, VA::Sd),
        12 => (Mnemonic::Ld1sb, VA::Sd), 13 => (Mnemonic::Ld1sb, VA::Ss), 14 => (Mnemonic::Ld1sb, VA::Sh), _ => (Mnemonic::Ld1d, VA::Sd),
    }
}
/// Contiguous STORE dtype table (`word<24:21>`) -> (mnemonic, element); `None` for
/// the unallocated rows and the `STR` rows (handled separately).
fn store_dtype(dt: u32) -> Option<(Mnemonic, VA)> {
    Some(match dt & 0xf {
        0 => (Mnemonic::St1b, VA::Sb), 1 => (Mnemonic::St1b, VA::Sh), 2 => (Mnemonic::St1b, VA::Ss), 3 => (Mnemonic::St1b, VA::Sd),
        5 => (Mnemonic::St1h, VA::Sh), 6 => (Mnemonic::St1h, VA::Ss), 7 => (Mnemonic::St1h, VA::Sd),
        10 => (Mnemonic::St1w, VA::Ss), 11 => (Mnemonic::St1w, VA::Sd), 15 => (Mnemonic::St1d, VA::Sd),
        _ => return None,
    })
}
/// LSL amount for a contiguous load/store from the mnemonic memory size.
fn lsl_of_mnem(m: Mnemonic) -> u8 {
    match m {
        Mnemonic::Ld1b | Mnemonic::Ld1sb | Mnemonic::St1b => 0,
        Mnemonic::Ld1h | Mnemonic::Ld1sh | Mnemonic::St1h => 1,
        Mnemonic::Ld1w | Mnemonic::Ld1sw | Mnemonic::St1w => 2,
        Mnemonic::Ld1d | Mnemonic::St1d => 3,
        _ => 0,
    }
}

/// Gather/scatter loaded-element mnemonics for `(msz, signed, ff)`. Returns the
/// element-size letter index into the `Ld1{b,h,w,d}`/`Ld1s{b,h,w}` families.
fn gather_mnem(msz: u32, signed: bool, ff: bool) -> Mnemonic {
    match (msz & 3, signed, ff) {
        (0, false, false) => Mnemonic::Ld1b, (1, false, false) => Mnemonic::Ld1h, (2, false, false) => Mnemonic::Ld1w, (3, false, false) => Mnemonic::Ld1d,
        (0, false, true) => Mnemonic::Ldff1b, (1, false, true) => Mnemonic::Ldff1h, (2, false, true) => Mnemonic::Ldff1w, (3, false, true) => Mnemonic::Ldff1d,
        (0, true, false) => Mnemonic::Ld1sb, (1, true, false) => Mnemonic::Ld1sh, (2, true, false) => Mnemonic::Ld1sw, (3, true, false) => Mnemonic::Ld1d,
        (0, true, true) => Mnemonic::Ldff1sb, (1, true, true) => Mnemonic::Ldff1sh, (2, true, true) => Mnemonic::Ldff1sw, _ => Mnemonic::Ldff1d,
    }
}
/// `true` if a 32-bit-element gather (`.s` destination, `0x84`/`0x85`) load of
/// `(msz, signed)` would write a 64-bit value — which a `.s` vector cannot hold,
/// so the encoding is reserved → UNDEFINED. This is the `dword` memory size
/// (`msz == 3`: `LD1D`/`LDFF1D`/`LDNT1D`) and the *signed* `word` size (`msz ==
/// 2`, which sign-extends to 64-bit: `LD1SW`/`LDFF1SW`/`LDNT1SW`). The 64-bit
/// gather quadrant (`0xC4`/`0xC5`, `.d` destination) is where those forms live.
#[inline]
fn gather32_load_reserved(msz: u32, signed: bool) -> bool {
    msz == 3 || (msz == 2 && signed)
}
/// Scatter `ST1{b,h,w,d}` mnemonic by `msz`.
fn st1_mnem(msz: u32) -> Mnemonic {
    match msz & 3 { 0 => Mnemonic::St1b, 1 => Mnemonic::St1h, 2 => Mnemonic::St1w, _ => Mnemonic::St1d }
}
/// `LDNT1{,s}{b,h,w,d}` mnemonic for the vector-base gather.
fn ldnt1_mnem(msz: u32, signed: bool) -> Mnemonic {
    match (msz & 3, signed) {
        (0, false) => Mnemonic::Ldnt1b, (1, false) => Mnemonic::Ldnt1h, (2, false) => Mnemonic::Ldnt1w, (3, false) => Mnemonic::Ldnt1d,
        (0, true) => Mnemonic::Ldnt1sb, (1, true) => Mnemonic::Ldnt1sh, _ => Mnemonic::Ldnt1sw,
    }
}
/// `STNT1{b,h,w,d}` mnemonic by `msz`.
fn stnt1_mnem(msz: u32) -> Mnemonic {
    match msz & 3 { 0 => Mnemonic::Stnt1b, 1 => Mnemonic::Stnt1h, 2 => Mnemonic::Stnt1w, _ => Mnemonic::Stnt1d }
}
/// `PRF{b,h,w,d}` mnemonic by size code.
fn prf_mnem(sz: u32) -> Mnemonic {
    match sz & 3 { 0 => Mnemonic::Prfb, 1 => Mnemonic::Prfh, 2 => Mnemonic::Prfw, _ => Mnemonic::Prfd }
}
/// Structured `LD{2,3,4}{b,h,w,d}` / `ST{2,3,4}{...}` mnemonic for `(nreg, msz)`.
fn struct_mnem(nreg: u8, msz: u32, store: bool) -> Mnemonic {
    let m = msz & 3;
    match (nreg, store, m) {
        (2, false, 0) => Mnemonic::Ld2b, (2, false, 1) => Mnemonic::Ld2h, (2, false, 2) => Mnemonic::Ld2w, (2, false, 3) => Mnemonic::Ld2d,
        (3, false, 0) => Mnemonic::Ld3b, (3, false, 1) => Mnemonic::Ld3h, (3, false, 2) => Mnemonic::Ld3w, (3, false, 3) => Mnemonic::Ld3d,
        (4, false, 0) => Mnemonic::Ld4b, (4, false, 1) => Mnemonic::Ld4h, (4, false, 2) => Mnemonic::Ld4w, (4, false, 3) => Mnemonic::Ld4d,
        (2, true, 0) => Mnemonic::St2b, (2, true, 1) => Mnemonic::St2h, (2, true, 2) => Mnemonic::St2w, (2, true, 3) => Mnemonic::St2d,
        (3, true, 0) => Mnemonic::St3b, (3, true, 1) => Mnemonic::St3h, (3, true, 2) => Mnemonic::St3w, (3, true, 3) => Mnemonic::St3d,
        (4, true, 0) => Mnemonic::St4b, (4, true, 1) => Mnemonic::St4h, (4, true, 2) => Mnemonic::St4w, _ => Mnemonic::St4d,
    }
}
/// LD1R broadcast table (key = `(msz << 3) | op`, `op >= 4`) -> (mnemonic, element).
fn ld1r_entry(key: u32) -> Option<(Mnemonic, VA)> {
    Some(match key {
        4 => (Mnemonic::Ld1rb, VA::Sb), 5 => (Mnemonic::Ld1rb, VA::Sh), 6 => (Mnemonic::Ld1rb, VA::Ss), 7 => (Mnemonic::Ld1rb, VA::Sd),
        12 => (Mnemonic::Ld1rsw, VA::Sd), 13 => (Mnemonic::Ld1rh, VA::Sh), 14 => (Mnemonic::Ld1rh, VA::Ss), 15 => (Mnemonic::Ld1rh, VA::Sd),
        20 => (Mnemonic::Ld1rsh, VA::Sd), 21 => (Mnemonic::Ld1rsh, VA::Ss), 22 => (Mnemonic::Ld1rw, VA::Ss), 23 => (Mnemonic::Ld1rw, VA::Sd),
        28 => (Mnemonic::Ld1rsb, VA::Sd), 29 => (Mnemonic::Ld1rsb, VA::Ss), 30 => (Mnemonic::Ld1rsb, VA::Sh), 31 => (Mnemonic::Ld1rd, VA::Sd),
        _ => return None,
    })
}
/// Memory access size (bytes) for an `LD1R*` mnemonic (the immediate scale).
fn ld1r_scale(m: Mnemonic) -> i32 {
    match m {
        Mnemonic::Ld1rb | Mnemonic::Ld1rsb => 1,
        Mnemonic::Ld1rh | Mnemonic::Ld1rsh => 2,
        Mnemonic::Ld1rw | Mnemonic::Ld1rsw => 4,
        _ => 8,
    }
}

// ===========================================================================
// Emit helpers.
// ===========================================================================

/// Finish a single-register LOAD: `{Zt.<T>}, Pg/Z, <addr>`.
#[inline]
fn ld(out: &mut Instruction, m: Mnemonic, form: Form, a: VA, zt: u32, pg: u32, addr: Operand) {
    out.set(code_for(m, form));
    out.set_mnemonic(m);
    out.push_operand(zlist(zt, 1, a));
    out.push_operand(pg_z(pg));
    out.push_operand(addr);
}
/// Finish a single-register STORE: `{Zt.<T>}, Pg, <addr>`.
#[inline]
fn st(out: &mut Instruction, m: Mnemonic, form: Form, a: VA, zt: u32, pg: u32, addr: Operand) {
    out.set(code_for(m, form));
    out.set_mnemonic(m);
    out.push_operand(zlist(zt, 1, a));
    out.push_operand(pg_plain(pg));
    out.push_operand(addr);
}
/// Finish a structured load/store: `{Zt.<T>, ...}, Pg{/Z}, <addr>`.
// Decode-shaped helper: each parameter is a distinct already-decoded field.
#[allow(clippy::too_many_arguments)]
#[inline]
fn structured(out: &mut Instruction, m: Mnemonic, form: Form, a: VA, zt: u32, nreg: u8, pg: u32, addr: Operand, store: bool) {
    out.set(code_for(m, form));
    out.set_mnemonic(m);
    out.push_operand(zlist(zt, nreg, a));
    out.push_operand(if store { pg_plain(pg) } else { pg_z(pg) });
    out.push_operand(addr);
}
/// Finish a prefetch: `<prfop>, Pg, <addr>`.
///
/// The prefetch operation `prfop` is a 4-bit field (`word<3:0>`); `word<4>` is
/// RES0 and any non-zero value there is UNDEFINED — leave [`Code::Invalid`].
#[inline]
fn prf(out: &mut Instruction, m: Mnemonic, form: Form, zt: u32, pg: u32, addr: Operand) {
    if zt & 0b1_0000 != 0 {
        return;
    }
    out.set(code_for(m, form));
    out.set_mnemonic(m);
    out.push_operand(prefetch_op_sve(zt));
    out.push_operand(pg_plain(pg));
    out.push_operand(addr);
}

// ===========================================================================
// Top-level dispatch.
// ===========================================================================

/// Decode a single SVE/SVE2 memory instruction `word` at `ip` into `out`.
///
/// Feature-gated on [`Feature::Sve`]; dispatches on the top byte `word<31:24>`.
/// Total and panic-free: unhandled encodings are left [`Code::Invalid`].
#[inline]
pub fn decode(word: u32, ip: u64, features: FeatureSet, out: &mut Instruction) {
    let _ = ip;
    if !features.has(Feature::Sve) {
        return;
    }
    // SVE2.1 quadword (`.q`) contiguous structured + gather/scatter forms share
    // the same top bytes as the byte/half/word/dword families; intercept them
    // first (they occupy otherwise-unallocated sub-op slots) before the legacy
    // dispatch. They are individually feature-gated on FEAT_SVE2p1.
    if features.has(Feature::Sve2p1) {
        decode_qword(word, out);
        if !out.is_invalid() {
            return;
        }
    }
    match bits(word, 24, 8) {
        0xa4 | 0xa5 | 0xe4 | 0xe5 => decode_contig(word, out),
        0x84 | 0x85 | 0xc4 | 0xc5 => decode_gather(word, out),
        _ => {}
    }
}

// ===========================================================================
// FEAT_SVE2p1 quadword (`.q`) load/store family.
// ===========================================================================

/// Decode the SVE2.1 quadword load/store forms, or leave `out` untouched.
///
/// Three groups share the SVE memory top bytes:
/// * `LD1Q` gather (`0xC4`) / `ST1Q` scatter (`0xE4`): `[<Zn>.D{, <Xm>}]`.
/// * `LD{2,3,4}Q` contiguous structured loads (`0xA4`/`0xA5`).
/// * `ST{2,3,4}Q` contiguous structured stores (`0xE4`).
///
/// Each path verifies its full skeleton and leaves [`Code::Invalid`] otherwise,
/// so it is safe to run ahead of the legacy contiguous/gather dispatch.
#[inline]
fn decode_qword(word: u32, out: &mut Instruction) {
    let top = bits(word, 24, 8);
    let pg = bits(word, 10, 3);
    let rn = bits(word, 5, 5);
    let zt = bits(word, 0, 5);
    // SVE2.1 quadword (`.q`) SINGLE-register contiguous loads/stores share these
    // top bytes with `LD1RQ*`/`STNT1*`/`STR`; they sit in otherwise-unallocated
    // sub-op slots. Intercept them first.
    decode_qword_single(word, top, pg, rn, zt, out);
    if !out.is_invalid() {
        return;
    }
    match top {
        // LD1Q gather: 11000100 000 Rm 101 Pg Zn Zt, base Zn.d + offset Xm.
        0xc4 if bits(word, 21, 3) == 0b000 && bits(word, 13, 3) == 0b101 => {
            let rm = bits(word, 16, 5);
            out.set(Code::SveLd1qG);
            out.push_operand(zlist(zt, 1, VA::Sq));
            out.push_operand(pg_z(pg));
            out.push_operand(q_gather_addr(rn, rm));
        }
        // ST1Q scatter: 11100100 001 Rm 001 Pg Zn Zt, base Zn.d + offset Xm.
        0xe4 if bits(word, 21, 3) == 0b001 && bits(word, 13, 3) == 0b001 => {
            let rm = bits(word, 16, 5);
            out.set(Code::SveSt1qS);
            out.push_operand(zlist(zt, 1, VA::Sq));
            out.push_operand(pg_plain(pg));
            out.push_operand(q_gather_addr(rn, rm));
        }
        // LD{2,3,4}Q contiguous structured: 1010010 nreg-1(24:23) 0 form ...
        0xa4 | 0xa5 if bit(word, 22) == 0 => {
            let nreg = match bits(word, 23, 2) {
                0b01 => 2,
                0b10 => 3,
                0b11 => 4,
                _ => return,
            };
            decode_qword_struct(word, nreg, false, pg, rn, zt, out);
        }
        // ST{2,3,4}Q contiguous structured: 1110010 nreg-1(24:22) ...
        0xe4 => {
            let nreg = match bits(word, 22, 3) {
                0b001 => 2,
                0b010 => 3,
                0b011 => 4,
                _ => return,
            };
            decode_qword_struct(word, nreg, true, pg, rn, zt, out);
        }
        _ => {}
    }
}

/// Decode the SVE2.1 quadword (`.q`) SINGLE-register contiguous loads/stores.
///
/// These render `{ <Zt>.Q }` and occupy sub-op slots that legacy `decode_contig`
/// either drops or over-decodes as `LD1RQ*` (the `op=1`/`b20=1` imm slot). The
/// element field selects `LD1W`/`ST1W` (W) vs `LD1D`/`ST1D` (D): loads (`0xa5`)
/// use `<23:22>` = `00` -> W, `10` -> D; stores (`0xe5`) use `00` -> W, `11`
/// -> D. Loads sit at `op<15:13>` = `100` (scalar+scalar, `Rm` = `<20:16>`) and
/// `001` with `<20>=1` (scalar+imm, `MUL VL`); stores at `op` = `010` (ss) and
/// `111` with `<20>=0` (imm).
#[inline]
fn decode_qword_single(word: u32, top: u32, pg: u32, rn: u32, zt: u32, out: &mut Instruction) {
    // Only the `0xa5` (loads) / `0xe5` (stores) top bytes carry these forms.
    // The single-register forms always have `<21> == 0`; `<21> == 1` belongs to
    // the structured `LD{3,4}Q`/`ST{3,4}Q` (which share `op<15:13> == 100`).
    if bit(word, 21) != 0 {
        return;
    }
    let sz = bits(word, 22, 2); // <23:22>
    let op = bits(word, 13, 3); // <15:13>
    let rm = bits(word, 16, 5); // <20:16> (scalar+scalar offset register)
    let i4 = sign_extend(bits(word, 16, 4) as u64, 4) as i32; // <19:16> imm4
    match top {
        0xa5 => {
            // Element: 00 -> W (lsl #2), 10 -> D (lsl #3); others not `.q`.
            let (code_ss, code_imm, lsl) = match sz {
                0b00 => (Code::SveLd1wqSs, Code::SveLd1wqImm, 2u8),
                0b10 => (Code::SveLd1dqSs, Code::SveLd1dqImm, 3u8),
                _ => return,
            };
            if op == 0b100 {
                // scalar + scalar, `[Xn, Xm, lsl #lsl]`. `Xm == 31` (xzr) is the
                // no-offset immediate form → UNDEFINED.
                if rm == 0b11111 {
                    return;
                }
                out.set(code_ss);
                out.push_operand(zlist(zt, 1, VA::Sq));
                out.push_operand(pg_z(pg));
                out.push_operand(m_ss(rn, rm, lsl));
            } else if op == 0b001 && bit(word, 20) == 1 {
                // scalar + imm `[Xn{, #imm, mul vl}]`.
                out.set(code_imm);
                out.push_operand(zlist(zt, 1, VA::Sq));
                out.push_operand(pg_z(pg));
                out.push_operand(m_mulvl(rn, i4));
            }
        }
        0xe5 => {
            // Element: 00 -> W (lsl #2), 11 -> D (lsl #3); others not `.q`.
            let (code_ss, code_imm, lsl) = match sz {
                0b00 => (Code::SveSt1wqSs, Code::SveSt1wqImm, 2u8),
                0b11 => (Code::SveSt1dqSs, Code::SveSt1dqImm, 3u8),
                _ => return,
            };
            if op == 0b010 {
                // scalar + scalar, `[Xn, Xm, lsl #lsl]`. `Xm == 31` (xzr) is the
                // no-offset immediate form → UNDEFINED.
                if rm == 0b11111 {
                    return;
                }
                out.set(code_ss);
                out.push_operand(zlist(zt, 1, VA::Sq));
                out.push_operand(pg_plain(pg));
                out.push_operand(m_ss(rn, rm, lsl));
            } else if op == 0b111 && bit(word, 20) == 0 {
                // scalar + imm `[Xn{, #imm, mul vl}]`.
                out.set(code_imm);
                out.push_operand(zlist(zt, 1, VA::Sq));
                out.push_operand(pg_plain(pg));
                out.push_operand(m_mulvl(rn, i4));
            }
        }
        _ => {}
    }
}

/// Decode the scalar+scalar / scalar+imm tail of a quadword structured form.
#[inline]
fn decode_qword_struct(word: u32, nreg: u8, store: bool, pg: u32, rn: u32, zt: u32, out: &mut Instruction) {
    let ss_sel = if store { 0b000 } else { 0b100 };
    let imm_sel = if store { 0b000 } else { 0b111 };
    if bit(word, 21) == 1 && bits(word, 13, 3) == ss_sel {
        // scalar + scalar, `[Xn, Xm, lsl #4]`. `Xm == 31` (xzr) is UNDEFINED for
        // the structured forms (the no-offset case is the immediate form).
        let rm = bits(word, 16, 5);
        if rm == 0b11111 {
            return;
        }
        let code = qword_struct_code(nreg, store, true);
        out.set(code);
        out.push_operand(zlist(zt, nreg, VA::Sq));
        out.push_operand(if store { pg_plain(pg) } else { pg_z(pg) });
        out.push_operand(m_ss(rn, rm, 4));
    } else if bit(word, 21) == 0 && bit(word, 20) == (if store { 0 } else { 1 }) && bits(word, 13, 3) == imm_sel {
        // scalar + imm `[Xn{, #imm, mul vl}]`, imm4 scaled by nreg.
        let i4 = sign_extend(bits(word, 16, 4) as u64, 4) as i32;
        let code = qword_struct_code(nreg, store, false);
        out.set(code);
        out.push_operand(zlist(zt, nreg, VA::Sq));
        out.push_operand(if store { pg_plain(pg) } else { pg_z(pg) });
        out.push_operand(m_mulvl(rn, i4 * nreg as i32));
    }
}

/// The [`Code`] for a quadword structured `(nreg, store, ss)` form.
fn qword_struct_code(nreg: u8, store: bool, ss: bool) -> Code {
    use Code::*;
    match (nreg, store, ss) {
        (2, false, true) => SveLd2qSs, (2, false, false) => SveLd2qImm,
        (3, false, true) => SveLd3qSs, (3, false, false) => SveLd3qImm,
        (4, false, true) => SveLd4qSs, (4, false, false) => SveLd4qImm,
        (2, true, true) => SveSt2qSs, (2, true, false) => SveSt2qImm,
        (3, true, true) => SveSt3qSs, (3, true, false) => SveSt3qImm,
        (_, true, true) => SveSt4qSs, (_, true, false) => SveSt4qImm,
        (_, false, true) => SveLd4qSs, (_, false, false) => SveLd4qImm,
    }
}

/// `[Zn.D{, Xm}]` for the quadword gather/scatter forms; an `Xm` of `31` (xzr)
/// renders bare (`[Zn.D]`).
#[inline]
fn q_gather_addr(zn: u32, rm: u32) -> Operand {
    if rm == 0b11111 {
        // No offset register: `[Zn.D]` (a vector-base with no immediate).
        m_vi(zn, VA::Sd, 0)
    } else {
        m_vs(zn, VA::Sd, rm)
    }
}

// ===========================================================================
// Contiguous quadrant: 0xA4/0xA5 (loads), 0xE4/0xE5 (stores + scatter).
// ===========================================================================

#[inline]
fn decode_contig(word: u32, out: &mut Instruction) {
    let top = bits(word, 24, 8);
    let msz = bits(word, 23, 2);
    let b22 = bits(word, 22, 1);
    let b21 = bits(word, 21, 1);
    let op = bits(word, 13, 3);
    let dtype = bits(word, 21, 4);
    let pg = bits(word, 10, 3);
    let rn = bits(word, 5, 5);
    let zt = bits(word, 0, 5);
    let rm = bits(word, 16, 5);
    let store = top == 0xe4 || top == 0xe5;
    let e = esz(msz);

    if !store {
        match op {
            0 | 1 => {
                // LD1RQ (b21==0) / LD1RO (b21==1). `word<22>` is RES0 for this
                // replicating-quadword/octword family; non-zero → UNDEFINED.
                if b22 == 1 {
                    return;
                }
                let m = match (b21, msz & 3) {
                    (0, 0) => Mnemonic::Ld1rqb, (0, 1) => Mnemonic::Ld1rqh, (0, 2) => Mnemonic::Ld1rqw, (0, 3) => Mnemonic::Ld1rqd,
                    (_, 0) => Mnemonic::Ld1rob, (_, 1) => Mnemonic::Ld1roh, (_, 2) => Mnemonic::Ld1row, _ => Mnemonic::Ld1rod,
                };
                if op == 0 {
                    // scalar + scalar, lsl #msz. `Xm == 31` (xzr) is the
                    // no-offset immediate form → UNDEFINED here.
                    if rm == 0b11111 {
                        return;
                    }
                    ld(out, m, Form::Ss, e, zt, pg, m_ss(rn, rm, msz as u8));
                } else {
                    // scalar + imm: LD1RQ scales by 16 (decimal radix); LD1RO is
                    // raw (hex). `word<20>` is RES0 for the immediate form →
                    // UNDEFINED if set.
                    if bits(word, 20, 1) == 1 {
                        return;
                    }
                    let i4 = sign_extend(bits(word, 16, 4) as u64, 4) as i32;
                    if b21 == 0 {
                        ld(out, m, Form::Imm, e, zt, pg, m_imm_dec(rn, i4 * 16));
                    } else {
                        ld(out, m, Form::Imm, e, zt, pg, m_imm_hex(rn, i4));
                    }
                }
            }
            2 | 3 => {
                // `Xm == 31` (xzr) is reserved for the non-fault-suppressing
                // `LD1*` scalar+scalar form (`op == 2`): the no-offset case is
                // the immediate form → UNDEFINED. The first-fault `LDFF1*`
                // (`op == 3`) form *does* allow `xzr` (renders `[Xn]`).
                if op == 2 && rm == 0b11111 {
                    return;
                }
                let (m, a) = load_dtype(dtype);
                let mn = if op == 3 { ff_of(m) } else { m };
                ld(out, mn, Form::Ss, a, zt, pg, m_ss(rn, rm, lsl_of_mnem(m) as u32 as u8));
            }
            5 => {
                let (m, a) = load_dtype(dtype);
                let i4 = sign_extend(bits(word, 16, 4) as u64, 4) as i32;
                let mn = if bits(word, 20, 1) == 1 { nf_of(m) } else { m };
                ld(out, mn, Form::Imm, a, zt, pg, m_mulvl(rn, i4));
            }
            6 | 7 => {
                // The scalar+scalar form (`op == 6`, both `LDNT1*` and the
                // structured `LD{2,3,4}*`) is UNDEFINED with `Xm == 31` (xzr).
                if op == 6 && rm == 0b11111 {
                    return;
                }
                // The scalar+imm form (`op == 7`) has `word<20>` RES0; a set bit
                // is UNDEFINED. (The `.q` `LD{2,3,4}Q` forms that reuse this slot
                // are decoded earlier in `decode_qword`.)
                if op == 7 && bits(word, 20, 1) == 1 {
                    return;
                }
                let nr = (b22 * 2 + b21) as u8; // 0 -> LDNT1, else nreg-1
                let e2 = esz(msz);
                if nr == 0 {
                    let m = ldnt1_mnem(msz, false);
                    if op == 6 {
                        ld(out, m, Form::Ss, e2, zt, pg, m_ss(rn, rm, msz as u8));
                    } else {
                        let i4 = sign_extend(bits(word, 16, 4) as u64, 4) as i32;
                        ld(out, m, Form::Imm, e2, zt, pg, m_mulvl(rn, i4));
                    }
                } else {
                    let nreg = nr + 1;
                    let m = struct_mnem(nreg, msz, false);
                    if op == 6 {
                        structured(out, m, Form::Ss, e2, zt, nreg, pg, m_ss(rn, rm, msz as u8), false);
                    } else {
                        let i4 = sign_extend(bits(word, 16, 4) as u64, 4) as i32;
                        structured(out, m, Form::Imm, e2, zt, nreg, pg, m_mulvl(rn, i4 * nreg as i32), false);
                    }
                }
            }
            _ => {}
        }
        return;
    }

    // ----- store quadrant -----
    // STR (vector/predicate): top 0xe5, msz==3, b22==0, op in {0,2}.
    if top == 0xe5 && msz == 3 && b22 == 0 && (op == 0 || op == 2) {
        decode_ldr_str(word, out, true);
        return;
    }
    match op {
        2 => {
            // `Xm == 31` (xzr) is the no-offset immediate form → UNDEFINED.
            if rm == 0b11111 {
                return;
            }
            if let Some((m, a)) = store_dtype(dtype) {
                st(out, m, Form::Ss, a, zt, pg, m_ss(rn, rm, lsl_of_mnem(m) as u32 as u8));
            }
        }
        3 => {
            // `Xm == 31` (xzr) UNDEFINED for `STNT1*`/structured scalar+scalar.
            if rm == 0b11111 {
                return;
            }
            let nr = (b22 * 2 + b21) as u8;
            let e2 = esz(msz);
            if nr == 0 {
                let m = stnt1_mnem(msz);
                st(out, m, Form::Ss, e2, zt, pg, m_ss(rn, rm, msz as u8));
            } else {
                let nreg = nr + 1;
                let m = struct_mnem(nreg, msz, true);
                structured(out, m, Form::Ss, e2, zt, nreg, pg, m_ss(rn, rm, msz as u8), true);
            }
        }
        7 => {
            let e2 = esz(msz);
            if bits(word, 20, 1) == 0 {
                // ST1 scalar+imm (dtype-based).
                if let Some((m, a)) = store_dtype(dtype) {
                    let i4 = sign_extend(bits(word, 16, 4) as u64, 4) as i32;
                    st(out, m, Form::Imm, a, zt, pg, m_mulvl(rn, i4));
                }
            } else {
                let nr = (b22 * 2 + b21) as u8;
                let i4 = sign_extend(bits(word, 16, 4) as u64, 4) as i32;
                if nr == 0 {
                    let m = stnt1_mnem(msz);
                    st(out, m, Form::Imm, e2, zt, pg, m_mulvl(rn, i4));
                } else {
                    let nreg = nr + 1;
                    let m = struct_mnem(nreg, msz, true);
                    structured(out, m, Form::Imm, e2, zt, nreg, pg, m_mulvl(rn, i4 * nreg as i32), true);
                }
            }
        }
        _ => decode_scatter(word, out),
    }
}

/// First-fault variant of a contiguous load mnemonic.
fn ff_of(m: Mnemonic) -> Mnemonic {
    match m {
        Mnemonic::Ld1b => Mnemonic::Ldff1b, Mnemonic::Ld1h => Mnemonic::Ldff1h, Mnemonic::Ld1w => Mnemonic::Ldff1w, Mnemonic::Ld1d => Mnemonic::Ldff1d,
        Mnemonic::Ld1sb => Mnemonic::Ldff1sb, Mnemonic::Ld1sh => Mnemonic::Ldff1sh, Mnemonic::Ld1sw => Mnemonic::Ldff1sw, other => other,
    }
}
/// Non-fault variant of a contiguous load mnemonic.
fn nf_of(m: Mnemonic) -> Mnemonic {
    match m {
        Mnemonic::Ld1b => Mnemonic::Ldnf1b, Mnemonic::Ld1h => Mnemonic::Ldnf1h, Mnemonic::Ld1w => Mnemonic::Ldnf1w, Mnemonic::Ld1d => Mnemonic::Ldnf1d,
        Mnemonic::Ld1sb => Mnemonic::Ldnf1sb, Mnemonic::Ld1sh => Mnemonic::Ldnf1sh, Mnemonic::Ld1sw => Mnemonic::Ldnf1sw, other => other,
    }
}

// ===========================================================================
// Scatter (store) quadrant low ops: STNT1 (vector base), ST1 scatter.
// ===========================================================================

#[inline]
fn decode_scatter(word: u32, out: &mut Instruction) {
    let msz = bits(word, 23, 2);
    let b22 = bits(word, 22, 1);
    let b21 = bits(word, 21, 1);
    let op = bits(word, 13, 3);
    let pg = bits(word, 10, 3);
    let rn = bits(word, 5, 5);
    let zt = bits(word, 0, 5);
    let rm = bits(word, 16, 5);
    let zm = bits(word, 16, 5);
    match op {
        1 => {
            // STNT1 vector base + scalar offset: base Zn = Rn, offset Xm = Rm.
            // `b21` (the scale bit) is RES0 here → reserved if set; a `.s`
            // (`b22==1`) offset cannot store a `dword` (`msz==3`).
            if b21 == 1 || (msz == 3 && b22 == 1) {
                return;
            }
            let oe = if b22 == 1 { VA::Ss } else { VA::Sd };
            st(out, stnt1_mnem(msz), Form::Vs, oe, zt, pg, m_vs(rn, oe, rm));
        }
        4 | 6 => {
            // ST1 scalar+vec, 32-bit-unpacked offset (uxtw/sxtw). element b22?s:d.
            // A `byte` store (`msz==0`) has no scale, so `b21==1` is reserved; a
            // `.s` (`b22==1`) offset cannot store a `dword` (`msz==3`).
            if (msz == 0 && b21 == 1) || (msz == 3 && b22 == 1) {
                return;
            }
            let oe = if b22 == 1 { VA::Ss } else { VA::Sd };
            let xs = if op == 6 { 1 } else { 0 };
            let (ext, amt) = gmod(xs, false, b21 == 1, msz);
            let f = if oe == VA::Ss { Form::G32 } else { Form::G64 };
            st(out, st1_mnem(msz), f, oe, zt, pg, m_xz(rn, zm, oe, ext, amt));
        }
        5 => {
            if b22 == 0 {
                // ST1 scalar+vec 64-bit [Xn, Zm.d{, lsl #msz}]. A `byte` store
                // (`msz==0`) has no scale → `b21==1` reserved.
                if msz == 0 && b21 == 1 {
                    return;
                }
                let (ext, amt) = gmod(0, true, b21 == 1, msz);
                st(out, st1_mnem(msz), Form::G64, VA::Sd, zt, pg, m_xz(rn, zm, VA::Sd, ext, amt));
            } else {
                // ST1 vec+imm [Zn.elt, #imm], element b21?s:d. A `.s` element
                // (`b21==1`) cannot store a `dword` (`msz==3`) → reserved.
                if msz == 3 && b21 == 1 {
                    return;
                }
                let oe = if b21 == 1 { VA::Ss } else { VA::Sd };
                let imm = (bits(word, 16, 5) as i32) * (1i32 << msz);
                let f = if oe == VA::Ss { Form::G32 } else { Form::G64 };
                st(out, st1_mnem(msz), f, oe, zt, pg, m_vi(rn, oe, imm));
            }
        }
        _ => {}
    }
}

// ===========================================================================
// Gather quadrant: 0x84/0x85 (32-bit), 0xC4/0xC5 (64-bit). Loads + PRF + LDR.
// ===========================================================================

#[inline]
fn decode_gather(word: u32, out: &mut Instruction) {
    let top = bits(word, 24, 8);
    let msz = bits(word, 23, 2);
    let b22 = bits(word, 22, 1);
    let b21 = bits(word, 21, 1);
    let op = bits(word, 13, 3);
    let pg = bits(word, 10, 3);
    let rn = bits(word, 5, 5);
    let zt = bits(word, 0, 5);
    let rm = bits(word, 16, 5);
    let zm = bits(word, 16, 5);
    let is64 = top == 0xc4 || top == 0xc5;
    let dst = if is64 { VA::Sd } else { VA::Ss };

    // ---- 0x85 msz==3: LDR (op0/op2, b22==0) and PRF-contiguous (b22==1). ----
    if top == 0x85 && msz == 3 {
        if (op == 0 || op == 2) && b22 == 0 {
            decode_ldr_str(word, out, false);
            return;
        }
        if b22 == 1 && op < 4 {
            let i6 = sign_extend(bits(word, 16, 6) as u64, 6) as i32;
            prf(out, prf_mnem(op), Form::Imm, zt, pg, m_mulvl(rn, i6));
            return;
        }
    }

    if !is64 {
        decode_gather_32(word, top, msz, b22, b21, op, pg, rn, zt, rm, zm, dst, out);
    } else {
        decode_gather_64(word, msz, b22, b21, op, pg, rn, zt, rm, zm, dst, out);
    }
}

/// 32-bit-element gather quadrant (0x84/0x85). Offsets are always 32-bit
/// unpacked (`uxtw`/`sxtw`); `b22==1, op>=4` is the LD1R* broadcast region.
#[allow(clippy::too_many_arguments)]
#[inline]
fn decode_gather_32(word: u32, _top: u32, msz: u32, b22: u32, b21: u32, op: u32, pg: u32, rn: u32, zt: u32, rm: u32, zm: u32, dst: VA, out: &mut Instruction) {
    // Region 10/11, op>=4: LD1R* broadcast.
    if b22 == 1 && op >= 4 {
        decode_ld1r(word, out);
        return;
    }
    // PRF scalar+vec (only msz==0) at region 01/11, op0-3.
    if msz == 0 && b21 == 1 && op < 4 {
        let (ext, amt) = gmod(b22, false, op > 0, op);
        prf(out, prf_mnem(op), Form::G32, zt, pg, m_xz(rn, zm, dst, ext, amt));
        return;
    }
    // Vector+imm gather: region 01 op4-7 (region 11 op>=4 handled as LD1R above).
    if b21 == 1 && op >= 4 {
        let ff = op == 5 || op == 7;
        let signed = op == 4 || op == 5;
        // A 64-bit-element load into a `.s` vector is reserved → UNDEFINED.
        if gather32_load_reserved(msz, signed) {
            return;
        }
        let m = gather_mnem(msz, signed, ff);
        let imm = (bits(word, 16, 5) as i32) * (1i32 << msz);
        ld(out, m, Form::Vi, dst, zt, pg, m_vi(rn, dst, imm));
        return;
    }
    // Scalar+vec gather op0-3 (xs=b22, scaled=b21).
    if op <= 3 {
        let ff = op == 1 || op == 3;
        let signed = op == 0 || op == 1;
        // A 64-bit-element load into a `.s` vector is reserved → UNDEFINED.
        if gather32_load_reserved(msz, signed) {
            return;
        }
        let m = gather_mnem(msz, signed, ff);
        let (ext, amt) = gmod(b22, false, b21 == 1, msz);
        ld(out, m, Form::G32, dst, zt, pg, m_xz(rn, zm, dst, ext, amt));
        return;
    }
    // Region 00, op4-7: LDNT1 (vector base) op4/5, PRF scalar+scalar op6, PRF vec+imm op7.
    match op {
        // PRF scalar+scalar: `Xm == 31` (xzr) is the no-offset immediate form
        // → UNDEFINED (same reservation as the contiguous ld/st ss forms).
        6 if rm != 0b11111 => prf(out, prf_mnem(msz), Form::Ss, zt, pg, m_ss(rn, rm, msz as u8)),
        7 => {
            let imm = (bits(word, 16, 5) as i32) * (1i32 << msz);
            prf(out, prf_mnem(msz), Form::Vi, zt, pg, m_vi(rn, dst, imm));
        }
        4 | 5 => {
            // Vector-base `LDNT1*` into a `.s` vector: a 64-bit-element load is
            // reserved → UNDEFINED (`LDNT1D` msz==3, `LDNT1SW` signed word).
            if gather32_load_reserved(msz, op == 4) {
                return;
            }
            let m = ldnt1_mnem(msz, op == 4);
            ld(out, m, Form::Vs, dst, zt, pg, m_vs(rn, dst, rm));
        }
        _ => {}
    }
}

/// 64-bit-element gather quadrant (0xC4/0xC5). Offsets are 32-bit unpacked
/// (`uxtw`/`sxtw`, `b22b21` regions 00/01/10/11... op0-3) or 64-bit packed
/// (`lsl`/none, op4-7 regions 10/11). PRF occupies `msz==0` slots.
#[allow(clippy::too_many_arguments)]
#[inline]
fn decode_gather_64(word: u32, msz: u32, b22: u32, b21: u32, op: u32, pg: u32, rn: u32, zt: u32, rm: u32, zm: u32, dst: VA, out: &mut Instruction) {
    // PRF scalar+vec op0-3 (msz==0, region 01 uxtw# / region 11 sxtw#).
    if msz == 0 && op < 4 && b21 == 1 {
        let (ext, amt) = gmod(b22, false, true, op);
        prf(out, prf_mnem(op), Form::G64, zt, pg, m_xz(rn, zm, dst, ext, amt));
        return;
    }
    // PRF op4-7 region 11 (64-bit packed: plain/lsl).
    if msz == 0 && op >= 4 && b22 == 1 && b21 == 1 {
        let sz = op - 4;
        let (ext, amt) = gmod(0, true, sz > 0, sz);
        prf(out, prf_mnem(sz), Form::G64, zt, pg, m_xz(rn, zm, dst, ext, amt));
        return;
    }
    // PRF vec+imm op7 region 00 (prfb only, msz==0).
    if msz == 0 && op == 7 && b22 == 0 && b21 == 0 {
        let imm = bits(word, 16, 5) as i32;
        prf(out, Mnemonic::Prfb, Form::Vi, zt, pg, m_vi(rn, dst, imm));
        return;
    }
    // Gather scalar+vec op0-3 (32-bit unpacked: xs=b22, scaled=b21).
    if op <= 3 {
        let ff = op == 1 || op == 3;
        let signed = op == 0 || op == 1;
        let m = gather_mnem(msz, signed, ff);
        let (ext, amt) = gmod(b22, false, b21 == 1, msz);
        ld(out, m, Form::G64, dst, zt, pg, m_xz(rn, zm, dst, ext, amt));
        return;
    }
    // op4-7:
    if b22 == 0 && b21 == 0 {
        // Region 00: LDNT1 (vector base) op4/6, PRF vec+imm op7 (msz>0).
        match op {
            7 => {
                let imm = (bits(word, 16, 5) as i32) * (1i32 << msz);
                prf(out, prf_mnem(msz), Form::Vi, zt, pg, m_vi(rn, dst, imm));
            }
            4 | 6 => {
                let m = ldnt1_mnem(msz, op == 4);
                ld(out, m, Form::Vs, dst, zt, pg, m_vs(rn, dst, rm));
            }
            _ => {}
        }
        return;
    }
    if b22 == 0 && b21 == 1 {
        // Region 01: vector+imm gather op4-7.
        let ff = op == 5 || op == 7;
        let signed = op == 4 || op == 5;
        let m = gather_mnem(msz, signed, ff);
        let imm = (bits(word, 16, 5) as i32) * (1i32 << msz);
        ld(out, m, Form::Vi, dst, zt, pg, m_vi(rn, dst, imm));
        return;
    }
    // Region 10/11 op4-7: 64-bit packed offset (plain b21==0 / lsl b21==1).
    let ff = op == 5 || op == 7;
    let signed = op == 4 || op == 5;
    let m = gather_mnem(msz, signed, ff);
    let (ext, amt) = gmod(0, true, b21 == 1, msz);
    ld(out, m, Form::G64, dst, zt, pg, m_xz(rn, zm, dst, ext, amt));
}

/// LD1R* broadcast (scalar + immediate). The immediate is `imm6 * mem_size`.
#[inline]
fn decode_ld1r(word: u32, out: &mut Instruction) {
    let key = (bits(word, 23, 2) << 3) | bits(word, 13, 3);
    let (m, a) = match ld1r_entry(key) {
        Some(v) => v,
        None => return,
    };
    let imm = (bits(word, 16, 6) as i32) * ld1r_scale(m);
    ld(out, m, Form::Imm, a, bits(word, 0, 5), bits(word, 10, 3), m_imm_hex(bits(word, 5, 5), imm));
}

// ===========================================================================
// LDR / STR (vector & predicate register).
// ===========================================================================

/// `LDR`/`STR` of a `Z` (vector) or `P` (predicate) register:
/// `[<Xn|SP>{, #imm, MUL VL}]`. `word<14>` selects `Z`(1)/`P`(0); the 9-bit
/// VL-scaled immediate is `word<21:16>:word<12:10>`.
#[inline]
fn decode_ldr_str(word: u32, out: &mut Instruction, store: bool) {
    let rn = bits(word, 5, 5);
    let imm9 = sign_extend(((bits(word, 16, 6) << 3) | bits(word, 10, 3)) as u64, 9) as i32;
    let is_vec = bits(word, 14, 1) == 1;
    let addr = m_mulvl(rn, imm9);
    if is_vec {
        let code = if store { Code::SveStrZ } else { Code::SveLdrZ };
        out.set(code);
        out.set_mnemonic(if store { Mnemonic::Str } else { Mnemonic::Ldr });
        out.push_operand(zbare(bits(word, 0, 5)));
        out.push_operand(addr);
    } else {
        // Predicate register transfer: the target is `Pt<3:0>`; `word<4>` is
        // RES0 and any non-zero value there is UNDEFINED.
        if bits(word, 4, 1) == 1 {
            return;
        }
        let code = if store { Code::SveStrP } else { Code::SveLdrP };
        out.set(code);
        out.set_mnemonic(if store { Mnemonic::Str } else { Mnemonic::Ldr });
        out.push_operand(pbare(bits(word, 0, 4)));
        out.push_operand(addr);
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
    fn contiguous_scalar_imm_and_scalar() {
        // LD1B scalar+imm (MUL VL) and scalar+scalar; LSL amount = mem size.
        check(0xA40A4000, "ld1b    {z0.b}, p0/z, [x0, x10]");
        check(0xA5E54000, "ld1d    {z0.d}, p0/z, [x0, x5, lsl #0x3]");
        // negative MUL VL immediate renders in decimal (SVE radix).
        check(0xA43AA5D5, "ldnf1b  {z21.h}, p1/z, [x14, #-6, mul vl]");
    }

    #[test]
    fn replicating_and_broadcast() {
        // LD1RQ scalar+imm scales by 16 and renders the byte offset in decimal.
        check(0xA40E2A74, "ld1rqb  {z20.b}, p2/z, [x19, #-32]");
        // LD1RO scalar+imm renders the raw (signed-hex) multiple.
        check(0xA42F2549, "ld1rob  {z9.b}, p1/z, [x10, #-0x1]");
        // LD1R* broadcast: immediate is imm6 * element size.
        check(0x84408C05, "ld1rb   {z5.b}, p3/z, [x0]");
        check(0x858010A0, "ldr     p0, [x5, #0x4, mul vl]");
    }

    #[test]
    fn structured_loads_stores() {
        check(0xA522FB80, "ld2w    {z0.s, z1.s}, p6/z, [x28, #0x4, mul vl]");
        check(0xE463F31B, "st1b    {z27.d}, p4, [x24, #0x3, mul vl]");
    }

    #[test]
    fn gather_and_scatter() {
        // 32-bit gather, scaled (uxtw #1).
        check(0x84BB1101, "ld1sh   {z1.s}, p4/z, [x8, z27.s, uxtw #0x1]");
        // 64-bit gather, plain 64-bit offset.
        check(0xC45B9A1F, "ld1sb   {z31.d}, p6/z, [x16, z27.d]");
        // vector-base gather (no offset).
        check(0x8420C000, "ld1b    {z0.s}, p0/z, [z0.s]");
        // scatter: vector base + scalar, and vector + immediate.
        check(0xE44A2137, "stnt1b  {z23.s}, p0, [z9.s, x10]");
        check(0xE456A5CC, "st1b    {z12.d}, p1, [z14.d, #0x16]");
    }

    #[test]
    fn prefetch_forms() {
        // PRF gather: byte size shows no `#amt`; dword shows the scale.
        check(0xC4679D42, "prfb    pldl2keep, p7, [x10, z7.d]");
        check(0xC4321D80, "prfb    pldl1keep, p7, [x12, z18.d, uxtw]");
    }

    #[test]
    fn never_panics_on_sve_mem_space() {
        // Exhaustively exercise the SVE memory quadrant prefixes for
        // panic-freedom across the full low 16 bits.
        let mut buf = [0u8; 128];
        for hi in [0x84u32, 0x85, 0xa4, 0xa5, 0xc4, 0xc5, 0xe4, 0xe5] {
            for low in 0u32..=0xffff {
                let word = (hi << 24) | (low << 8) | (low & 0xff);
                let _ = render(word, &mut buf);
            }
        }
    }
}
