//! Inverse of [`crate::decode::sve::sve_mem`] — SVE/SVE2 loads / stores / prf.
//!
//! The decoder maps `(mnemonic, addressing-form) -> Code`; this inverts that and
//! reconstructs the raw `msz` / `dtype` / `op` / form fields from the
//! instruction's [`Operand`]s. Each [`Code`] already pins the mnemonic + form, so
//! the field math is a direct (if voluminous) inversion of `sve_mem`.

use super::{fld, p};
use crate::encode::EncodeError;
use crate::enums::{ExtendType, VectorArrangement as VA};
use crate::instruction::Instruction;
use crate::mnemonic::{Code, Mnemonic};
use crate::operand::{Operand, SveMemMode};
use crate::register::RegClass;

use Code::*;

/// `true` for every memory SVE [`Code`].
pub(super) fn is_mem(code: Code) -> bool {
    // The memory codes are a large contiguous run; classify by trying the
    // `(mnemonic, form)` decomposition.
    decompose(code).is_some() || is_qword(code)
}

/// Encode a memory SVE instruction.
pub(super) fn enc(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
    if is_qword(code) {
        return Ok(Some(enc_qword(insn, code)?));
    }
    let Some((m, form)) = decompose(code) else {
        return Ok(None);
    };
    // LDR/STR (vector/predicate register) are special.
    match code {
        SveLdrZ | SveStrZ | SveLdrP | SveStrP => return Ok(Some(enc_ldr_str(insn, code)?)),
        _ => {}
    }
    Ok(Some(enc_mem(insn, m, form, code)?))
}

// ---------------------------------------------------------------------------
// The (mnemonic, form) decomposition of a memory Code.
// ---------------------------------------------------------------------------

/// Addressing form (mirrors the decoder's private `Form`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Form {
    Imm,
    Ss,
    G32,
    G64,
    Vi,
    Vs,
    /// LDR/STR register forms (handled separately).
    Reg,
}

/// Decompose a memory [`Code`] into `(mnemonic, form)`, or `None` if not memory.
fn decompose(code: Code) -> Option<(Mnemonic, Form)> {
    use Form::*;
    use Mnemonic as M;
    Some(match code {
        SveLd1bImm => (M::Ld1b, Imm), SveLd1bSs => (M::Ld1b, Ss), SveLd1bG32 => (M::Ld1b, G32), SveLd1bG64 => (M::Ld1b, G64), SveLd1bVi => (M::Ld1b, Vi),
        SveLd1hImm => (M::Ld1h, Imm), SveLd1hSs => (M::Ld1h, Ss), SveLd1hG32 => (M::Ld1h, G32), SveLd1hG64 => (M::Ld1h, G64), SveLd1hVi => (M::Ld1h, Vi),
        SveLd1wImm => (M::Ld1w, Imm), SveLd1wSs => (M::Ld1w, Ss), SveLd1wG32 => (M::Ld1w, G32), SveLd1wG64 => (M::Ld1w, G64), SveLd1wVi => (M::Ld1w, Vi),
        SveLd1dImm => (M::Ld1d, Imm), SveLd1dSs => (M::Ld1d, Ss), SveLd1dG32 => (M::Ld1d, G32), SveLd1dG64 => (M::Ld1d, G64), SveLd1dVi => (M::Ld1d, Vi),
        SveLd1sbImm => (M::Ld1sb, Imm), SveLd1sbSs => (M::Ld1sb, Ss), SveLd1sbG32 => (M::Ld1sb, G32), SveLd1sbG64 => (M::Ld1sb, G64), SveLd1sbVi => (M::Ld1sb, Vi),
        SveLd1shImm => (M::Ld1sh, Imm), SveLd1shSs => (M::Ld1sh, Ss), SveLd1shG32 => (M::Ld1sh, G32), SveLd1shG64 => (M::Ld1sh, G64), SveLd1shVi => (M::Ld1sh, Vi),
        SveLd1swImm => (M::Ld1sw, Imm), SveLd1swSs => (M::Ld1sw, Ss), SveLd1swG32 => (M::Ld1sw, G32), SveLd1swG64 => (M::Ld1sw, G64), SveLd1swVi => (M::Ld1sw, Vi),
        SveLdff1bSs => (M::Ldff1b, Ss), SveLdff1bG32 => (M::Ldff1b, G32), SveLdff1bG64 => (M::Ldff1b, G64), SveLdff1bVi => (M::Ldff1b, Vi),
        SveLdff1hSs => (M::Ldff1h, Ss), SveLdff1hG32 => (M::Ldff1h, G32), SveLdff1hG64 => (M::Ldff1h, G64), SveLdff1hVi => (M::Ldff1h, Vi),
        SveLdff1wSs => (M::Ldff1w, Ss), SveLdff1wG32 => (M::Ldff1w, G32), SveLdff1wG64 => (M::Ldff1w, G64), SveLdff1wVi => (M::Ldff1w, Vi),
        SveLdff1dSs => (M::Ldff1d, Ss), SveLdff1dG32 => (M::Ldff1d, G32), SveLdff1dG64 => (M::Ldff1d, G64), SveLdff1dVi => (M::Ldff1d, Vi),
        SveLdff1sbSs => (M::Ldff1sb, Ss), SveLdff1sbG32 => (M::Ldff1sb, G32), SveLdff1sbG64 => (M::Ldff1sb, G64), SveLdff1sbVi => (M::Ldff1sb, Vi),
        SveLdff1shSs => (M::Ldff1sh, Ss), SveLdff1shG32 => (M::Ldff1sh, G32), SveLdff1shG64 => (M::Ldff1sh, G64), SveLdff1shVi => (M::Ldff1sh, Vi),
        SveLdff1swSs => (M::Ldff1sw, Ss), SveLdff1swG32 => (M::Ldff1sw, G32), SveLdff1swG64 => (M::Ldff1sw, G64), SveLdff1swVi => (M::Ldff1sw, Vi),
        SveLdnf1bImm => (M::Ldnf1b, Imm), SveLdnf1hImm => (M::Ldnf1h, Imm), SveLdnf1wImm => (M::Ldnf1w, Imm), SveLdnf1dImm => (M::Ldnf1d, Imm),
        SveLdnf1sbImm => (M::Ldnf1sb, Imm), SveLdnf1shImm => (M::Ldnf1sh, Imm), SveLdnf1swImm => (M::Ldnf1sw, Imm),
        SveLdnt1bImm => (M::Ldnt1b, Imm), SveLdnt1bSs => (M::Ldnt1b, Ss), SveLdnt1bVs => (M::Ldnt1b, Vs),
        SveLdnt1hImm => (M::Ldnt1h, Imm), SveLdnt1hSs => (M::Ldnt1h, Ss), SveLdnt1hVs => (M::Ldnt1h, Vs),
        SveLdnt1wImm => (M::Ldnt1w, Imm), SveLdnt1wSs => (M::Ldnt1w, Ss), SveLdnt1wVs => (M::Ldnt1w, Vs),
        SveLdnt1dImm => (M::Ldnt1d, Imm), SveLdnt1dSs => (M::Ldnt1d, Ss), SveLdnt1dVs => (M::Ldnt1d, Vs),
        SveLdnt1sbImm => (M::Ldnt1sb, Imm), SveLdnt1sbSs => (M::Ldnt1sb, Ss), SveLdnt1sbVs => (M::Ldnt1sb, Vs),
        SveLdnt1shImm => (M::Ldnt1sh, Imm), SveLdnt1shSs => (M::Ldnt1sh, Ss), SveLdnt1shVs => (M::Ldnt1sh, Vs),
        SveLdnt1swImm => (M::Ldnt1sw, Imm), SveLdnt1swSs => (M::Ldnt1sw, Ss), SveLdnt1swVs => (M::Ldnt1sw, Vs),
        SveLd1rbImm => (M::Ld1rb, Imm), SveLd1rhImm => (M::Ld1rh, Imm), SveLd1rwImm => (M::Ld1rw, Imm), SveLd1rdImm => (M::Ld1rd, Imm),
        SveLd1rsbImm => (M::Ld1rsb, Imm), SveLd1rshImm => (M::Ld1rsh, Imm), SveLd1rswImm => (M::Ld1rsw, Imm),
        SveLd1rqbImm => (M::Ld1rqb, Imm), SveLd1rqbSs => (M::Ld1rqb, Ss),
        SveLd1rqhImm => (M::Ld1rqh, Imm), SveLd1rqhSs => (M::Ld1rqh, Ss),
        SveLd1rqwImm => (M::Ld1rqw, Imm), SveLd1rqwSs => (M::Ld1rqw, Ss),
        SveLd1rqdImm => (M::Ld1rqd, Imm), SveLd1rqdSs => (M::Ld1rqd, Ss),
        SveLd1robImm => (M::Ld1rob, Imm), SveLd1robSs => (M::Ld1rob, Ss),
        SveLd1rohImm => (M::Ld1roh, Imm), SveLd1rohSs => (M::Ld1roh, Ss),
        SveLd1rowImm => (M::Ld1row, Imm), SveLd1rowSs => (M::Ld1row, Ss),
        SveLd1rodImm => (M::Ld1rod, Imm), SveLd1rodSs => (M::Ld1rod, Ss),
        SveLd2bImm => (M::Ld2b, Imm), SveLd2bSs => (M::Ld2b, Ss), SveLd2hImm => (M::Ld2h, Imm), SveLd2hSs => (M::Ld2h, Ss),
        SveLd2wImm => (M::Ld2w, Imm), SveLd2wSs => (M::Ld2w, Ss), SveLd2dImm => (M::Ld2d, Imm), SveLd2dSs => (M::Ld2d, Ss),
        SveLd3bImm => (M::Ld3b, Imm), SveLd3bSs => (M::Ld3b, Ss), SveLd3hImm => (M::Ld3h, Imm), SveLd3hSs => (M::Ld3h, Ss),
        SveLd3wImm => (M::Ld3w, Imm), SveLd3wSs => (M::Ld3w, Ss), SveLd3dImm => (M::Ld3d, Imm), SveLd3dSs => (M::Ld3d, Ss),
        SveLd4bImm => (M::Ld4b, Imm), SveLd4bSs => (M::Ld4b, Ss), SveLd4hImm => (M::Ld4h, Imm), SveLd4hSs => (M::Ld4h, Ss),
        SveLd4wImm => (M::Ld4w, Imm), SveLd4wSs => (M::Ld4w, Ss), SveLd4dImm => (M::Ld4d, Imm), SveLd4dSs => (M::Ld4d, Ss),
        SveSt1bImm => (M::St1b, Imm), SveSt1bSs => (M::St1b, Ss), SveSt1bG32 => (M::St1b, G32), SveSt1bG64 => (M::St1b, G64), SveSt1bVi => (M::St1b, Vi),
        SveSt1hImm => (M::St1h, Imm), SveSt1hSs => (M::St1h, Ss), SveSt1hG32 => (M::St1h, G32), SveSt1hG64 => (M::St1h, G64), SveSt1hVi => (M::St1h, Vi),
        SveSt1wImm => (M::St1w, Imm), SveSt1wSs => (M::St1w, Ss), SveSt1wG32 => (M::St1w, G32), SveSt1wG64 => (M::St1w, G64), SveSt1wVi => (M::St1w, Vi),
        SveSt1dImm => (M::St1d, Imm), SveSt1dSs => (M::St1d, Ss), SveSt1dG32 => (M::St1d, G32), SveSt1dG64 => (M::St1d, G64), SveSt1dVi => (M::St1d, Vi),
        SveSt2bImm => (M::St2b, Imm), SveSt2bSs => (M::St2b, Ss), SveSt2hImm => (M::St2h, Imm), SveSt2hSs => (M::St2h, Ss),
        SveSt2wImm => (M::St2w, Imm), SveSt2wSs => (M::St2w, Ss), SveSt2dImm => (M::St2d, Imm), SveSt2dSs => (M::St2d, Ss),
        SveSt3bImm => (M::St3b, Imm), SveSt3bSs => (M::St3b, Ss), SveSt3hImm => (M::St3h, Imm), SveSt3hSs => (M::St3h, Ss),
        SveSt3wImm => (M::St3w, Imm), SveSt3wSs => (M::St3w, Ss), SveSt3dImm => (M::St3d, Imm), SveSt3dSs => (M::St3d, Ss),
        SveSt4bImm => (M::St4b, Imm), SveSt4bSs => (M::St4b, Ss), SveSt4hImm => (M::St4h, Imm), SveSt4hSs => (M::St4h, Ss),
        SveSt4wImm => (M::St4w, Imm), SveSt4wSs => (M::St4w, Ss), SveSt4dImm => (M::St4d, Imm), SveSt4dSs => (M::St4d, Ss),
        SveStnt1bImm => (M::Stnt1b, Imm), SveStnt1bSs => (M::Stnt1b, Ss), SveStnt1bVs => (M::Stnt1b, Vs),
        SveStnt1hImm => (M::Stnt1h, Imm), SveStnt1hSs => (M::Stnt1h, Ss), SveStnt1hVs => (M::Stnt1h, Vs),
        SveStnt1wImm => (M::Stnt1w, Imm), SveStnt1wSs => (M::Stnt1w, Ss), SveStnt1wVs => (M::Stnt1w, Vs),
        SveStnt1dImm => (M::Stnt1d, Imm), SveStnt1dSs => (M::Stnt1d, Ss), SveStnt1dVs => (M::Stnt1d, Vs),
        SvePrfbImm => (M::Prfb, Imm), SvePrfbSs => (M::Prfb, Ss), SvePrfbG32 => (M::Prfb, G32), SvePrfbG64 => (M::Prfb, G64), SvePrfbVi => (M::Prfb, Vi),
        SvePrfhImm => (M::Prfh, Imm), SvePrfhSs => (M::Prfh, Ss), SvePrfhG32 => (M::Prfh, G32), SvePrfhG64 => (M::Prfh, G64), SvePrfhVi => (M::Prfh, Vi),
        SvePrfwImm => (M::Prfw, Imm), SvePrfwSs => (M::Prfw, Ss), SvePrfwG32 => (M::Prfw, G32), SvePrfwG64 => (M::Prfw, G64), SvePrfwVi => (M::Prfw, Vi),
        SvePrfdImm => (M::Prfd, Imm), SvePrfdSs => (M::Prfd, Ss), SvePrfdG32 => (M::Prfd, G32), SvePrfdG64 => (M::Prfd, G64), SvePrfdVi => (M::Prfd, Vi),
        SveLdrZ | SveStrZ | SveLdrP | SveStrP => (M::Ldr, Form::Reg),
        _ => return None,
    })
}

// (the rest of the encoder is in mem_enc.rs)
include!("mem_enc.rs");
