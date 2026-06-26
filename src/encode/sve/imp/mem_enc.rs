// Included into `mem.rs`. The per-(mnemonic, form) field reconstruction.

// ---------------------------------------------------------------------------
// Operand readers.
// ---------------------------------------------------------------------------

/// The first vector register of the data list `{Zt...}` (operand 0).
fn zt(insn: &Instruction) -> Result<u32, EncodeError> {
    match insn.op(0) {
        Operand::MultiReg { regs, .. } => Ok(regs[0].number() as u32),
        Operand::Reg { reg, .. } if reg.class() == RegClass::Sve => Ok(reg.number() as u32),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The data list's element arrangement.
fn zt_arr(insn: &Instruction) -> Result<VA, EncodeError> {
    match insn.op(0) {
        Operand::MultiReg { arr: Some(a), .. } => Ok(a),
        Operand::Reg { arr: Some(a), .. } => Ok(a),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The governing predicate `Pg` (operand 1).
fn pg(insn: &Instruction) -> Result<u32, EncodeError> {
    p(insn, 1)
}

/// The prefetch operation code (operand 0), the inverse of `prefetch_op_sve`.
fn prf_op(insn: &Instruction) -> Result<u32, EncodeError> {
    match insn.op(0) {
        Operand::SysOp(tok) => prf_token_code(tok.name()),
        Operand::ImmUnsigned(v) => Ok(v as u32 & 0xf),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Map a prefetch token name back to its 4-bit code.
fn prf_token_code(name: &str) -> Result<u32, EncodeError> {
    Ok(match name {
        "pldl1keep" => 0b0000,
        "pldl1strm" => 0b0001,
        "pldl2keep" => 0b0010,
        "pldl2strm" => 0b0011,
        "pldl3keep" => 0b0100,
        "pldl3strm" => 0b0101,
        "pstl1keep" => 0b1000,
        "pstl1strm" => 0b1001,
        "pstl2keep" => 0b1010,
        "pstl2strm" => 0b1011,
        "pstl3keep" => 0b1100,
        "pstl3strm" => 0b1101,
        _ => return Err(EncodeError::InvalidOperand),
    })
}

/// The memory operand of a load/store: always the last operand.
fn addr(insn: &Instruction) -> Operand {
    insn.op(insn.op_count() - 1)
}

/// Read a scalar-base `[Xn{, #imm, MUL VL}]` (ScalarImmMulVl): returns `(rn, imm)`.
fn read_mulvl(insn: &Instruction) -> Result<(u32, i32), EncodeError> {
    match addr(insn) {
        Operand::SveMem {
            base, imm, mode: SveMemMode::ScalarImmMulVl, ..
        } => Ok((base.number() as u32, imm)),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Read a scalar-base plain-imm `[Xn{, #imm}]` (ScalarImm / ScalarImmDec).
fn read_imm_plain(insn: &Instruction) -> Result<(u32, i32), EncodeError> {
    match addr(insn) {
        Operand::SveMem {
            base,
            imm,
            mode: SveMemMode::ScalarImm | SveMemMode::ScalarImmDec,
            ..
        } => Ok((base.number() as u32, imm)),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Read a scalar+scalar `[Xn, Xm{, lsl #amt}]` (MemExt): returns `(rn, rm)`.
fn read_ss(insn: &Instruction) -> Result<(u32, u32), EncodeError> {
    match addr(insn) {
        Operand::MemExt { base, index, .. } => Ok((base.number() as u32, index.number() as u32)),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Read a scalar+vector gather/scatter `[Xn, Zm.T{, mod #amt}]` (ScalarVec):
/// returns `(rn, zm, ext, amount)`.
fn read_xz(insn: &Instruction) -> Result<(u32, u32, ExtendType, u8), EncodeError> {
    match addr(insn) {
        Operand::SveMem {
            base,
            offset,
            extend,
            amount,
            mode: SveMemMode::ScalarVec,
            ..
        } => Ok((base.number() as u32, offset.number() as u32, extend, amount)),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Read a vector+imm `[Zn.T{, #imm}]` (VecImm): returns `(zn, imm)`.
fn read_vi(insn: &Instruction) -> Result<(u32, i32), EncodeError> {
    match addr(insn) {
        Operand::SveMem {
            base, imm, mode: SveMemMode::VecImm, ..
        } => Ok((base.number() as u32, imm)),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Read a vector+scalar `[Zn.T, Xm]` (VecScalar): returns `(zn, rm)`.
fn read_vs(insn: &Instruction) -> Result<(u32, u32), EncodeError> {
    match addr(insn) {
        Operand::SveMem {
            base, offset, mode: SveMemMode::VecScalar, ..
        } => Ok((base.number() as u32, offset.number() as u32)),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Pack a scalar+scalar shift bit `S` (`amt != 0 -> 0x80|amt`). The decoder
/// passes `lsl = mem size`, and the formatter shows `lsl` iff `amt != 0`; we
/// recover the original `Rm`-scale from the mnemonic's mem size, so the shift
/// byte itself is not needed for the word (the `Rm` field is unscaled).
#[inline]
fn _unused_ss_shift() {}

// ---------------------------------------------------------------------------
// Memory-size helpers.
// ---------------------------------------------------------------------------

/// The memory access size `msz` (0=byte .. 3=dword) for a load/store mnemonic.
fn msz_of(m: Mnemonic) -> Result<u32, EncodeError> {
    use Mnemonic as M;
    Ok(match m {
        M::Ld1b | M::Ld1sb | M::Ldff1b | M::Ldff1sb | M::Ldnf1b | M::Ldnf1sb | M::Ldnt1b
        | M::Ldnt1sb | M::St1b | M::Stnt1b | M::Prfb | M::Ld1rqb | M::Ld1rob | M::Ld2b | M::Ld3b
        | M::Ld4b | M::St2b | M::St3b | M::St4b | M::Ld1rb | M::Ld1rsb => 0,
        M::Ld1h | M::Ld1sh | M::Ldff1h | M::Ldff1sh | M::Ldnf1h | M::Ldnf1sh | M::Ldnt1h
        | M::Ldnt1sh | M::St1h | M::Stnt1h | M::Prfh | M::Ld1rqh | M::Ld1roh | M::Ld2h | M::Ld3h
        | M::Ld4h | M::St2h | M::St3h | M::St4h | M::Ld1rh | M::Ld1rsh => 1,
        M::Ld1w | M::Ld1sw | M::Ldff1w | M::Ldff1sw | M::Ldnf1w | M::Ldnf1sw | M::Ldnt1w
        | M::Ldnt1sw | M::St1w | M::Stnt1w | M::Prfw | M::Ld1rqw | M::Ld1row | M::Ld2w | M::Ld3w
        | M::Ld4w | M::St2w | M::St3w | M::St4w | M::Ld1rw | M::Ld1rsw => 2,
        M::Ld1d | M::Ldff1d | M::Ldnf1d | M::Ldnt1d | M::St1d | M::Stnt1d | M::Prfd | M::Ld1rqd
        | M::Ld1rod | M::Ld2d | M::Ld3d | M::Ld4d | M::St2d | M::St3d | M::St4d | M::Ld1rd => 3,
        _ => return Err(EncodeError::InvalidOperand),
    })
}

/// `true` if the mnemonic is a signed-element load.
fn is_signed(m: Mnemonic) -> bool {
    use Mnemonic as M;
    matches!(
        m,
        M::Ld1sb | M::Ld1sh | M::Ld1sw | M::Ldff1sb | M::Ldff1sh | M::Ldff1sw | M::Ldnf1sb
            | M::Ldnf1sh | M::Ldnf1sw | M::Ldnt1sb | M::Ldnt1sh | M::Ldnt1sw
    )
}

/// `true` if the mnemonic is a first-fault load.
fn is_ff(m: Mnemonic) -> bool {
    use Mnemonic as M;
    matches!(m, M::Ldff1b | M::Ldff1h | M::Ldff1w | M::Ldff1d | M::Ldff1sb | M::Ldff1sh | M::Ldff1sw)
}

/// Number of structured registers for a struct mnemonic (`Ld2*`=2 etc.), else 1.
fn nreg_of(m: Mnemonic) -> u8 {
    use Mnemonic as M;
    match m {
        M::Ld2b | M::Ld2h | M::Ld2w | M::Ld2d | M::St2b | M::St2h | M::St2w | M::St2d => 2,
        M::Ld3b | M::Ld3h | M::Ld3w | M::Ld3d | M::St3b | M::St3h | M::St3w | M::St3d => 3,
        M::Ld4b | M::Ld4h | M::Ld4w | M::Ld4d | M::St4b | M::St4h | M::St4w | M::St4d => 4,
        _ => 1,
    }
}

/// The `dtype` (`<24:21>`) for a contiguous load `(mnemonic, element arr)`.
fn load_dtype_inv(m: Mnemonic, a: VA) -> Result<u32, EncodeError> {
    use Mnemonic as M;
    use VA::{Sb, Sd, Sh, Ss};
    Ok(match (m, a) {
        (M::Ld1b, Sb) => 0, (M::Ld1b, Sh) => 1, (M::Ld1b, Ss) => 2, (M::Ld1b, Sd) => 3,
        (M::Ld1sw, Sd) => 4, (M::Ld1h, Sh) => 5, (M::Ld1h, Ss) => 6, (M::Ld1h, Sd) => 7,
        (M::Ld1sh, Sd) => 8, (M::Ld1sh, Ss) => 9, (M::Ld1w, Ss) => 10, (M::Ld1w, Sd) => 11,
        (M::Ld1sb, Sd) => 12, (M::Ld1sb, Ss) => 13, (M::Ld1sb, Sh) => 14, (M::Ld1d, Sd) => 15,
        // first-fault / non-fault share the same dtype table.
        (M::Ldff1b, Sb) => 0, (M::Ldff1b, Sh) => 1, (M::Ldff1b, Ss) => 2, (M::Ldff1b, Sd) => 3,
        (M::Ldff1sw, Sd) => 4, (M::Ldff1h, Sh) => 5, (M::Ldff1h, Ss) => 6, (M::Ldff1h, Sd) => 7,
        (M::Ldff1sh, Sd) => 8, (M::Ldff1sh, Ss) => 9, (M::Ldff1w, Ss) => 10, (M::Ldff1w, Sd) => 11,
        (M::Ldff1sb, Sd) => 12, (M::Ldff1sb, Ss) => 13, (M::Ldff1sb, Sh) => 14, (M::Ldff1d, Sd) => 15,
        (M::Ldnf1b, Sb) => 0, (M::Ldnf1b, Sh) => 1, (M::Ldnf1b, Ss) => 2, (M::Ldnf1b, Sd) => 3,
        (M::Ldnf1sw, Sd) => 4, (M::Ldnf1h, Sh) => 5, (M::Ldnf1h, Ss) => 6, (M::Ldnf1h, Sd) => 7,
        (M::Ldnf1sh, Sd) => 8, (M::Ldnf1sh, Ss) => 9, (M::Ldnf1w, Ss) => 10, (M::Ldnf1w, Sd) => 11,
        (M::Ldnf1sb, Sd) => 12, (M::Ldnf1sb, Ss) => 13, (M::Ldnf1sb, Sh) => 14, (M::Ldnf1d, Sd) => 15,
        _ => return Err(EncodeError::InvalidOperand),
    })
}

/// The `dtype` (`<24:21>`) for a contiguous store `(mnemonic, element arr)`.
fn store_dtype_inv(m: Mnemonic, a: VA) -> Result<u32, EncodeError> {
    use Mnemonic as M;
    use VA::{Sb, Sd, Sh, Ss};
    Ok(match (m, a) {
        (M::St1b, Sb) => 0, (M::St1b, Sh) => 1, (M::St1b, Ss) => 2, (M::St1b, Sd) => 3,
        (M::St1h, Sh) => 5, (M::St1h, Ss) => 6, (M::St1h, Sd) => 7,
        (M::St1w, Ss) => 10, (M::St1w, Sd) => 11, (M::St1d, Sd) => 15,
        _ => return Err(EncodeError::InvalidOperand),
    })
}

// ---------------------------------------------------------------------------
// Top-level memory word build.
// ---------------------------------------------------------------------------

/// Reconstruct the word for a (decomposed) memory instruction.
fn enc_mem(insn: &Instruction, m: Mnemonic, form: Form, code: Code) -> Result<u32, EncodeError> {
    use Mnemonic as M;
    // Route by mnemonic family.
    match m {
        // ---- LD1R* broadcast (scalar+imm, hex offset) ----
        M::Ld1rb | M::Ld1rh | M::Ld1rw | M::Ld1rd | M::Ld1rsb | M::Ld1rsh | M::Ld1rsw => {
            enc_ld1r(insn, m)
        }
        // ---- LD1RQ / LD1RO (replicating) ----
        M::Ld1rqb | M::Ld1rqh | M::Ld1rqw | M::Ld1rqd => enc_ld1rq(insn, m, false),
        M::Ld1rob | M::Ld1roh | M::Ld1row | M::Ld1rod => enc_ld1rq(insn, m, true),
        // ---- contiguous / first-fault / non-fault single-register loads ----
        M::Ld1b | M::Ld1h | M::Ld1w | M::Ld1d | M::Ld1sb | M::Ld1sh | M::Ld1sw | M::Ldff1b
        | M::Ldff1h | M::Ldff1w | M::Ldff1d | M::Ldff1sb | M::Ldff1sh | M::Ldff1sw | M::Ldnf1b
        | M::Ldnf1h | M::Ldnf1w | M::Ldnf1d | M::Ldnf1sb | M::Ldnf1sh | M::Ldnf1sw => {
            enc_load_single(insn, m, form, code)
        }
        // ---- contiguous single-register stores ----
        M::St1b | M::St1h | M::St1w | M::St1d => enc_store_single(insn, m, form),
        // ---- structured loads/stores ----
        M::Ld2b | M::Ld2h | M::Ld2w | M::Ld2d | M::Ld3b | M::Ld3h | M::Ld3w | M::Ld3d | M::Ld4b
        | M::Ld4h | M::Ld4w | M::Ld4d => enc_struct(insn, m, form, false),
        M::St2b | M::St2h | M::St2w | M::St2d | M::St3b | M::St3h | M::St3w | M::St3d | M::St4b
        | M::St4h | M::St4w | M::St4d => enc_struct(insn, m, form, true),
        // ---- LDNT1 / STNT1 ----
        M::Ldnt1b | M::Ldnt1h | M::Ldnt1w | M::Ldnt1d | M::Ldnt1sb | M::Ldnt1sh | M::Ldnt1sw => {
            enc_ldnt1(insn, m, form)
        }
        M::Stnt1b | M::Stnt1h | M::Stnt1w | M::Stnt1d => enc_stnt1(insn, m, form),
        // ---- prefetch ----
        M::Prfb | M::Prfh | M::Prfw | M::Prfd => enc_prf(insn, m, form),
        _ => Err(EncodeError::Unsupported),
    }
}

// ---------------------------------------------------------------------------
// FEAT_SVE2p1 quadword (`.q`) load/store encode (inverse of `decode_qword`).
// ---------------------------------------------------------------------------

/// `true` if `code` is a quadword load/store form handled by [`enc_qword`].
pub(super) fn is_qword(code: Code) -> bool {
    matches!(
        code,
        Code::SveLd1qG
            | Code::SveSt1qS
            | Code::SveLd2qSs | Code::SveLd2qImm
            | Code::SveLd3qSs | Code::SveLd3qImm
            | Code::SveLd4qSs | Code::SveLd4qImm
            | Code::SveSt2qSs | Code::SveSt2qImm
            | Code::SveSt3qSs | Code::SveSt3qImm
            | Code::SveSt4qSs | Code::SveSt4qImm
    )
}

/// Reconstruct the word for a quadword load/store form from its operands.
fn enc_qword(insn: &Instruction, code: Code) -> Result<u32, EncodeError> {
    let zt = zt(insn)?;
    let pgv = pg(insn)?;
    match code {
        // LD1Q gather: 11000100 000 Rm 101 Pg Zn Zt.
        Code::SveLd1qG => {
            let (zn, rm) = read_q_gather(insn)?;
            Ok(0xc400_0000 | fld(0b101, 13) | fld(rm, 16) | fld(pgv, 10) | fld(zn, 5) | zt)
        }
        // ST1Q scatter: 11100100 001 Rm 001 Pg Zn Zt.
        Code::SveSt1qS => {
            let (zn, rm) = read_q_gather(insn)?;
            Ok(0xe400_0000 | fld(0b001, 21) | fld(0b001, 13) | fld(rm, 16) | fld(pgv, 10) | fld(zn, 5) | zt)
        }
        // LD{2,3,4}Q: 1010010 nreg-1(24:23) 0 ...
        Code::SveLd2qSs | Code::SveLd3qSs | Code::SveLd4qSs
        | Code::SveLd2qImm | Code::SveLd3qImm | Code::SveLd4qImm => {
            enc_qword_struct(insn, code, false, zt, pgv)
        }
        // ST{2,3,4}Q: 1110010 nreg-1(24:22) ...
        Code::SveSt2qSs | Code::SveSt3qSs | Code::SveSt4qSs
        | Code::SveSt2qImm | Code::SveSt3qImm | Code::SveSt4qImm => {
            enc_qword_struct(insn, code, true, zt, pgv)
        }
        _ => Err(EncodeError::Unsupported),
    }
}

/// Encode a quadword structured `(nreg, store, ss/imm)` form.
fn enc_qword_struct(insn: &Instruction, code: Code, store: bool, zt: u32, pgv: u32) -> Result<u32, EncodeError> {
    let (nreg, ss) = qword_struct_params(code);
    let base = if store { 0xe400_0000 } else { 0xa400_0000 };
    let nfield = if store {
        // stores: nreg-1 in bits<24:22>.
        fld((nreg - 1) as u32, 22)
    } else {
        // loads: nreg-1 in bits<24:23>.
        fld((nreg - 1) as u32, 23)
    };
    if ss {
        let (rn, rm) = read_ss(insn)?;
        let sel = if store { 0b000 } else { 0b100 };
        Ok(base | nfield | fld(1, 21) | fld(rm, 16) | fld(sel, 13) | fld(pgv, 10) | fld(rn, 5) | zt)
    } else {
        let (rn, imm) = read_mulvl(insn)?;
        if imm % nreg as i32 != 0 {
            return Err(EncodeError::InvalidImmediate);
        }
        let i4 = imm / nreg as i32;
        if !(-8..=7).contains(&i4) {
            return Err(EncodeError::InvalidImmediate);
        }
        let sel = if store { 0b000 } else { 0b111 };
        let b20 = if store { 0 } else { 1 };
        Ok(base
            | nfield
            | fld(b20, 20)
            | fld((i4 as u32) & 0xf, 16)
            | fld(sel, 13)
            | fld(pgv, 10)
            | fld(rn, 5)
            | zt)
    }
}

/// `(nreg, is_ss)` for a quadword structured [`Code`].
fn qword_struct_params(code: Code) -> (u8, bool) {
    match code {
        Code::SveLd2qSs | Code::SveSt2qSs => (2, true),
        Code::SveLd3qSs | Code::SveSt3qSs => (3, true),
        Code::SveLd4qSs | Code::SveSt4qSs => (4, true),
        Code::SveLd2qImm | Code::SveSt2qImm => (2, false),
        Code::SveLd3qImm | Code::SveSt3qImm => (3, false),
        _ => (4, false),
    }
}

/// Read a quadword gather/scatter address `[Zn.D{, Xm}]`: returns `(zn, rm)`,
/// where `rm == 31` (xzr) for the bare `[Zn.D]` form (a `VecImm` operand).
fn read_q_gather(insn: &Instruction) -> Result<(u32, u32), EncodeError> {
    match addr(insn) {
        Operand::SveMem { base, mode: SveMemMode::VecImm, .. } => Ok((base.number() as u32, 0b11111)),
        Operand::SveMem { base, offset, mode: SveMemMode::VecScalar, .. } => {
            Ok((base.number() as u32, offset.number() as u32))
        }
        _ => Err(EncodeError::InvalidOperand),
    }
}

include!("mem_forms.rs");
