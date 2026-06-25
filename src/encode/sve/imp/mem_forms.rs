// Included into `mem.rs` (via mem_enc.rs). Per-family field reconstruction.

/// Base for contiguous loads (`<31:25>=1010010`).
#[inline]
fn base_ld() -> u32 {
    fld(0b1010010, 25)
}
/// Base for contiguous stores (`<31:25>=1110010`).
#[inline]
fn base_st() -> u32 {
    fld(0b1110010, 25)
}
/// Base for the 32-bit gather/scatter quadrant (`<31:25>=1000010` load).
#[inline]
fn base_g32_ld() -> u32 {
    fld(0b1000010, 25)
}
/// Base for the 64-bit gather quadrant (`<31:25>=1100010` load).
#[inline]
fn base_g64_ld() -> u32 {
    fld(0b1100010, 25)
}

/// Signed 4-bit immediate field for a `, #imm, MUL VL` operand, after dividing
/// out the implicit `nreg` scale (the decoder multiplies struct imms by nreg).
fn imm4_field(imm: i32, scale: i32) -> Result<u32, EncodeError> {
    if scale == 0 || imm % scale != 0 {
        return Err(EncodeError::InvalidImmediate);
    }
    let v = imm / scale;
    if !(-8..=7).contains(&v) {
        return Err(EncodeError::InvalidImmediate);
    }
    Ok((v as u32) & 0xf)
}

/// Signed 6-bit immediate field.
fn imm6_field(imm: i32) -> Result<u32, EncodeError> {
    if !(-32..=31).contains(&imm) {
        return Err(EncodeError::InvalidImmediate);
    }
    Ok((imm as u32) & 0x3f)
}

/// Encode a contiguous / first-fault / non-fault single load.
fn enc_load_single(
    insn: &Instruction,
    m: Mnemonic,
    form: Form,
    _code: Code,
) -> Result<u32, EncodeError> {
    let a = zt_arr(insn)?;
    let dtype = load_dtype_inv(m, a)?;
    let zt = zt(insn)?;
    let pg = pg(insn)?;
    let base = base_ld() | fld(dtype, 21) | fld(pg, 10) | zt;
    match form {
        Form::Ss => {
            // scalar+scalar: op=2 (plain) / 3 (first-fault).
            let op = if is_ff(m) { 3 } else { 2 };
            let (rn, rm) = read_ss(insn)?;
            Ok(base | fld(op, 13) | fld(rm, 16) | fld(rn, 5))
        }
        Form::Imm => {
            // scalar+imm (MUL VL): op=5; non-fault sets <20>=1.
            let nf = matches!(
                m,
                Mnemonic::Ldnf1b
                    | Mnemonic::Ldnf1h
                    | Mnemonic::Ldnf1w
                    | Mnemonic::Ldnf1d
                    | Mnemonic::Ldnf1sb
                    | Mnemonic::Ldnf1sh
                    | Mnemonic::Ldnf1sw
            );
            let (rn, imm) = read_mulvl(insn)?;
            let i4 = imm4_field(imm, 1)?;
            Ok(base | fld(5, 13) | fld(u32::from(nf), 20) | fld(i4, 16) | fld(rn, 5))
        }
        Form::G32 | Form::G64 | Form::Vi => enc_gather(insn, m, form),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Encode a contiguous single store.
fn enc_store_single(insn: &Instruction, m: Mnemonic, form: Form) -> Result<u32, EncodeError> {
    match form {
        Form::G32 | Form::G64 | Form::Vi => return enc_scatter(insn, m, form),
        _ => {}
    }
    let a = zt_arr(insn)?;
    let dtype = store_dtype_inv(m, a)?;
    let zt = zt(insn)?;
    let pg = pg(insn)?;
    let base = base_st() | fld(dtype, 21) | fld(pg, 10) | zt;
    match form {
        Form::Ss => {
            let (rn, rm) = read_ss(insn)?;
            Ok(base | fld(2, 13) | fld(rm, 16) | fld(rn, 5))
        }
        Form::Imm => {
            let (rn, imm) = read_mulvl(insn)?;
            let i4 = imm4_field(imm, 1)?;
            // op=7, <20>=0 selects ST1 scalar+imm.
            Ok(base | fld(7, 13) | fld(i4, 16) | fld(rn, 5))
        }
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Encode a structured load/store.
fn enc_struct(
    insn: &Instruction,
    m: Mnemonic,
    form: Form,
    store: bool,
) -> Result<u32, EncodeError> {
    let msz = msz_of(m)?;
    let nreg = nreg_of(m);
    let zt = zt(insn)?;
    let pg = pg(insn)?;
    // nr = nreg-1 encoded in <22:21> (b22<<1 | b21).
    let nr = (nreg - 1) as u32;
    let b22 = (nr >> 1) & 1;
    let b21 = nr & 1;
    let base = if store { base_st() } else { base_ld() };
    let base = base | fld(msz, 23) | fld(b22, 22) | fld(b21, 21) | fld(pg, 10) | zt;
    match form {
        Form::Ss => {
            let (rn, rm) = read_ss(insn)?;
            // loads use op=7? No: contiguous structured loads use op=6 (Ss) / 7 (Imm).
            let op = if store { 3 } else { 6 };
            Ok(base | fld(op, 13) | fld(rm, 16) | fld(rn, 5))
        }
        Form::Imm => {
            let (rn, imm) = read_mulvl(insn)?;
            let i4 = imm4_field(imm, nreg as i32)?;
            let op = 7;
            // stores set <20>=1 for the struct/stnt imm group.
            let extra = if store { fld(1, 20) } else { 0 };
            Ok(base | fld(op, 13) | extra | fld(i4, 16) | fld(rn, 5))
        }
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Encode LD1RQ / LD1RO.
fn enc_ld1rq(insn: &Instruction, m: Mnemonic, ro: bool) -> Result<u32, EncodeError> {
    let msz = msz_of(m)?;
    let zt = zt(insn)?;
    let pg = pg(insn)?;
    let b21 = if ro { 1 } else { 0 };
    let base = base_ld() | fld(msz, 23) | fld(b21, 21) | fld(pg, 10) | zt;
    // op0 = scalar+scalar, op1 = scalar+imm.
    match addr(insn) {
        Operand::MemExt { base: bn, index, .. } => {
            Ok(base | fld(0, 13) | fld(index.number() as u32, 16) | fld(bn.number() as u32, 5))
        }
        Operand::SveMem { base: bn, imm, mode, .. } => {
            let i4 = if matches!(mode, SveMemMode::ScalarImmDec) {
                // LD1RQ scales by 16.
                imm4_field(imm, 16)?
            } else {
                imm4_field(imm, 1)?
            };
            Ok(base | fld(1, 13) | fld(i4, 16) | fld(bn.number() as u32, 5))
        }
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Encode LD1R* broadcast (scalar+imm, hex offset). The immediate is `imm6 *
/// mem_size`; the encoding lives in the 32-bit gather quadrant region 1x op>=4.
fn enc_ld1r(insn: &Instruction, m: Mnemonic) -> Result<u32, EncodeError> {
    let a = zt_arr(insn)?;
    let key = ld1r_key(m, a)?;
    let key_msz = key >> 3; // <24:23>
    let op = key & 7; // <15:13>
    let zt = zt(insn)?;
    let pg = pg(insn)?;
    let (rn, imm) = read_imm_plain(insn)?;
    let scale = 1i32 << msz_of(m)?;
    if imm % scale != 0 {
        return Err(EncodeError::InvalidImmediate);
    }
    let imm6 = imm / scale;
    if !(0..=63).contains(&imm6) {
        return Err(EncodeError::InvalidImmediate);
    }
    // 0x84/0x85 region: <31:25>=1000010, <24:23>=key_msz, <22>=1, op=<15:13>.
    Ok(base_g32_ld()
        | fld(key_msz, 23)
        | fld(1, 22)
        | fld(op, 13)
        | fld(imm6 as u32, 16)
        | fld(pg, 10)
        | fld(rn, 5)
        | zt)
}

/// The decoder's `ld1r_entry` key for an `(mnemonic, element arr)` pair
/// (`key = (<24:23> << 3) | <15:13>`).
fn ld1r_key(m: Mnemonic, a: VA) -> Result<u32, EncodeError> {
    use Mnemonic as M;
    use VA::{Sb, Sd, Sh, Ss};
    Ok(match (m, a) {
        (M::Ld1rb, Sb) => 4, (M::Ld1rb, Sh) => 5, (M::Ld1rb, Ss) => 6, (M::Ld1rb, Sd) => 7,
        (M::Ld1rsw, Sd) => 12, (M::Ld1rh, Sh) => 13, (M::Ld1rh, Ss) => 14, (M::Ld1rh, Sd) => 15,
        (M::Ld1rsh, Sd) => 20, (M::Ld1rsh, Ss) => 21, (M::Ld1rw, Ss) => 22, (M::Ld1rw, Sd) => 23,
        (M::Ld1rsb, Sd) => 28, (M::Ld1rsb, Ss) => 29, (M::Ld1rsb, Sh) => 30, (M::Ld1rd, Sd) => 31,
        _ => return Err(EncodeError::InvalidOperand),
    })
}

/// Encode a gather load. The G32/G64 [`Code`] covers both scalar+vec and
/// vector+imm gathers, so dispatch on the addressing operand.
fn enc_gather(insn: &Instruction, m: Mnemonic, _form: Form) -> Result<u32, EncodeError> {
    let msz = msz_of(m)?;
    let signed = is_signed(m);
    let ff = is_ff(m);
    let zt = zt(insn)?;
    let pg = pg(insn)?;
    match addr(insn) {
        // vector+imm gather: [Zn.T{, #imm}].
        Operand::SveMem { base: zn, imm, arr, mode: SveMemMode::VecImm, .. } => {
            let is64 = matches!(arr, Some(VA::Sd));
            let imm5 = vi_imm5(imm, msz)?;
            let (op, _) = gather_vi_op(signed, ff);
            let zn = zn.number() as u32;
            if is64 {
                Ok(base_g64_ld() | fld(msz, 23) | fld(0, 22) | fld(1, 21) | fld(op, 13)
                    | fld(imm5, 16)
                    | fld(pg, 10)
                    | fld(zn, 5)
                    | zt)
            } else {
                Ok(base_g32_ld() | fld(msz, 23) | fld(1, 21) | fld(op, 13) | fld(imm5, 16)
                    | fld(pg, 10)
                    | fld(zn, 5)
                    | zt)
            }
        }
        // scalar+vec gather.
        Operand::SveMem {
            base: bn,
            offset: zm,
            arr,
            extend,
            amount,
            mode: SveMemMode::ScalarVec,
            ..
        } => {
            let rn = bn.number() as u32;
            let zm = zm.number() as u32;
            let is64 = matches!(arr, Some(VA::Sd));
            if is64 {
                enc_gather64_xz(zt, pg, rn, zm, extend, amount, msz, signed, ff)
            } else {
                let (b22, b21, op) = gather_g32_fields(extend, amount, msz, signed, ff)?;
                Ok(base_g32_ld() | fld(msz, 23) | fld(b22, 22) | fld(b21, 21) | fld(op, 13)
                    | fld(zm, 16)
                    | fld(pg, 10)
                    | fld(rn, 5)
                    | zt)
            }
        }
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The arrangement of the SVE-vector base/index in the memory operand.
fn zn_arr(insn: &Instruction) -> Result<VA, EncodeError> {
    match addr(insn) {
        Operand::SveMem { arr: Some(a), .. } => Ok(a),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// `(b22, b21, op)` for a 32-bit gather scalar+vec.
fn gather_g32_fields(
    ext: ExtendType,
    amount: u8,
    msz: u32,
    signed: bool,
    ff: bool,
) -> Result<(u32, u32, u32), EncodeError> {
    // xs: Sxtw->1, Uxtw->0. scaled: amount != 0xFF (and the decoder scaled bit).
    let xs = match ext {
        ExtendType::Sxtw => 1,
        ExtendType::Uxtw => 0,
        _ => return Err(EncodeError::InvalidOperand),
    };
    let scaled = amount != 0xFF; // present #amt means scaled (amt == msz)
    let b21 = u32::from(scaled);
    let b22 = xs;
    // op0-3: ff=op&1, signed=op<2 even. op = (signed?0:2) + (ff?1:0)... decoder:
    // ff = op==1||op==3 ; signed = op==0||op==1. So op: signed&!ff=0, signed&ff=1,
    // !signed&!ff=2, !signed&ff=3.
    let op = match (signed, ff) {
        (true, false) => 0,
        (true, true) => 1,
        (false, false) => 2,
        (false, true) => 3,
    };
    let _ = msz;
    Ok((b22, b21, op))
}

/// 64-bit gather scalar+vec dispatch (32-bit-unpacked op0-3 vs 64-bit op4-7).
#[allow(clippy::too_many_arguments)]
fn enc_gather64_xz(
    zt: u32,
    pg: u32,
    rn: u32,
    zm: u32,
    ext: ExtendType,
    amount: u8,
    msz: u32,
    signed: bool,
    ff: bool,
) -> Result<u32, EncodeError> {
    let scaled = amount != 0xFF;
    let b21 = u32::from(scaled);
    match ext {
        ExtendType::Uxtw | ExtendType::Sxtw => {
            // op0-3 region: xs=b22.
            let b22 = if matches!(ext, ExtendType::Sxtw) { 1 } else { 0 };
            let op = match (signed, ff) {
                (true, false) => 0,
                (true, true) => 1,
                (false, false) => 2,
                (false, true) => 3,
            };
            Ok(base_g64_ld()
                | fld(msz, 23)
                | fld(b22, 22)
                | fld(b21, 21)
                | fld(op, 13)
                | fld(zm, 16)
                | fld(pg, 10)
                | fld(rn, 5)
                | zt)
        }
        ExtendType::Uxtx => {
            // 64-bit packed offset (lsl / none): region 10/11 op4-7, b22=0.
            let op = match (signed, ff) {
                (true, false) => 4,
                (true, true) => 5,
                (false, false) => 6,
                (false, true) => 7,
            };
            Ok(base_g64_ld()
                | fld(msz, 23)
                | fld(1, 22)
                | fld(b21, 21)
                | fld(op, 13)
                | fld(zm, 16)
                | fld(pg, 10)
                | fld(rn, 5)
                | zt)
        }
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// `op` for a vector+imm gather, by (signed, ff).
fn gather_vi_op(signed: bool, ff: bool) -> (u32, u32) {
    let op = match (signed, ff) {
        (true, false) => 4,
        (true, true) => 5,
        (false, false) => 6,
        (false, true) => 7,
    };
    (op, 0)
}

/// The 5-bit vector+imm offset field (`imm = field * (1<<msz)`).
fn vi_imm5(imm: i32, msz: u32) -> Result<u32, EncodeError> {
    let scale = 1i32 << msz;
    if scale == 0 || imm % scale != 0 {
        return Err(EncodeError::InvalidImmediate);
    }
    let v = imm / scale;
    if !(0..=31).contains(&v) {
        return Err(EncodeError::InvalidImmediate);
    }
    Ok(v as u32)
}

/// Encode a scatter store. The G32/G64 [`Code`] is shared by the scalar+vec and
/// vector+imm scatter forms, so we dispatch on the actual addressing operand.
fn enc_scatter(insn: &Instruction, m: Mnemonic, _form: Form) -> Result<u32, EncodeError> {
    let msz = msz_of(m)?;
    let zt = zt(insn)?;
    let pg = pg(insn)?;
    let base = base_st() | fld(msz, 23) | fld(pg, 10) | zt;
    match addr(insn) {
        // ST1 vector+imm scatter: op5, b22=1, element b21 (.s->1, .d->0).
        Operand::SveMem { base: zn, imm, arr, mode: SveMemMode::VecImm, .. } => {
            let imm5 = vi_imm5(imm, msz)?;
            let b21 = if matches!(arr, Some(VA::Ss)) { 1 } else { 0 };
            Ok(base | fld(1, 22) | fld(b21, 21) | fld(5, 13) | fld(imm5, 16) | fld(zn.number() as u32, 5))
        }
        // ST1 scalar+vec scatter.
        Operand::SveMem {
            base: bn,
            offset: zm,
            arr,
            extend,
            amount,
            mode: SveMemMode::ScalarVec,
            ..
        } => {
            let rn = bn.number() as u32;
            let zm = zm.number() as u32;
            let scaled = amount != 0xFF;
            let b21 = u32::from(scaled);
            let s_elt = matches!(arr, Some(VA::Ss));
            match extend {
                ExtendType::Uxtx => {
                    // 64-bit [Xn, Zm.d{, lsl #msz}]: op5, b22=0.
                    Ok(base | fld(0, 22) | fld(b21, 21) | fld(5, 13) | fld(zm, 16) | fld(rn, 5))
                }
                ExtendType::Uxtw | ExtendType::Sxtw => {
                    // 32-bit-unpacked: op4 (uxtw) / op6 (sxtw); b22 = (.s element).
                    let b22 = u32::from(s_elt);
                    let op = if matches!(extend, ExtendType::Sxtw) { 6 } else { 4 };
                    Ok(base | fld(b22, 22) | fld(b21, 21) | fld(op, 13) | fld(zm, 16) | fld(rn, 5))
                }
                _ => Err(EncodeError::InvalidOperand),
            }
        }
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Encode LDNT1 (Imm / Ss / Vs).
fn enc_ldnt1(insn: &Instruction, m: Mnemonic, form: Form) -> Result<u32, EncodeError> {
    let msz = msz_of(m)?;
    let signed = is_signed(m);
    let zt = zt(insn)?;
    let pg = pg(insn)?;
    match form {
        Form::Imm => {
            // contiguous LDNT1 load: 0xa4/a5 region, op7, nr=0 (no <20> bit).
            let (rn, imm) = read_mulvl(insn)?;
            let i4 = imm4_field(imm, 1)?;
            Ok(base_ld() | fld(msz, 23) | fld(7, 13) | fld(i4, 16) | fld(pg, 10) | fld(rn, 5) | zt)
        }
        Form::Ss => {
            let (rn, rm) = read_ss(insn)?;
            Ok(base_ld() | fld(msz, 23) | fld(6, 13) | fld(rm, 16) | fld(pg, 10) | fld(rn, 5) | zt)
        }
        Form::Vs => {
            // vector base + scalar offset gather (SVE2). Region 00: 32-bit uses
            // op4(signed)/op5(unsigned); 64-bit uses op4(signed)/op6(unsigned).
            let (zn, rm) = read_vs(insn)?;
            let is64 = matches!(zn_arr(insn)?, VA::Sd);
            let op = if signed {
                4
            } else if is64 {
                6
            } else {
                5
            };
            let base = if is64 { base_g64_ld() } else { base_g32_ld() };
            Ok(base | fld(msz, 23) | fld(op, 13) | fld(rm, 16) | fld(pg, 10) | fld(zn, 5) | zt)
        }
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Encode STNT1 (Imm / Ss / Vs).
fn enc_stnt1(insn: &Instruction, m: Mnemonic, form: Form) -> Result<u32, EncodeError> {
    let msz = msz_of(m)?;
    let zt = zt(insn)?;
    let pg = pg(insn)?;
    match form {
        Form::Imm => {
            let (rn, imm) = read_mulvl(insn)?;
            let i4 = imm4_field(imm, 1)?;
            // op7, <20>=1, nr=0.
            Ok(base_st() | fld(msz, 23) | fld(7, 13) | fld(1, 20) | fld(i4, 16) | fld(pg, 10)
                | fld(rn, 5)
                | zt)
        }
        Form::Ss => {
            let (rn, rm) = read_ss(insn)?;
            Ok(base_st() | fld(msz, 23) | fld(3, 13) | fld(rm, 16) | fld(pg, 10) | fld(rn, 5) | zt)
        }
        Form::Vs => {
            // scatter vector base + scalar: op1, element b22 (.s->1, .d->0).
            let (zn, rm) = read_vs(insn)?;
            let b22 = if matches!(zn_arr(insn)?, VA::Ss) { 1 } else { 0 };
            Ok(base_st() | fld(msz, 23) | fld(b22, 22) | fld(1, 13) | fld(rm, 16) | fld(pg, 10)
                | fld(zn, 5)
                | zt)
        }
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// Encode a prefetch (Imm / Ss / G32 / G64 / Vi).
fn enc_prf(insn: &Instruction, m: Mnemonic, form: Form) -> Result<u32, EncodeError> {
    let sz = prf_sz(m);
    let prfop = prf_op(insn)?;
    let pg = pg(insn)?;
    match form {
        Form::Imm => {
            // contiguous PRF: 0x85 region msz==3, b22==1, op<4 = sz; imm6 MUL VL.
            let (rn, imm) = read_mulvl(insn)?;
            let i6 = imm6_field(imm)?;
            Ok(base_g32_ld()
                | fld(3, 23)
                | fld(1, 22)
                | fld(sz, 13)
                | fld(i6, 16)
                | fld(pg, 10)
                | fld(rn, 5)
                | prfop)
        }
        Form::Ss => {
            // 32-bit gather quadrant region 00 op6 scalar+scalar PRF.
            let (rn, rm) = read_ss(insn)?;
            Ok(base_g32_ld()
                | fld(sz, 23)
                | fld(6, 13)
                | fld(rm, 16)
                | fld(pg, 10)
                | fld(rn, 5)
                | prfop)
        }
        Form::Vi => enc_prf_vi(insn, m, sz, prfop, pg),
        Form::G32 => {
            let (rn, zm, ext, amount) = read_xz(insn)?;
            let b22 = if matches!(ext, ExtendType::Sxtw) { 1 } else { 0 };
            let scaled = amount != 0xFF;
            // PRF scalar+vec 32-bit: msz==0, b21=1, op<4=sz.
            let _ = scaled;
            Ok(base_g32_ld()
                | fld(0, 23)
                | fld(b22, 22)
                | fld(1, 21)
                | fld(sz, 13)
                | fld(zm, 16)
                | fld(pg, 10)
                | fld(rn, 5)
                | prfop)
        }
        Form::G64 => {
            let (rn, zm, ext, amount) = read_xz(insn)?;
            let _ = amount;
            match ext {
                ExtendType::Uxtw | ExtendType::Sxtw => {
                    let b22 = if matches!(ext, ExtendType::Sxtw) { 1 } else { 0 };
                    Ok(base_g64_ld()
                        | fld(0, 23)
                        | fld(b22, 22)
                        | fld(1, 21)
                        | fld(sz, 13)
                        | fld(zm, 16)
                        | fld(pg, 10)
                        | fld(rn, 5)
                        | prfop)
                }
                ExtendType::Uxtx => {
                    // 64-bit packed: op4-7 region 11, op=sz+4.
                    Ok(base_g64_ld()
                        | fld(0, 23)
                        | fld(1, 22)
                        | fld(1, 21)
                        | fld(sz + 4, 13)
                        | fld(zm, 16)
                        | fld(pg, 10)
                        | fld(rn, 5)
                        | prfop)
                }
                _ => Err(EncodeError::InvalidOperand),
            }
        }
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// PRF vector+imm.
fn enc_prf_vi(
    insn: &Instruction,
    _m: Mnemonic,
    sz: u32,
    prfop: u32,
    pg: u32,
) -> Result<u32, EncodeError> {
    let (zn, imm) = read_vi(insn)?;
    let is64 = matches!(zn_arr(insn)?, VA::Sd);
    if is64 {
        // 0xC4 region 00 op7: prfb msz==0 (imm raw) or prf msz>0 (imm scaled).
        if sz == 0 {
            let imm5 = if (0..=31).contains(&imm) { imm as u32 } else { return Err(EncodeError::InvalidImmediate) };
            Ok(base_g64_ld() | fld(0, 23) | fld(0, 22) | fld(0, 21) | fld(7, 13) | fld(imm5, 16)
                | fld(pg, 10)
                | fld(zn, 5)
                | prfop)
        } else {
            let imm5 = vi_imm5(imm, sz)?;
            Ok(base_g64_ld() | fld(sz, 23) | fld(0, 22) | fld(0, 21) | fld(7, 13) | fld(imm5, 16)
                | fld(pg, 10)
                | fld(zn, 5)
                | prfop)
        }
    } else {
        // 0x84 region 00 op7: imm scaled by msz.
        let imm5 = vi_imm5(imm, sz)?;
        Ok(base_g32_ld() | fld(sz, 23) | fld(7, 13) | fld(imm5, 16) | fld(pg, 10) | fld(zn, 5)
            | prfop)
    }
}

/// PRF size code from mnemonic.
fn prf_sz(m: Mnemonic) -> u32 {
    match m {
        Mnemonic::Prfb => 0,
        Mnemonic::Prfh => 1,
        Mnemonic::Prfw => 2,
        _ => 3,
    }
}

/// Encode LDR/STR (vector / predicate register).
fn enc_ldr_str(insn: &Instruction, code: Code) -> Result<u32, EncodeError> {
    let store = matches!(code, SveStrZ | SveStrP);
    let is_vec = matches!(code, SveLdrZ | SveStrZ);
    let (rn, imm) = match addr(insn) {
        Operand::SveMem { base, imm, mode: SveMemMode::ScalarImmMulVl, .. } => {
            (base.number() as u32, imm)
        }
        _ => return Err(EncodeError::InvalidOperand),
    };
    if !(-256..=255).contains(&imm) {
        return Err(EncodeError::InvalidImmediate);
    }
    let imm9 = (imm as u32) & 0x1ff;
    let imm6 = (imm9 >> 3) & 0x3f; // <21:16>
    let imm3 = imm9 & 7; // <12:10>
    let rt = reg(insn, 0)?;
    // STR vector: 0xe5, msz=3 (<24:23>=11), b22=0, op0/op2 (<14>=Z), <31:25>?
    // The decoder reaches LDR/STR via the contiguous/gather quadrant: STR at
    // 0xe5 msz==3 op0/2; LDR at 0x85 msz==3 op0/2.
    let base = if store { base_st() } else { base_g32_ld() };
    Ok(base | fld(3, 23) | fld(0, 22) | fld(imm6, 16) | fld(u32::from(is_vec), 14) | fld(imm3, 10)
        | fld(rn, 5)
        | rt)
}

/// `reg` re-export (data register of LDR/STR, which is a Z or P).
fn reg(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    super::reg(insn, n)
}
