// Included into `int.rs`. Inverts the SVE2 0x44 / 0x45 integer families.

/// Encode the SVE2 integer (0x44 / 0x45) codes. Returns `Ok(None)` if the code
/// is not one of these.
fn enc_sve2(insn: &Instruction, code: Code) -> Result<Option<u32>, EncodeError> {
    let w = match code {
        // ===== 0x44 vector dot / mul-add / widening =====
        SveSdot | SveUdot => {
            // vector form (0x44, <21>=0, <15:11>=00000).
            let u = if matches!(code, SveUdot) { 1 } else { 0 };
            let zda = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            // size: .s/.b -> size=2(<23:22>=10), .d/.h -> size=3.
            let size = if arr_of(insn, 0)? == VA::Sd { 3 } else { 2 };
            base44(0) | fld(size, 22) | fld(zm, 16) | fld(0b00000, 11) | fld(u, 10) | fld(zn, 5) | zda
        }
        SveSdotIdx | SveUdotIdx => enc_44_dot_idx(insn, code)?,
        // ---- SDOT/UDOT 2-way `.h` (SVE2.1): <Zda>.S, <Zn>.H, <Zm>.H{[idx]} ----
        SveSdotH | SveUdotH => {
            // vector form (0x44, <21>=0, <23:22>=00, <15:11>=11001).
            let u = if matches!(code, SveUdotH) { 1 } else { 0 };
            let zda = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            base44(0) | fld(0b00, 22) | fld(zm, 16) | fld(0b11001, 11) | fld(u, 10) | fld(zn, 5) | zda
        }
        SveSdotHIdx | SveUdotHIdx => {
            // indexed form (0x44, <21>=0, <23:22>=10, i2=<20:19>, Zm=<18:16>,
            // <15:11>=11001).
            let u = if matches!(code, SveUdotHIdx) { 1 } else { 0 };
            let zda = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            let idx = lane(insn, 2)?;
            base44(0) | fld(0b10, 22) | fld(idx & 3, 19) | fld(zm & 7, 16) | fld(0b11001, 11)
                | fld(u, 10)
                | fld(zn, 5)
                | zda
        }
        // ---- ZIPQ1/2, UZPQ1/2 (SVE2.1 128-bit-segment permute) ----
        SveZipqUzpq => {
            let op = match insn.mnemonic() {
                Mnemonic::Zipq1 => 0b000,
                Mnemonic::Zipq2 => 0b001,
                Mnemonic::Uzpq1 => 0b010,
                Mnemonic::Uzpq2 => 0b011,
                _ => return Err(EncodeError::InvalidOperand),
            };
            let a = arr_of(insn, 0)?;
            let size = arr_size(a)?;
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            base44(0) | fld(size, 22) | fld(zm, 16) | fld(0b111, 13) | fld(op, 10) | fld(zn, 5) | zd
        }
        // ---- TBLQ (SVE2.1 128-bit-segment table lookup) ----
        SveTblq => {
            let a = arr_of(insn, 0)?;
            let size = arr_size(a)?;
            let zd = z(insn, 0)?;
            let zn = match insn.op(1) {
                Operand::MultiReg { regs, .. } => regs[0].number() as u32,
                _ => return Err(EncodeError::InvalidOperand),
            };
            let zm = z(insn, 2)?;
            base44(0) | fld(size, 22) | fld(zm, 16) | fld(0b111110, 10) | fld(zn, 5) | zd
        }
        SveDotMixed => enc_44_dot_mixed(insn)?,
        SveCdot => enc_44_cdot_vec(insn)?,
        SveCdotIdx => enc_44_cdot_idx(insn)?,
        SveCmla | SveSqrdcmlah => enc_44_cmla_vec(insn, code)?,
        SveCmlaIdx | SveSqrdcmlahIdx => enc_44_cmla_idx(insn, code)?,
        SveMlaIdx | SveMlsIdx => {
            let s = if matches!(code, SveMlsIdx) { 1 } else { 0 };
            let zda = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let (sizef, idx, zm) = idx_same_fields(insn, 2)?;
            base44(1) | sizef | fld(0b00001, 11) | fld(s, 10) | idx_bits(idx, zm, &idx_layout(insn)?)
                | fld(zn, 5)
                | zda
        }
        SveMulIdx => {
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let layout = idx_layout(insn)?;
            let (idx, zm) = idx_read(insn, 2)?;
            base44(1) | fld(0b111110, 10) | idx_bits(idx, zm, &layout) | size_idx(&layout) | fld(zn, 5)
                | zd
        }
        SveSqrdmlah | SveSqrdmlsh => {
            let s = if matches!(code, SveSqrdmlsh) { 1 } else { 0 };
            let a = arr_of(insn, 0)?;
            let size = arr_size(a)?;
            let zda = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            base44(0) | fld(size, 22) | fld(zm, 16) | fld(0b011, 13) | fld(0b10, 11) | fld(s, 10)
                | fld(zn, 5)
                | zda
        }
        SveSqrdmlahIdx | SveSqrdmlshIdx => {
            let s = if matches!(code, SveSqrdmlshIdx) { 1 } else { 0 };
            let zda = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let layout = idx_layout(insn)?;
            let (idx, zm) = idx_read(insn, 2)?;
            base44(1) | fld(0b00010, 11) | fld(s, 10) | idx_bits(idx, zm, &layout) | size_idx(&layout)
                | fld(zn, 5)
                | zda
        }
        SveSqdmulhIdx | SveSqrdmulhIdx => {
            let r = if matches!(code, SveSqrdmulhIdx) { 1 } else { 0 };
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let layout = idx_layout(insn)?;
            let (idx, zm) = idx_read(insn, 2)?;
            base44(1) | fld(0b11110, 11) | fld(r, 10) | idx_bits(idx, zm, &layout) | size_idx(&layout)
                | fld(zn, 5)
                | zd
        }
        SveMlaLong => enc_44_mlal_vec(insn)?,
        SveMlaLongIdx => enc_44_mlal_idx(insn)?,
        SveMulLong => enc_mull_vec(insn, code)?,
        SveMulLongIdx => enc_44_mull_idx(insn)?,
        SvePmulLong => enc_mull_vec(insn, code)?,
        SveSqdmlalLong => enc_44_sqdmlal_vec(insn)?,
        SveSqdmlalLongBt => enc_44_sqdmlal_bt(insn)?,
        SveSqdmlalLongIdx => enc_44_sqdmlal_idx(insn)?,
        SveSqdmulLong => enc_mull_vec(insn, code)?,
        SveSqdmulLongIdx => enc_44_sqdmull_idx(insn)?,
        SveSclamp | SveUclamp => {
            let u = if matches!(code, SveUclamp) { 1 } else { 0 };
            let a = arr_of(insn, 0)?;
            let size = arr_size(a)?;
            let zda = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            base44(0) | fld(size, 22) | fld(zm, 16) | fld(0b110, 13) | fld(0b00, 11) | fld(u, 10)
                | fld(zn, 5)
                | zda
        }
        // ---- 0x44 {S,U}ABAL (SVE2.3 abs-diff accumulate long) ----
        SveSabal | SveUabal => {
            let da = arr_of(insn, 0)?;
            let (size, _) = widen2_of(da)?;
            let u = if matches!(code, SveUabal) { 1 } else { 0 };
            let zda = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            base44(0) | fld(size, 22) | fld(zm, 16) | fld(0b110, 13) | fld(1, 12) | fld(u, 11)
                | fld(1, 10)
                | fld(zn, 5)
                | zda
        }
        // ---- 0x44 predicated SVE2 ----
        SveHalvingZpzz | SveSatRoundZpzz => enc_44_pred_binary(insn, code)?,
        SvePairZpzz => enc_44_pairwise(insn)?,
        SveAdalp => enc_44_adalp(insn)?,
        SveSatUnaryZpz => enc_44_sat_unary(insn)?,
        SveRecipEst => enc_44_recip_est(insn)?,
        // ===== 0x45 family =====
        SveRax1 => {
            // RAX1 <Zd>.D, <Zn>.D, <Zm>.D: top byte 0x45, <23:22>=00, <21>=1,
            // <15:11>=11110, <10>=1.
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            base45(1) | fld(zm, 16) | fld(0b11110, 11) | fld(1, 10) | fld(zn, 5) | zd
        }
        SveShiftAccum | SveShiftLongImm | SveShiftInsert => enc_45_shift_imm(insn, code)?,
        SveShiftNarrow | SveExtractNarrow => enc_45_narrow(insn, code)?,
        SveAddLong | SveAbdLong => enc_45_addsub_long(insn, code)?,
        SveAddWide => enc_45_addsub_wide(insn)?,
        SveAddHighNarrow => enc_45_high_narrow(insn)?,
        SveAddLongBt => enc_45_long_bt(insn)?,
        SveAbaLong => enc_45_abal(insn)?,
        SveAddCarryLong => enc_45_adcl(insn)?,
        SveCadd | SveSqcadd => enc_45_cadd(insn, code)?,
        SveAbaSame => enc_45_aba_same(insn)?,
        SveMatmulInt => enc_45_mmla(insn)?,
        SveBitPerm => enc_45_bitperm(insn)?,
        SveEorInterleave => enc_45_eor_il(insn)?,
        SveHistcnt => enc_45_histcnt(insn)?,
        SveHistseg => enc_45_histseg(insn)?,
        SveAesMc => enc_45_aesmc(insn)?,
        SveAesZz => enc_45_aesz(insn)?,
        SveSm4e => enc_45_sm4e(insn)?,
        SveSm4ekey => enc_45_sm4ekey(insn)?,
        SveMatch => enc_45_match(insn)?,
        // ---- i3: SVE2.3 quadword pair add / add-subtract (0x04, <15:13>=011) ----
        SveAddqp | SveAddsubp => {
            let size = esize(insn, 0)?;
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            let op = if matches!(code, SveAddqp) { 0b110 } else { 0b111 };
            base04(1, 0b011) | fld(size, 22) | fld(op, 10) | fld(zm, 16) | fld(zn, 5) | zd
        }
        // ---- i3: SVE2.3 2-way SDOT/UDOT (.h <- .b) (0x44, <23:22>=01, <15:11>=00000) ----
        SveSdotHb | SveUdotHb => {
            let u = if matches!(code, SveUdotHb) { 1 } else { 0 };
            let zda = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            base44(0) | fld(0b01, 22) | fld(zm, 16) | fld(0b00000, 11) | fld(u, 10) | fld(zn, 5) | zda
        }
        // ---- i3: SVE2.2 SQABS/SQNEG zeroing (0x44, <15:13>=101, <20:16>=0101 op) ----
        SveSqabsZ | SveSqnegZ => {
            let a = arr_of(insn, 0)?;
            let size = arr_size(a)?;
            let zd = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            let op = if matches!(code, SveSqnegZ) { 1 } else { 0 };
            // <19>=1, <18:17>=01 (zeroing), <16>=op.
            base44(0) | fld(size, 22) | fld(1, 19) | fld(0b01, 17) | fld(op, 16) | fld(0b101, 13)
                | fld(pg, 10)
                | fld(zn, 5)
                | zd
        }
        // ---- K4: SVE2.2 URECPE/URSQRTE zeroing (0x44, <15:13>=101, <18:17>=01) ----
        SveUrecpeZ | SveUrsqrteZ => {
            // `.s` only (size=10); <19>=0, <18:17>=01 (zeroing), <16>=op.
            let zd = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zn = z(insn, 2)?;
            let op = if matches!(code, SveUrsqrteZ) { 1 } else { 0 };
            base44(0) | fld(0b10, 22) | fld(0b01, 17) | fld(op, 16) | fld(0b101, 13)
                | fld(pg, 10)
                | fld(zn, 5)
                | zd
        }
        // ---- i3: FEAT_CPA MADPT/MLAPT (0x44, <21>=0, size=11, <15:10>) ----
        SveMadpt => {
            // <15:10>=110110; operands print as Zdn(<4:0>), Zm(<20:16>), Za(<9:5>).
            let zdn = z(insn, 0)?;
            let zm = z(insn, 1)?;
            let za = z(insn, 2)?;
            base44(0) | fld(0b11, 22) | fld(zm, 16) | fld(0b110110, 10) | fld(za, 5) | zdn
        }
        SveMlapt => {
            // <15:10>=110100; operands print as Zda(<4:0>), Zn(<9:5>), Zm(<20:16>).
            let zda = z(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            base44(0) | fld(0b11, 22) | fld(zm, 16) | fld(0b110100, 10) | fld(zn, 5) | zda
        }
        // ---- i3: FEAT_CPA predicated SUBP (0x44, <15:13>=101, <20:16>=10000) ----
        SveSubpPred => {
            let a = arr_of(insn, 0)?;
            let size = arr_size(a)?;
            let zdn = z(insn, 0)?;
            let pg = p(insn, 1)?;
            let zm = z(insn, 3)?;
            // <21:19>=010, <18:16>=000.
            base44(0) | fld(size, 22) | fld(0b010, 19) | fld(0b101, 13) | fld(pg, 10) | fld(zm, 5)
                | zdn
        }
        // ---- K4: FEAT_SVE_AES2 multi-vector quadword AES round ----
        // `{ Zdn.b, .. }, { Zdn.b, .. }, Zm.q[i]`. The destructive group base is
        // `<4:0>`; the indexed Zm is `<9:5>`; quad-vs-pair is `<18>`, index is
        // `<20:19>`, MC is `<16>`, encrypt/decrypt is `<10>`.
        SveAese2 | SveAesd2 | SveAesemc2 | SveAesdimc2 => {
            let (zdn, count) = group_first(insn, 0)?;
            let (zm, idx) = idx_q(insn, 2)?;
            let quad = match count {
                2 => 0,
                4 => 1,
                _ => return Err(EncodeError::InvalidOperand),
            };
            let (mc, dec) = match code {
                SveAese2 => (0, 0),
                SveAesd2 => (0, 1),
                SveAesemc2 => (1, 0),
                _ => (1, 1),
            };
            // <20:19>=idx, <18>=quad, <17>=1, <16>=mc, <15:13>=111, <12:11>=01,
            // <10>=dec.
            base45(1) | fld(idx, 19) | fld(quad, 18) | fld(1, 17) | fld(mc, 16) | fld(0b111, 13)
                | fld(0b01, 11)
                | fld(dec, 10)
                | fld(zm, 5)
                | zdn
        }
        // ---- K4: FEAT_SVE_AES2 polynomial multiply-long, quadword (.q <- .d) ----
        // `{ Zd.q, Zd+1.q }, Zn.d, Zm.d`. <15:13>=111, <12:10>=110 PMULL / 111 PMLAL.
        SvePmull2 | SvePmlal2 => {
            let (zd, _count) = group_first(insn, 0)?;
            let zn = z(insn, 1)?;
            let zm = z(insn, 2)?;
            let op = if matches!(code, SvePmull2) { 0b110 } else { 0b111 };
            base45(1) | fld(zm, 16) | fld(0b111, 13) | fld(op, 10) | fld(zn, 5) | zd
        }
        // ---- K4: SVE2.1 multi-vector saturating narrowing converts ----
        // `Zd.h, { Zn.s, Zn+1.s }`. <15:13>=010, <20:16>=10001, <12:10> sign.
        SveSqcvtn | SveUqcvtn | SveSqcvtun => {
            let zd = z(insn, 0)?;
            let (zn, _count) = group_first(insn, 1)?;
            let op = match code {
                SveSqcvtn => 0b000,
                SveUqcvtn => 0b010,
                _ => 0b100,
            };
            base45(1) | fld(0b10001, 16) | fld(0b010, 13) | fld(op, 10) | fld(zn, 5) | zd
        }
        _ => return Ok(None),
    };
    Ok(Some(w))
}

/// Base word for top byte 0x44 with `<21>=b21`.
#[inline]
fn base44(b21: u32) -> u32 {
    fld(0b01000100, 24) | fld(b21, 21)
}

/// Base word for top byte 0x45 with `<21>=b21`.
#[inline]
fn base45(b21: u32) -> u32 {
    fld(0b01000101, 24) | fld(b21, 21)
}

/// The first register number and member count of an [`Operand::SveVecGroup`] at
/// operand `n`. Used by the FEAT_SVE_AES2 multi-vector group operands.
#[inline]
fn group_first(insn: &Instruction, n: usize) -> Result<(u32, u8), EncodeError> {
    match insn.op(n) {
        Operand::SveVecGroup { first, count, .. } => Ok((first.number() as u32, count)),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The register number and quadword lane index of an indexed `Z.q[i]` operand.
#[inline]
fn idx_q(insn: &Instruction, n: usize) -> Result<(u32, u32), EncodeError> {
    match insn.op(n) {
        Operand::Reg { reg, lane: Some(l), .. } => Ok((reg.number() as u32, l as u32)),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The by-element index layout for an SVE2 same-size indexed form, derived from
/// the destination arrangement (`.h`/`.s`/`.d`).
struct IdxLayout {
    /// Element arrangement.
    a: VA,
}

/// Read the index layout from the destination operand's arrangement.
fn idx_layout(insn: &Instruction) -> Result<IdxLayout, EncodeError> {
    Ok(IdxLayout { a: arr_of(insn, 0)? })
}

/// Pack the `size`/index high bits for a same-size by-element form per layout.
/// Returns the `<23:22>` size contribution.
fn size_idx(layout: &IdxLayout) -> u32 {
    match layout.a {
        VA::Sh => 0, // <23>=0
        VA::Ss => fld(0b10, 22),
        _ => fld(0b11, 22),
    }
}

/// Read the `(idx, zm)` from an indexed operand at `n`.
fn idx_read(insn: &Instruction, n: usize) -> Result<(u32, u32), EncodeError> {
    Ok((lane(insn, n)?, z(insn, n)?))
}

/// Pack the index + Zm bits for a same-size by-element form per layout.
fn idx_bits(idx: u32, zm: u32, layout: &IdxLayout) -> u32 {
    match layout.a {
        VA::Sh => {
            // idx = i3h:i3l = <22>:<20:19>, Zm = <18:16>.
            fld((idx >> 2) & 1, 22) | fld(idx & 3, 19) | fld(zm & 7, 16)
        }
        VA::Ss => fld(idx & 3, 19) | fld(zm & 7, 16),
        _ => fld(idx & 1, 20) | fld(zm & 0xf, 16),
    }
}

/// `(sizef, idx, zm)` for a same-size by-element form (helper for MLA/MLS idx).
fn idx_same_fields(insn: &Instruction, n: usize) -> Result<(u32, u32, u32), EncodeError> {
    let layout = idx_layout(insn)?;
    let (idx, zm) = idx_read(insn, n)?;
    Ok((size_idx(&layout), idx, zm))
}

/// `(wide, narrow)` size code for a 2x-widening op (from the destination arr).
fn widen2_of(da: VA) -> Result<(u32, VA), EncodeError> {
    match da {
        VA::Sh => Ok((0b01, VA::Sb)),
        VA::Ss => Ok((0b10, VA::Sh)),
        VA::Sd => Ok((0b11, VA::Ss)),
        _ => Err(EncodeError::InvalidOperand),
    }
}

/// The 2x-widening index layout for a long indexed form: `.s<-.h` (size=10,
/// idx=i3h:i3l=<20:19>:<11>, Zm=<18:16>), `.d<-.s` (size=11, idx=i2h:i2l=
/// <20>:<11>, Zm=<19:16>).
fn long_idx_bits(da: VA, idx: u32, zm: u32) -> Result<(u32, u32), EncodeError> {
    // Returns (size_field, index_zm_bits).
    match da {
        VA::Ss => {
            let bits = fld((idx >> 1) & 3, 19) | fld(idx & 1, 11) | fld(zm & 7, 16);
            Ok((fld(0b10, 22), bits))
        }
        VA::Sd => {
            let bits = fld((idx >> 1) & 1, 20) | fld(idx & 1, 11) | fld(zm & 0xf, 16);
            Ok((fld(0b11, 22), bits))
        }
        _ => Err(EncodeError::InvalidOperand),
    }
}

// ---- 0x44 dot indexed (SDOT/UDOT) ----
fn enc_44_dot_idx(insn: &Instruction, code: Code) -> Result<u32, EncodeError> {
    let u = if matches!(code, SveUdotIdx) { 1 } else { 0 };
    let zda = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let idx = lane(insn, 2)?;
    let zm = z(insn, 2)?;
    // .s/.b: <23:22>=10, i2=<20:19>, Zm=<18:16>; .d/.h: <23:22>=11, i1=<20>, Zm=<19:16>.
    let da = arr_of(insn, 0)?;
    if da == VA::Ss {
        Ok(base44(1) | fld(0b10, 22) | fld(idx & 3, 19) | fld(zm & 7, 16) | fld(0b00000, 11)
            | fld(u, 10)
            | fld(zn, 5)
            | zda)
    } else {
        Ok(base44(1) | fld(0b11, 22) | fld(idx & 1, 20) | fld(zm & 0xf, 16) | fld(0b00000, 11)
            | fld(u, 10)
            | fld(zn, 5)
            | zda)
    }
}

// ---- 0x44 USDOT/SUDOT (mixed dot) ----
fn enc_44_dot_mixed(insn: &Instruction) -> Result<u32, EncodeError> {
    let m = insn.mnemonic();
    let zda = z(insn, 0)?;
    let zn = z(insn, 1)?;
    // Indexed if operand 2 has a lane.
    if matches!(insn.op(2), Operand::Reg { lane: Some(_), .. }) {
        let idx = lane(insn, 2)?;
        let zm = z(insn, 2)?;
        let u = if matches!(m, Mnemonic::Sudot) { 1 } else { 0 };
        // size=10, <15:11>=00011, i2=<20:19>, Zm=<18:16>.
        Ok(base44(1) | fld(0b10, 22) | fld(idx & 3, 19) | fld(zm & 7, 16) | fld(0b00011, 11)
            | fld(u, 10)
            | fld(zn, 5)
            | zda)
    } else {
        // vector USDOT: size=10, <15:10>=011110.
        let zm = z(insn, 2)?;
        Ok(base44(0) | fld(0b10, 22) | fld(zm, 16) | fld(0b011110, 10) | fld(zn, 5) | zda)
    }
}

// ---- 0x44 CDOT vector ----
fn enc_44_cdot_vec(insn: &Instruction) -> Result<u32, EncodeError> {
    let zda = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let rot = rot_field(insn, 3)?;
    let size = if arr_of(insn, 0)? == VA::Sd { 3 } else { 2 };
    Ok(base44(0) | fld(size, 22) | fld(zm, 16) | fld(0b0001, 12) | fld(rot, 10) | fld(zn, 5) | zda)
}

// ---- 0x44 CDOT indexed ----
fn enc_44_cdot_idx(insn: &Instruction) -> Result<u32, EncodeError> {
    let zda = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let idx = lane(insn, 2)?;
    let rot = rot_field(insn, 3)?;
    let da = arr_of(insn, 0)?;
    if da == VA::Ss {
        Ok(base44(1) | fld(0b10, 22) | fld(idx & 3, 19) | fld(zm & 7, 16) | fld(0b0100, 12)
            | fld(rot, 10)
            | fld(zn, 5)
            | zda)
    } else {
        Ok(base44(1) | fld(0b11, 22) | fld(idx & 1, 20) | fld(zm & 0xf, 16) | fld(0b0100, 12)
            | fld(rot, 10)
            | fld(zn, 5)
            | zda)
    }
}

// ---- 0x44 CMLA / SQRDCMLAH vector ----
fn enc_44_cmla_vec(insn: &Instruction, code: Code) -> Result<u32, EncodeError> {
    let op = if matches!(code, SveSqrdcmlah) { 1 } else { 0 };
    let a = arr_of(insn, 0)?;
    let size = arr_size(a)?;
    let zda = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let rot = rot_field(insn, 3)?;
    Ok(base44(0) | fld(size, 22) | fld(zm, 16) | fld(0b001, 13) | fld(op, 12) | fld(rot, 10)
        | fld(zn, 5)
        | zda)
}

// ---- 0x44 CMLA / SQRDCMLAH indexed ----
fn enc_44_cmla_idx(insn: &Instruction, code: Code) -> Result<u32, EncodeError> {
    let op = if matches!(code, SveSqrdcmlahIdx) { 1 } else { 0 };
    let zda = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let idx = lane(insn, 2)?;
    let rot = rot_field(insn, 3)?;
    let a = arr_of(insn, 0)?;
    // .h (size=10): i2=<20:19>, Zm=<18:16>; .s (size=11): i1=<20>, Zm=<19:16>.
    if a == VA::Sh {
        Ok(base44(1) | fld(0b10, 22) | fld(idx & 3, 19) | fld(zm & 7, 16) | fld(0b011, 13)
            | fld(op, 12)
            | fld(rot, 10)
            | fld(zn, 5)
            | zda)
    } else {
        Ok(base44(1) | fld(0b11, 22) | fld(idx & 1, 20) | fld(zm & 0xf, 16) | fld(0b011, 13)
            | fld(op, 12)
            | fld(rot, 10)
            | fld(zn, 5)
            | zda)
    }
}

/// Recover the 2-bit `rot` field from a rotation immediate operand (0/90/180/270).
fn rot_field(insn: &Instruction, n: usize) -> Result<u32, EncodeError> {
    match imm(insn, n)? {
        0 => Ok(0),
        90 => Ok(1),
        180 => Ok(2),
        270 => Ok(3),
        _ => Err(EncodeError::InvalidImmediate),
    }
}

// ---- 0x44 {S,U}ML{A,S}L{B,T} vector long ----
fn enc_44_mlal_vec(insn: &Instruction) -> Result<u32, EncodeError> {
    let da = arr_of(insn, 0)?;
    let (size, _) = widen2_of(da)?;
    let (s, u, t) = mlal_sut(insn.mnemonic())?;
    let zda = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    Ok(base44(0) | fld(size, 22) | fld(zm, 16) | fld(0b010, 13) | fld(s, 12) | fld(u, 11) | fld(t, 10)
        | fld(zn, 5)
        | zda)
}

// ---- 0x44 {S,U}ML{A,S}L{B,T} indexed long ----
fn enc_44_mlal_idx(insn: &Instruction) -> Result<u32, EncodeError> {
    let da = arr_of(insn, 0)?;
    let (s, u, t) = mlal_sut(insn.mnemonic())?;
    let zda = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let idx = lane(insn, 2)?;
    let (sizef, ib) = long_idx_bits(da, idx, zm)?;
    Ok(base44(1) | sizef | ib | fld(0b10, 14) | fld(s, 13) | fld(u, 12) | fld(t, 10) | fld(zn, 5)
        | zda)
}

/// `(S, U, T)` for the {S,U}ML{A,S}L{B,T} family.
fn mlal_sut(m: Mnemonic) -> Result<(u32, u32, u32), EncodeError> {
    Ok(match m {
        Mnemonic::Smlalb => (0, 0, 0),
        Mnemonic::Smlalt => (0, 0, 1),
        Mnemonic::Umlalb => (0, 1, 0),
        Mnemonic::Umlalt => (0, 1, 1),
        Mnemonic::Smlslb => (1, 0, 0),
        Mnemonic::Smlslt => (1, 0, 1),
        Mnemonic::Umlslb => (1, 1, 0),
        Mnemonic::Umlslt => (1, 1, 1),
        _ => return Err(EncodeError::InvalidOperand),
    })
}

// ---- 0x44 {S,U}MULL{B,T} indexed long ----
fn enc_44_mull_idx(insn: &Instruction) -> Result<u32, EncodeError> {
    let da = arr_of(insn, 0)?;
    let (u, t) = match insn.mnemonic() {
        Mnemonic::Smullb => (0, 0),
        Mnemonic::Smullt => (0, 1),
        Mnemonic::Umullb => (1, 0),
        Mnemonic::Umullt => (1, 1),
        _ => return Err(EncodeError::InvalidOperand),
    };
    let zd = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let idx = lane(insn, 2)?;
    let (sizef, ib) = long_idx_bits(da, idx, zm)?;
    Ok(base44(1) | sizef | ib | fld(0b110, 13) | fld(u, 12) | fld(t, 10) | fld(zn, 5) | zd)
}

// ---- 0x44 SQDML{A,S}L{B,T} vector ----
fn enc_44_sqdmlal_vec(insn: &Instruction) -> Result<u32, EncodeError> {
    let da = arr_of(insn, 0)?;
    let (size, _) = widen2_of(da)?;
    let (s, t) = sqdmlal_st(insn.mnemonic())?;
    let zda = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    Ok(base44(0) | fld(size, 22) | fld(zm, 16) | fld(0b011, 13) | fld(0, 12) | fld(s, 11) | fld(t, 10)
        | fld(zn, 5)
        | zda)
}

// ---- 0x44 SQDML{A,S}L{B,T} indexed ----
fn enc_44_sqdmlal_idx(insn: &Instruction) -> Result<u32, EncodeError> {
    let da = arr_of(insn, 0)?;
    let (s, t) = sqdmlal_st(insn.mnemonic())?;
    let zda = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let idx = lane(insn, 2)?;
    let (sizef, ib) = long_idx_bits(da, idx, zm)?;
    Ok(base44(1) | sizef | ib | fld(0b001, 13) | fld(s, 12) | fld(t, 10) | fld(zn, 5) | zda)
}

/// `(S, T)` for the SQDML{A,S}L{B,T} family.
fn sqdmlal_st(m: Mnemonic) -> Result<(u32, u32), EncodeError> {
    Ok(match m {
        Mnemonic::Sqdmlalb => (0, 0),
        Mnemonic::Sqdmlalt => (0, 1),
        Mnemonic::Sqdmlslb => (1, 0),
        Mnemonic::Sqdmlslt => (1, 1),
        _ => return Err(EncodeError::InvalidOperand),
    })
}

// ---- 0x44 SQDMLALBT/SQDMLSLBT ----
fn enc_44_sqdmlal_bt(insn: &Instruction) -> Result<u32, EncodeError> {
    let da = arr_of(insn, 0)?;
    let (size, _) = widen2_of(da)?;
    let s = if matches!(insn.mnemonic(), Mnemonic::Sqdmlslbt) { 1 } else { 0 };
    let zda = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    Ok(base44(0) | fld(size, 22) | fld(zm, 16) | fld(0b00001, 11) | fld(s, 10) | fld(zn, 5) | zda)
}

// ---- 0x44 SQDMULL{B,T} indexed ----
fn enc_44_sqdmull_idx(insn: &Instruction) -> Result<u32, EncodeError> {
    let da = arr_of(insn, 0)?;
    let t = if matches!(insn.mnemonic(), Mnemonic::Sqdmullt) { 1 } else { 0 };
    let zd = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let idx = lane(insn, 2)?;
    let (sizef, ib) = long_idx_bits(da, idx, zm)?;
    Ok(base44(1) | sizef | ib | fld(0b1110, 12) | fld(t, 10) | fld(zn, 5) | zd)
}

// ---- 0x45 multiply-long vector (SMULL/UMULL/SQDMULL/PMULL B/T) ----
fn enc_mull_vec(insn: &Instruction, code: Code) -> Result<u32, EncodeError> {
    let da = arr_of(insn, 0)?;
    let (op, size) = match (code, insn.mnemonic()) {
        (SveSqdmulLong, Mnemonic::Sqdmullb) => (0b1000, widen2_of(da)?.0),
        (SveSqdmulLong, Mnemonic::Sqdmullt) => (0b1001, widen2_of(da)?.0),
        (SvePmulLong, Mnemonic::Pmullb) => (0b1010, pmull_size(da)?),
        (SvePmulLong, Mnemonic::Pmullt) => (0b1011, pmull_size(da)?),
        (SveMulLong, Mnemonic::Smullb) => (0b1100, widen2_of(da)?.0),
        (SveMulLong, Mnemonic::Smullt) => (0b1101, widen2_of(da)?.0),
        (SveMulLong, Mnemonic::Umullb) => (0b1110, widen2_of(da)?.0),
        (SveMulLong, Mnemonic::Umullt) => (0b1111, widen2_of(da)?.0),
        _ => return Err(EncodeError::InvalidOperand),
    };
    let zd = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    Ok(base45(0) | fld(size, 22) | fld(zm, 16) | fld(0b011, 13) | fld(op, 10) | fld(zn, 5) | zd)
}

/// Size code for PMULL{B,T}, which adds the `.q<-.d` (size=00) crypto form.
fn pmull_size(da: VA) -> Result<u32, EncodeError> {
    match da {
        VA::Sq => Ok(0b00),
        _ => Ok(widen2_of(da)?.0),
    }
}

// ---- 0x45 shift-immediate family (<21>=0) ----
fn enc_45_shift_imm(insn: &Instruction, code: Code) -> Result<u32, EncodeError> {
    let m = insn.mnemonic();
    match code {
        SveShiftAccum => {
            let a = arr_of(insn, 0)?;
            let amount = imm(insn, 2)? as u32;
            let (tsz, imm3) = enc_right_shift(a, amount)?;
            let (r, u) = match m {
                Mnemonic::Ssra => (0, 0),
                Mnemonic::Usra => (0, 1),
                Mnemonic::Srsra => (1, 0),
                _ => (1, 1),
            };
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            Ok(shift45_imm(tsz, imm3) | fld(0b1110, 12) | fld(r, 11) | fld(u, 10) | fld(zn, 5) | zd)
        }
        SveShiftLongImm => {
            let sa = arr_of(insn, 1)?; // narrow source carries the shift amount
            let amount = imm(insn, 2)? as u32;
            let (tsz, imm3) = enc_left_shift(sa, amount)?;
            let (u, t) = match m {
                Mnemonic::Sshllb => (0, 0),
                Mnemonic::Sshllt => (0, 1),
                Mnemonic::Ushllb => (1, 0),
                _ => (1, 1),
            };
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            Ok(shift45_imm(tsz, imm3) | fld(0b1010, 12) | fld(u, 11) | fld(t, 10) | fld(zn, 5) | zd)
        }
        _ => {
            // SLI/SRI.
            let a = arr_of(insn, 0)?;
            let amount = imm(insn, 2)? as u32;
            let op = if matches!(m, Mnemonic::Sli) { 1 } else { 0 };
            let (tsz, imm3) = if op == 1 {
                enc_left_shift(a, amount)?
            } else {
                enc_right_shift(a, amount)?
            };
            let zd = z(insn, 0)?;
            let zn = z(insn, 1)?;
            Ok(shift45_imm(tsz, imm3) | fld(0b11110, 11) | fld(op, 10) | fld(zn, 5) | zd)
        }
    }
}

/// 0x45 shift-immediate skeleton: tszh=<23:22>, tszl=<20:19>, imm3=<18:16>,
/// with `<21>=0`.
#[inline]
fn shift45_imm(tsz: u32, imm3: u32) -> u32 {
    fld(0b01000101, 24) | fld(tsz >> 2, 22) | fld(tsz & 3, 19) | fld(imm3, 16)
}

// ---- 0x45 narrowing-shift / saturating-extract-narrow (<21>=1) ----
fn enc_45_narrow(insn: &Instruction, code: Code) -> Result<u32, EncodeError> {
    let m = insn.mnemonic();
    let zd = z(insn, 0)?;
    let zn = z(insn, 1)?;
    if matches!(code, SveExtractNarrow) {
        // saturating extract narrow: dest narrow arr carries the size.
        let da = arr_of(insn, 0)?;
        let idx = arr_size(da)?;
        let tszn = 1u32 << idx; // tsz with top bit at idx, imm3=0 region.
        let (opc, t) = match m {
            Mnemonic::Sqxtnb => (0b00, 0),
            Mnemonic::Sqxtnt => (0b00, 1),
            Mnemonic::Uqxtnb => (0b01, 0),
            Mnemonic::Uqxtnt => (0b01, 1),
            Mnemonic::Sqxtunb => (0b10, 0),
            _ => (0b10, 1),
        };
        // narrowing tsz layout: <22>:<20:19>; imm3=<18:16>=000 -> <18:13>=000010.
        return Ok(base45(1) | fld(tszn >> 2, 22) | fld(tszn & 3, 19) | fld(0b000010, 13)
            | fld(opc, 11)
            | fld(t, 10)
            | fld(zn, 5)
            | zd);
    }
    // narrowing shift right: dest narrow arr carries the shift amount.
    let da = arr_of(insn, 0)?;
    let amount = imm(insn, 2)? as u32;
    let (tszn, imm3) = enc_right_shift(da, amount)?;
    let (op, u, r, t) = match m {
        Mnemonic::Shrnb => (0, 1, 0, 0),
        Mnemonic::Shrnt => (0, 1, 0, 1),
        Mnemonic::Rshrnb => (0, 1, 1, 0),
        Mnemonic::Rshrnt => (0, 1, 1, 1),
        Mnemonic::Sqshrnb => (1, 0, 0, 0),
        Mnemonic::Sqshrnt => (1, 0, 0, 1),
        Mnemonic::Sqrshrnb => (1, 0, 1, 0),
        Mnemonic::Sqrshrnt => (1, 0, 1, 1),
        Mnemonic::Sqshrunb => (0, 0, 0, 0),
        Mnemonic::Sqshrunt => (0, 0, 0, 1),
        Mnemonic::Sqrshrunb => (0, 0, 1, 0),
        Mnemonic::Sqrshrunt => (0, 0, 1, 1),
        Mnemonic::Uqshrnb => (1, 1, 0, 0),
        Mnemonic::Uqshrnt => (1, 1, 0, 1),
        Mnemonic::Uqrshrnb => (1, 1, 1, 0),
        _ => (1, 1, 1, 1),
    };
    // <14:13>=00, then op<13>,u<12>,r<11>,t<10>; tsz=<22>:<20:19>, imm3=<18:16>.
    Ok(base45(1) | fld(tszn >> 2, 22) | fld(tszn & 3, 19) | fld(imm3, 16) | fld(op, 13) | fld(u, 12)
        | fld(r, 11)
        | fld(t, 10)
        | fld(zn, 5)
        | zd)
}

// ---- 0x45 add/sub long & abd long (<21>=0, <15:13>=000/001) ----
fn enc_45_addsub_long(insn: &Instruction, code: Code) -> Result<u32, EncodeError> {
    let da = arr_of(insn, 0)?;
    let (size, _) = widen2_of(da)?;
    let zd = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let m = insn.mnemonic();
    if matches!(code, SveAbdLong) {
        let (u, t) = match m {
            Mnemonic::Sabdlb => (0, 0),
            Mnemonic::Sabdlt => (0, 1),
            Mnemonic::Uabdlb => (1, 0),
            _ => (1, 1),
        };
        // <15:13>=001, <12>=S=1 (abdl uses S=1? decoder: abd=bit13). Actually
        // ABDL has sel<15:13>=001; s=<12>=1 per `abd = bit(word,13)==1`. The
        // decoder reads s=<12> but ABDL ignores it; binja sets it 1. Use s=1.
        return Ok(base45(0) | fld(size, 22) | fld(zm, 16) | fld(0b001, 13) | fld(1, 12) | fld(u, 11)
            | fld(t, 10)
            | fld(zn, 5)
            | zd);
    }
    let (s, u, t) = match m {
        Mnemonic::Saddlb => (0, 0, 0),
        Mnemonic::Saddlt => (0, 0, 1),
        Mnemonic::Uaddlb => (0, 1, 0),
        Mnemonic::Uaddlt => (0, 1, 1),
        Mnemonic::Ssublb => (1, 0, 0),
        Mnemonic::Ssublt => (1, 0, 1),
        Mnemonic::Usublb => (1, 1, 0),
        _ => (1, 1, 1),
    };
    Ok(base45(0) | fld(size, 22) | fld(zm, 16) | fld(0b000, 13) | fld(s, 12) | fld(u, 11) | fld(t, 10)
        | fld(zn, 5)
        | zd)
}

// ---- 0x45 add/sub wide (<21>=0, <15:13>=010) ----
fn enc_45_addsub_wide(insn: &Instruction) -> Result<u32, EncodeError> {
    let da = arr_of(insn, 0)?;
    let (size, _) = widen2_of(da)?;
    let zd = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let (s, u, t) = match insn.mnemonic() {
        Mnemonic::Saddwb => (0, 0, 0),
        Mnemonic::Saddwt => (0, 0, 1),
        Mnemonic::Uaddwb => (0, 1, 0),
        Mnemonic::Uaddwt => (0, 1, 1),
        Mnemonic::Ssubwb => (1, 0, 0),
        Mnemonic::Ssubwt => (1, 0, 1),
        Mnemonic::Usubwb => (1, 1, 0),
        _ => (1, 1, 1),
    };
    Ok(base45(0) | fld(size, 22) | fld(zm, 16) | fld(0b010, 13) | fld(s, 12) | fld(u, 11) | fld(t, 10)
        | fld(zn, 5)
        | zd)
}

// ---- 0x45 high-narrowing (<21>=1, <15:13>=011) ----
fn enc_45_high_narrow(insn: &Instruction) -> Result<u32, EncodeError> {
    // Zd narrow, Zn/Zm wide. size from wide arrangement.
    let da = arr_of(insn, 1)?; // wide
    let (size, _) = widen2_of(da)?;
    let zd = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let (s, r, t) = match insn.mnemonic() {
        Mnemonic::Addhnb => (0, 0, 0),
        Mnemonic::Addhnt => (0, 0, 1),
        Mnemonic::Raddhnb => (0, 1, 0),
        Mnemonic::Raddhnt => (0, 1, 1),
        Mnemonic::Subhnb => (1, 0, 0),
        Mnemonic::Subhnt => (1, 0, 1),
        Mnemonic::Rsubhnb => (1, 1, 0),
        _ => (1, 1, 1),
    };
    Ok(base45(1) | fld(size, 22) | fld(zm, 16) | fld(0b011, 13) | fld(s, 12) | fld(r, 11) | fld(t, 10)
        | fld(zn, 5)
        | zd)
}

// ---- 0x45 SADDLBT/SSUBLBT/SSUBLTB (<21>=0, <15:13>=100) ----
fn enc_45_long_bt(insn: &Instruction) -> Result<u32, EncodeError> {
    let da = arr_of(insn, 0)?;
    let (size, _) = widen2_of(da)?;
    let zd = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let (s, tb) = match insn.mnemonic() {
        Mnemonic::Saddlbt => (0, 0),
        Mnemonic::Ssublbt => (1, 0),
        _ => (1, 1), // Ssubltb
    };
    Ok(base45(0) | fld(size, 22) | fld(zm, 16) | fld(0b100, 13) | fld(0, 12) | fld(s, 11) | fld(tb, 10)
        | fld(zn, 5)
        | zd)
}

// ---- 0x45 {S,U}ABAL{B,T} (<21>=0, <15:12>=1100) ----
fn enc_45_abal(insn: &Instruction) -> Result<u32, EncodeError> {
    let da = arr_of(insn, 0)?;
    let (size, _) = widen2_of(da)?;
    let zd = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let (u, t) = match insn.mnemonic() {
        Mnemonic::Sabalb => (0, 0),
        Mnemonic::Sabalt => (0, 1),
        Mnemonic::Uabalb => (1, 0),
        _ => (1, 1),
    };
    Ok(base45(0) | fld(size, 22) | fld(zm, 16) | fld(0b110, 13) | fld(0, 12) | fld(u, 11) | fld(t, 10)
        | fld(zn, 5)
        | zd)
}

// ---- 0x45 ADCL/SBCL (<21>=0, <15:11>=11010) ----
fn enc_45_adcl(insn: &Instruction) -> Result<u32, EncodeError> {
    let a = arr_of(insn, 0)?;
    // .s -> <22>=0, .d -> <22>=1.
    let sz = if a == VA::Sd { 1 } else { 0 };
    let zd = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let (op, t) = match insn.mnemonic() {
        Mnemonic::Adclb => (0, 0),
        Mnemonic::Adclt => (0, 1),
        Mnemonic::Sbclb => (1, 0),
        _ => (1, 1),
    };
    // <23>=op, <22>=sz, <15:11>=11010, <12:10>=10T.
    Ok(base45(0) | fld(op, 23) | fld(sz, 22) | fld(zm, 16) | fld(0b110, 13) | fld(0b10, 11)
        | fld(t, 10)
        | fld(zn, 5)
        | zd)
}

// ---- 0x45 CADD / SQCADD (<21>=0, <15:11>=11011) ----
fn enc_45_cadd(insn: &Instruction, code: Code) -> Result<u32, EncodeError> {
    let a = arr_of(insn, 0)?;
    let size = arr_size(a)?;
    let zdn = z(insn, 0)?;
    let zm = z(insn, 2)?;
    let sqr = matches!(code, SveSqcadd);
    let rot = match imm(insn, 3)? {
        90 => 0,
        270 => 1,
        _ => return Err(EncodeError::InvalidImmediate),
    };
    // <16>=sqr, <15:11>=11011, <10>=rot, <21:17>=0.
    Ok(base45(0) | fld(size, 22) | fld(u32::from(sqr), 16) | fld(0b110, 13) | fld(0b11, 11)
        | fld(rot, 10)
        | fld(zm, 5)
        | zdn)
}

// ---- 0x45 {S,U}ABA same-size (<21>=0, <15:11>=11111) ----
fn enc_45_aba_same(insn: &Instruction) -> Result<u32, EncodeError> {
    let a = arr_of(insn, 0)?;
    let size = arr_size(a)?;
    let zd = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let u = if matches!(insn.mnemonic(), Mnemonic::Uaba) { 1 } else { 0 };
    Ok(base45(0) | fld(size, 22) | fld(zm, 16) | fld(0b11111, 11) | fld(u, 10) | fld(zn, 5) | zd)
}

// ---- 0x45 integer matmul (SMMLA/USMMLA/UMMLA) (<21>=0, <15:10>=100110) ----
fn enc_45_mmla(insn: &Instruction) -> Result<u32, EncodeError> {
    let size = match insn.mnemonic() {
        Mnemonic::Smmla => 0b00,
        Mnemonic::Usmmla => 0b10,
        _ => 0b11,
    };
    let zd = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    Ok(base45(0) | fld(size, 22) | fld(zm, 16) | fld(0b100110, 10) | fld(zn, 5) | zd)
}

// ---- 0x45 bit-permute (BEXT/BDEP/BGRP) (<21>=0, <15:12>=1011) ----
fn enc_45_bitperm(insn: &Instruction) -> Result<u32, EncodeError> {
    let a = arr_of(insn, 0)?;
    let size = arr_size(a)?;
    let zd = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let opc = match insn.mnemonic() {
        Mnemonic::Bext => 0b00,
        Mnemonic::Bdep => 0b01,
        _ => 0b10,
    };
    Ok(base45(0) | fld(size, 22) | fld(zm, 16) | fld(0b1011, 12) | fld(opc, 10) | fld(zn, 5) | zd)
}

// ---- 0x45 EORBT/EORTB (<21>=0, <15:11>=10010) ----
fn enc_45_eor_il(insn: &Instruction) -> Result<u32, EncodeError> {
    let a = arr_of(insn, 0)?;
    let size = arr_size(a)?;
    let zd = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    let tb = if matches!(insn.mnemonic(), Mnemonic::Eortb) { 1 } else { 0 };
    Ok(base45(0) | fld(size, 22) | fld(zm, 16) | fld(0b10010, 11) | fld(tb, 10) | fld(zn, 5) | zd)
}

// ---- 0x45 HISTCNT (<21>=1, <15:13>=110) ----
fn enc_45_histcnt(insn: &Instruction) -> Result<u32, EncodeError> {
    let a = arr_of(insn, 0)?;
    let size = arr_size(a)?;
    let zd = z(insn, 0)?;
    let pg = p(insn, 1)?;
    let zn = z(insn, 2)?;
    let zm = z(insn, 3)?;
    Ok(base45(1) | fld(size, 22) | fld(zm, 16) | fld(0b110, 13) | fld(pg, 10) | fld(zn, 5) | zd)
}

// ---- 0x45 HISTSEG (<21>=1, <15:10>=101000) ----
fn enc_45_histseg(insn: &Instruction) -> Result<u32, EncodeError> {
    let zd = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    Ok(base45(1) | fld(0b00, 22) | fld(zm, 16) | fld(0b101000, 10) | fld(zn, 5) | zd)
}

// ---- 0x45 AESMC/AESIMC (<21>=1) ----
fn enc_45_aesmc(insn: &Instruction) -> Result<u32, EncodeError> {
    let zd = z(insn, 0)?;
    let imc = if matches!(insn.mnemonic(), Mnemonic::Aesimc) { 1 } else { 0 };
    Ok(base45(1) | fld(0b100000, 16) | fld(0b11100, 11) | fld(imc, 10) | zd)
}

// ---- 0x45 AESE/AESD (<21>=1) ----
fn enc_45_aesz(insn: &Instruction) -> Result<u32, EncodeError> {
    let zd = z(insn, 0)?;
    let zm = z(insn, 2)?;
    let d = if matches!(insn.mnemonic(), Mnemonic::Aesd) { 1 } else { 0 };
    Ok(base45(1) | fld(0b100010, 16) | fld(0b11100, 11) | fld(d, 10) | fld(zm, 5) | zd)
}

// ---- 0x45 SM4E (<21>=1) ----
fn enc_45_sm4e(insn: &Instruction) -> Result<u32, EncodeError> {
    let zd = z(insn, 0)?;
    let zn = z(insn, 2)?;
    Ok(base45(1) | fld(0b100011, 16) | fld(0b111000, 10) | fld(zn, 5) | zd)
}

// ---- 0x45 SM4EKEY (<21>=1) ----
fn enc_45_sm4ekey(insn: &Instruction) -> Result<u32, EncodeError> {
    let zd = z(insn, 0)?;
    let zn = z(insn, 1)?;
    let zm = z(insn, 2)?;
    Ok(base45(1) | fld(zm, 16) | fld(0b11110, 11) | fld(0, 10) | fld(zn, 5) | zd)
}

// ---- 0x45 MATCH/NMATCH (<21>=1, <15:13>=100) ----
fn enc_45_match(insn: &Instruction) -> Result<u32, EncodeError> {
    let a = arr_of(insn, 0)?;
    let size = arr_size(a)?;
    let pd = p(insn, 0)?;
    let pg = p(insn, 1)?;
    let zn = z(insn, 2)?;
    let zm = z(insn, 3)?;
    let op = if matches!(insn.mnemonic(), Mnemonic::Nmatch) { 1 } else { 0 };
    Ok(base45(1) | fld(size, 22) | fld(zm, 16) | fld(0b100, 13) | fld(pg, 10) | fld(zn, 5)
        | fld(op, 4)
        | pd)
}

// ---- 0x44 predicated binary (halving / saturating-rounding) ----
fn enc_44_pred_binary(insn: &Instruction, code: Code) -> Result<u32, EncodeError> {
    let a = arr_of(insn, 0)?;
    let size = arr_size(a)?;
    let zdn = z(insn, 0)?;
    let pg = p(insn, 1)?;
    let zm = z(insn, 3)?;
    let m = insn.mnemonic();
    let (sel19, opc16) = match (code, m) {
        (SveHalvingZpzz, Mnemonic::Shadd) => (0b010, 0b000),
        (SveHalvingZpzz, Mnemonic::Uhadd) => (0b010, 0b001),
        (SveHalvingZpzz, Mnemonic::Shsub) => (0b010, 0b010),
        (SveHalvingZpzz, Mnemonic::Uhsub) => (0b010, 0b011),
        (SveHalvingZpzz, Mnemonic::Srhadd) => (0b010, 0b100),
        (SveHalvingZpzz, Mnemonic::Urhadd) => (0b010, 0b101),
        (SveHalvingZpzz, Mnemonic::Shsubr) => (0b010, 0b110),
        (SveHalvingZpzz, Mnemonic::Uhsubr) => (0b010, 0b111),
        (SveSatRoundZpzz, Mnemonic::Sqadd) => (0b011, 0b000),
        (SveSatRoundZpzz, Mnemonic::Uqadd) => (0b011, 0b001),
        (SveSatRoundZpzz, Mnemonic::Sqsub) => (0b011, 0b010),
        (SveSatRoundZpzz, Mnemonic::Uqsub) => (0b011, 0b011),
        (SveSatRoundZpzz, Mnemonic::Suqadd) => (0b011, 0b100),
        (SveSatRoundZpzz, Mnemonic::Usqadd) => (0b011, 0b101),
        (SveSatRoundZpzz, Mnemonic::Sqsubr) => (0b011, 0b110),
        (SveSatRoundZpzz, Mnemonic::Uqsubr) => (0b011, 0b111),
        // shift-left rounding/saturating (<21:19>=00X): keyed by (Q,R,N,U).
        (SveSatRoundZpzz, _) => return enc_44_shift_left_reg(insn, zdn, pg, zm, size),
        _ => return Err(EncodeError::InvalidOperand),
    };
    Ok(base44(0) | fld(size, 22) | fld(sel19, 19) | fld(opc16, 16) | fld(0b100, 13) | fld(pg, 10)
        | fld(zm, 5)
        | zdn)
}

/// SVE2 predicated shift-left (rounding/saturating) register form.
fn enc_44_shift_left_reg(
    insn: &Instruction,
    zdn: u32,
    pg: u32,
    zm: u32,
    size: u32,
) -> Result<u32, EncodeError> {
    let (q, r, n, u) = match insn.mnemonic() {
        Mnemonic::Srshl => (0, 0, 1, 0),
        Mnemonic::Urshl => (0, 0, 1, 1),
        Mnemonic::Srshlr => (0, 1, 1, 0),
        Mnemonic::Urshlr => (0, 1, 1, 1),
        Mnemonic::Sqshl => (1, 0, 0, 0),
        Mnemonic::Uqshl => (1, 0, 0, 1),
        Mnemonic::Sqrshl => (1, 0, 1, 0),
        Mnemonic::Uqrshl => (1, 0, 1, 1),
        Mnemonic::Sqshlr => (1, 1, 0, 0),
        Mnemonic::Uqshlr => (1, 1, 0, 1),
        Mnemonic::Sqrshlr => (1, 1, 1, 0),
        Mnemonic::Uqrshlr => (1, 1, 1, 1),
        _ => return Err(EncodeError::InvalidOperand),
    };
    // <21:20>=00 (sel19 top two bits 0), <19>=Q,<18>=R,<17>=N,<16>=U.
    Ok(base44(0) | fld(size, 22) | fld(q, 19) | fld(r, 18) | fld(n, 17) | fld(u, 16) | fld(0b100, 13)
        | fld(pg, 10)
        | fld(zm, 5)
        | zdn)
}

// ---- 0x44 pairwise (ADDP / {S,U}{MAX,MIN}P) ----
fn enc_44_pairwise(insn: &Instruction) -> Result<u32, EncodeError> {
    let a = arr_of(insn, 0)?;
    let size = arr_size(a)?;
    let zdn = z(insn, 0)?;
    let pg = p(insn, 1)?;
    let zm = z(insn, 3)?;
    let opc = match insn.mnemonic() {
        Mnemonic::Addp => 0b001,
        Mnemonic::Smaxp => 0b100,
        Mnemonic::Umaxp => 0b101,
        Mnemonic::Sminp => 0b110,
        _ => 0b111,
    };
    Ok(base44(0) | fld(size, 22) | fld(0b010, 19) | fld(opc, 16) | fld(0b101, 13) | fld(pg, 10)
        | fld(zm, 5)
        | zdn)
}

// ---- 0x44 SADALP/UADALP ----
fn enc_44_adalp(insn: &Instruction) -> Result<u32, EncodeError> {
    let da = arr_of(insn, 0)?;
    let (size, _) = widen2_of(da)?;
    let zdn = z(insn, 0)?;
    let pg = p(insn, 1)?;
    let zn = z(insn, 2)?;
    let u = if matches!(insn.mnemonic(), Mnemonic::Uadalp) { 1 } else { 0 };
    // <15:13>=101, <20:16>=00010U.
    Ok(base44(0) | fld(size, 22) | fld(0b00010, 17) | fld(u, 16) | fld(0b101, 13) | fld(pg, 10)
        | fld(zn, 5)
        | zdn)
}

// ---- 0x44 SQABS/SQNEG ----
fn enc_44_sat_unary(insn: &Instruction) -> Result<u32, EncodeError> {
    let a = arr_of(insn, 0)?;
    let size = arr_size(a)?;
    let zdn = z(insn, 0)?;
    let pg = p(insn, 1)?;
    let zn = z(insn, 2)?;
    let op = if matches!(insn.mnemonic(), Mnemonic::Sqneg) { 1 } else { 0 };
    // <15:13>=101, <20:16>=01000/01001 -> <19>=1,<18:16>=00 op.
    Ok(base44(0) | fld(size, 22) | fld(1, 19) | fld(op, 16) | fld(0b101, 13) | fld(pg, 10)
        | fld(zn, 5)
        | zdn)
}

// ---- 0x44 URECPE/URSQRTE ----
fn enc_44_recip_est(insn: &Instruction) -> Result<u32, EncodeError> {
    let a = arr_of(insn, 0)?;
    let size = arr_size(a)?;
    let zdn = z(insn, 0)?;
    let pg = p(insn, 1)?;
    let zn = z(insn, 2)?;
    let op = if matches!(insn.mnemonic(), Mnemonic::Ursqrte) { 1 } else { 0 };
    // <15:13>=101, <20:16>=00000/00001 -> <19>=0,<18:17>=00,<16>=op.
    Ok(base44(0) | fld(size, 22) | fld(op, 16) | fld(0b101, 13) | fld(pg, 10) | fld(zn, 5) | zdn)
}
