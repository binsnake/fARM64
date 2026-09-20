//! Implicit register read/write tests for [`fARM64::implicit`].
//!
//! Each case decodes a single A64 word and asserts the exact set of registers
//! the instruction touches *without naming them in an operand*, plus the way
//! those merge into the full [`fARM64::instruction_info`] access set.
//!
//! There is no oracle for implicit-access info, so these expectations are the
//! validation surface: every one is hand-derived from the ARM ARM description
//! of the instruction, and the encodings are cross-checked against `llvm-mc`
//! (the same corpus the rest of the test suite uses).

use fARM64::{implicit_registers, instruction_info, Decoder, DecoderOptions, OpAccess, Register};

/// Decode a single 32-bit word at `ip` with all runtime features accepted.
fn decode_at(word: u32, ip: u64) -> fARM64::Instruction {
    let bytes = word.to_le_bytes();
    let mut dec = Decoder::new(&bytes, ip, DecoderOptions::NONE);
    dec.decode()
}

/// Decode a single 32-bit word at ip 0.
fn decode(word: u32) -> fARM64::Instruction {
    decode_at(word, 0)
}

/// Assert the implicit register set is exactly `expected` (order-independent).
#[track_caller]
fn assert_implicit(word: u32, expected: &[(Register, OpAccess)]) {
    assert_implicit_at(word, 0, expected)
}

/// [`assert_implicit`] at a chosen `ip`.
#[track_caller]
fn assert_implicit_at(word: u32, ip: u64, expected: &[(Register, OpAccess)]) {
    let insn = decode_at(word, ip);
    let imp = implicit_registers(&insn);
    assert_eq!(
        imp.len(),
        expected.len(),
        "implicit count mismatch for {:#010x} ({:?}): got {:?}, expected {:?}",
        word,
        insn.mnemonic(),
        imp.as_slice(),
        expected
    );
    for &(reg, access) in expected {
        assert_eq!(
            imp.access_of(reg),
            access,
            "implicit access mismatch for {reg:?} in {:#010x} ({:?}): got {:?}",
            word,
            insn.mnemonic(),
            imp.as_slice()
        );
    }
}

/// Assert the *full* used-register set (explicit + implicit) is exactly
/// `expected` (order-independent).
#[track_caller]
fn assert_used(word: u32, expected: &[(Register, OpAccess)]) {
    let insn = decode(word);
    let info = instruction_info(&insn);
    let got = info.used_registers();
    assert_eq!(
        got.len(),
        expected.len(),
        "used-register count mismatch for {:#010x} ({:?}): got {:?}, expected {:?}",
        word,
        insn.mnemonic(),
        got,
        expected
    );
    for &(reg, access) in expected {
        match got.iter().find(|u| u.register == reg) {
            Some(u) => assert_eq!(
                u.access,
                access,
                "access mismatch for {reg:?} in {:#010x} ({:?}): got {:?}",
                word,
                insn.mnemonic(),
                got
            ),
            None => panic!(
                "register {reg:?} missing for {:#010x} ({:?}); got {got:?}",
                word,
                insn.mnemonic()
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Link register.
// ---------------------------------------------------------------------------

#[test]
fn direct_and_indirect_calls_write_lr() {
    // bl #4 -> writes the return address to x30.
    assert_implicit(0x9400_0001, &[(Register::X30, OpAccess::Write)]);
    // blr x1 -> same, plus the explicit x1 target.
    assert_implicit(0xD63F_0020, &[(Register::X30, OpAccess::Write)]);
    assert_used(
        0xD63F_0020,
        &[
            (Register::X1, OpAccess::Read),
            (Register::X30, OpAccess::Write),
        ],
    );
}

#[test]
fn ret_reads_lr_implicitly_only_without_an_operand() {
    // `ret` returns through x30 without naming it.
    assert_implicit(0xD65F_03C0, &[(Register::X30, OpAccess::Read)]);
    // `ret x5` names its register: nothing implicit at all.
    assert_implicit(0xD65F_00A0, &[]);
    assert_used(0xD65F_00A0, &[(Register::X5, OpAccess::Read)]);
}

#[test]
fn b_and_br_touch_no_link_register() {
    // b #4 is not a call.
    assert_implicit(0x1400_0001, &[]);
    // br x1 is an indirect branch, not an indirect call.
    assert_implicit(0xD61F_0020, &[]);
}

// ---------------------------------------------------------------------------
// Pointer authentication.
// ---------------------------------------------------------------------------

#[test]
fn paciasp_signs_lr_with_sp() {
    // paciasp (HINT #25): x30 = AddPAC(x30, SP).
    assert_implicit(
        0xD503_233F,
        &[
            (Register::X30, OpAccess::ReadWrite),
            (Register::Sp, OpAccess::Read),
        ],
    );
}

#[test]
fn paciaz_signs_lr_with_a_zero_modifier() {
    // paciaz (HINT #24): x30 = AddPAC(x30, 0) — no SP involved.
    assert_implicit(0xD503_231F, &[(Register::X30, OpAccess::ReadWrite)]);
}

#[test]
fn pacia1716_uses_x17_and_x16() {
    // pacia1716 (HINT #8): x17 = AddPAC(x17, x16).
    assert_implicit(
        0xD503_211F,
        &[
            (Register::X17, OpAccess::ReadWrite),
            (Register::X16, OpAccess::Read),
        ],
    );
}

#[test]
fn xpaclri_strips_the_pac_from_lr() {
    assert_implicit(0xD500_20FF, &[(Register::X30, OpAccess::ReadWrite)]);
}

#[test]
fn retaa_authenticates_lr_against_sp() {
    // retaa: authenticate x30 using SP as the modifier, then return.
    assert_implicit(
        0xD65F_0BFF,
        &[
            (Register::X30, OpAccess::Read),
            (Register::Sp, OpAccess::Read),
        ],
    );
}

#[test]
fn pauth_lr_sppc_forms_fold_in_sp_and_pc() {
    // paciasppc: sign x30 using SP and the PC, neither of them an operand.
    assert_implicit(
        0xDAC1_A3FE,
        &[
            (Register::X30, OpAccess::ReadWrite),
            (Register::Sp, OpAccess::Read),
            (Register::Pc, OpAccess::Read),
        ],
    );
    // autiasppcr x2: the modifier is the explicit x2, so no implicit PC.
    assert_implicit(
        0xDAC1_905E,
        &[
            (Register::X30, OpAccess::ReadWrite),
            (Register::Sp, OpAccess::Read),
        ],
    );
    // The x2 modifier is *read*, not written: the authenticated register is the
    // implicit x30.
    assert_used(
        0xDAC1_905E,
        &[
            (Register::X2, OpAccess::Read),
            (Register::X30, OpAccess::ReadWrite),
            (Register::Sp, OpAccess::Read),
        ],
    );
}

#[test]
fn pauth_lr_returns_are_classified_as_returns() {
    use fARM64::FlowControl;
    // retaasppc <label> and retaasppcr <Xm> both return from a subroutine.
    assert_eq!(
        decode_at(0x5500_001F, 0x1000).flow_control(),
        FlowControl::Return
    );
    assert_eq!(decode(0xD65F_0BE2).flow_control(), FlowControl::Return);
    assert_implicit_at(
        0x5500_001F,
        0x1000,
        &[
            (Register::X30, OpAccess::Read),
            (Register::Sp, OpAccess::Read),
        ],
    );
}

#[test]
fn pac_dataproc_forms_read_modify_their_destination() {
    // pacia x0, x1 -> x0 = AddPAC(x0, x1): the destination is also an input.
    assert_used(
        0xDAC1_0020,
        &[
            (Register::X0, OpAccess::ReadWrite),
            (Register::X1, OpAccess::Read),
        ],
    );
}

// ---------------------------------------------------------------------------
// FEAT_CHK.
// ---------------------------------------------------------------------------

#[test]
fn chkfeat_read_modifies_its_named_x16() {
    // `chkfeat x16` names x16, so nothing is implicit — but x16 is both the
    // feature-request bitmap and the result.
    assert_implicit(0xD500_251F, &[]);
    assert_used(0xD500_251F, &[(Register::X16, OpAccess::ReadWrite)]);
}

// ---------------------------------------------------------------------------
// FEAT_LS64: the eight-register 64-byte transfers.
// ---------------------------------------------------------------------------

#[test]
fn ld64b_writes_eight_consecutive_registers() {
    // ld64b x0, [x1] loads x0..x7; only x0 is spelled.
    assert_implicit(
        0xF83F_D020,
        &[
            (Register::X1, OpAccess::Write),
            (Register::X2, OpAccess::Write),
            (Register::X3, OpAccess::Write),
            (Register::X4, OpAccess::Write),
            (Register::X5, OpAccess::Write),
            (Register::X6, OpAccess::Write),
            (Register::X7, OpAccess::Write),
        ],
    );
    // x1 is both the address base (read) and part of the destination group.
    assert_used(
        0xF83F_D020,
        &[
            (Register::X0, OpAccess::Write),
            (Register::X1, OpAccess::ReadWrite),
            (Register::X2, OpAccess::Write),
            (Register::X3, OpAccess::Write),
            (Register::X4, OpAccess::Write),
            (Register::X5, OpAccess::Write),
            (Register::X6, OpAccess::Write),
            (Register::X7, OpAccess::Write),
        ],
    );
}

#[test]
fn st64bv_reads_eight_and_writes_only_the_status_register() {
    // st64bv x0, x2, [x1]: x0 is the status result, x2..x9 the data source.
    assert_implicit(
        0xF820_B022,
        &[
            (Register::X3, OpAccess::Read),
            (Register::X4, OpAccess::Read),
            (Register::X5, OpAccess::Read),
            (Register::X6, OpAccess::Read),
            (Register::X7, OpAccess::Read),
            (Register::X8, OpAccess::Read),
            (Register::X9, OpAccess::Read),
        ],
    );
    assert_used(
        0xF820_B022,
        &[
            (Register::X0, OpAccess::Write),
            (Register::X1, OpAccess::Read),
            (Register::X2, OpAccess::Read),
            (Register::X3, OpAccess::Read),
            (Register::X4, OpAccess::Read),
            (Register::X5, OpAccess::Read),
            (Register::X6, OpAccess::Read),
            (Register::X7, OpAccess::Read),
            (Register::X8, OpAccess::Read),
            (Register::X9, OpAccess::Read),
        ],
    );
}

// ---------------------------------------------------------------------------
// SVE first-fault register.
// ---------------------------------------------------------------------------

#[test]
#[cfg(feature = "sve")]
fn first_faulting_load_read_modifies_ffr() {
    // ldff1b {z0.b}, p0/z, [x0, x0]
    assert_implicit(0xA400_6000, &[(Register::Ffr, OpAccess::ReadWrite)]);
}

#[test]
#[cfg(feature = "sve")]
fn non_faulting_load_only_reads_ffr() {
    // ldnf1b {z0.b}, p0/z, [x0]
    assert_implicit(0xA410_A000, &[(Register::Ffr, OpAccess::Read)]);
}

#[test]
#[cfg(feature = "sve")]
fn explicit_ffr_moves() {
    // setffr / wrffr p0.b write FFR; rdffr p0.b, p0/z reads it.
    assert_implicit(0x252C_9000, &[(Register::Ffr, OpAccess::Write)]);
    assert_implicit(0x2528_9000, &[(Register::Ffr, OpAccess::Write)]);
    assert_implicit(0x2518_F000, &[(Register::Ffr, OpAccess::Read)]);
    // rdffrs also writes NZCV.
    assert_implicit(
        0x2558_F000,
        &[
            (Register::Ffr, OpAccess::Read),
            (Register::Nzcv, OpAccess::Write),
        ],
    );
    // `wrffr <Pn>.B` has no destination register: its predicate is the source.
    assert_used(
        0x2528_9000,
        &[
            (Register::P0, OpAccess::Read),
            (Register::Ffr, OpAccess::Write),
        ],
    );
}

#[test]
#[cfg(feature = "sve")]
fn a_governing_predicate_is_read_even_on_a_load() {
    // ldnf1b {z0.b}, p0/z, [x0] — z0 is written, p0 governs and is only read.
    assert_used(
        0xA410_A000,
        &[
            (Register::Z0, OpAccess::Write),
            (Register::P0, OpAccess::Read),
            (Register::X0, OpAccess::Read),
            (Register::Ffr, OpAccess::Read),
        ],
    );
}

// ---------------------------------------------------------------------------
// SME ZA array.
// ---------------------------------------------------------------------------

#[test]
fn smstart_and_smstop_invalidate_za_only_when_they_name_it() {
    // Bare `smstart`/`smstop` change both PSTATE.SM and PSTATE.ZA.
    assert_implicit(0xD503_417F, &[(Register::Za, OpAccess::Write)]);
    assert_implicit(0xD503_407F, &[(Register::Za, OpAccess::Write)]);
    // `smstart za` / `smstop za` name the array explicitly.
    assert_implicit(0xD503_457F, &[(Register::Za, OpAccess::Write)]);
    assert_implicit(0xD503_447F, &[(Register::Za, OpAccess::Write)]);
    // `smstart sm` / `smstop sm` touch streaming mode only — ZA is untouched.
    assert_implicit(0xD503_437F, &[]);
    assert_implicit(0xD503_427F, &[]);
}

#[test]
#[cfg(feature = "sme")]
fn zero_za_writes_the_array() {
    // zero { za } — the mask names ZA tiles, reported through the ZA
    // pseudo-register.
    assert_implicit(0xC008_00FF, &[]);
    assert_used(0xC008_00FF, &[(Register::Za, OpAccess::Write)]);
}

// ---------------------------------------------------------------------------
// PC as a data input.
// ---------------------------------------------------------------------------

#[test]
fn pc_relative_address_generation_reads_pc() {
    // adr x0, #0
    assert_implicit_at(0x1000_0000, 0x1000, &[(Register::Pc, OpAccess::Read)]);
    // ldr w0, <literal>
    assert_implicit_at(0x1800_0020, 0x1000, &[(Register::Pc, OpAccess::Read)]);
}

#[test]
fn hbc_conditional_branch_is_classified_as_a_branch() {
    use fARM64::{Code, FlowControl};
    // `bc.ne 0x110c` @ 0x1000 — the FEAT_HBC hinted conditional branch carries
    // its own mnemonic, so it needs its own flow-control arm. Before that it
    // fell through to `Next`, which lost its branch target and made the access
    // analysis report a spurious PC read.
    let insn = decode_at(0x5400_0871, 0x1000);
    assert_eq!(insn.code(), Code::BcCond);
    assert_eq!(insn.flow_control(), FlowControl::ConditionalBranch);
    assert_eq!(insn.near_branch_target(), 0x110C);
    assert_implicit_at(0x5400_0871, 0x1000, &[(Register::Nzcv, OpAccess::Read)]);
}

#[test]
fn ordinary_branches_do_not_report_a_pc_read() {
    // Branch-target formation reads PC architecturally, but reporting it would
    // put PC on every branch in the program; the resolved target is enough.
    assert_implicit_at(0x1400_0001, 0x1000, &[]);
    // b.eq still reports its NZCV read — just not a PC read.
    assert_implicit_at(0x5400_0040, 0x1000, &[(Register::Nzcv, OpAccess::Read)]);
}

// ---------------------------------------------------------------------------
// NZCV.
// ---------------------------------------------------------------------------

#[test]
fn nzcv_appears_as_a_pseudo_register() {
    // adds writes the flags; adcs reads the carry and writes them; csel only
    // reads them.
    assert_implicit(0xAB02_0020, &[(Register::Nzcv, OpAccess::Write)]);
    assert_implicit(0xBA02_0020, &[(Register::Nzcv, OpAccess::ReadWrite)]);
    assert_implicit(0x9A82_1020, &[(Register::Nzcv, OpAccess::Read)]);
    // add touches no flags at all.
    assert_implicit(0x8B02_0020, &[]);
}

#[test]
fn flag_manipulation_forms_read_modify_nzcv() {
    // cfinv inverts C and preserves N/Z/V.
    assert_implicit(0xD500_401F, &[(Register::Nzcv, OpAccess::ReadWrite)]);
    // rmif inserts a rotated bitfield of Xn into the mask-selected flags; Xn is
    // the source, not a destination.
    assert_implicit(0xBA00_0400, &[(Register::Nzcv, OpAccess::ReadWrite)]);
    assert_used(
        0xBA00_0400,
        &[
            (Register::X0, OpAccess::Read),
            (Register::Nzcv, OpAccess::ReadWrite),
        ],
    );
}

#[test]
fn nzcv_agrees_with_the_scalar_flag_booleans() {
    for w in [
        0xAB02_0020u32,
        0xBA02_0020,
        0x9A82_1020,
        0x8B02_0020,
        0xD500_401F,
        0xEB02_003F,
    ] {
        let insn = decode(w);
        let info = instruction_info(&insn);
        let imp = implicit_registers(&insn);
        assert_eq!(
            imp.reads(Register::Nzcv),
            info.flags_read(),
            "flags_read disagrees with the NZCV pseudo-register for {w:#010x}"
        );
        assert_eq!(
            imp.writes(Register::Nzcv),
            info.flags_written(),
            "flags_written disagrees with the NZCV pseudo-register for {w:#010x}"
        );
    }
}

// ---------------------------------------------------------------------------
// MOPS.
// ---------------------------------------------------------------------------

#[test]
fn mops_writeback_operands_are_read_modified() {
    // cpyp [x2]!, [x0]!, x1! — destination pointer, source pointer and count
    // are all consumed and written back.
    assert_used(
        0x1D00_0422,
        &[
            (Register::X2, OpAccess::ReadWrite),
            (Register::X0, OpAccess::ReadWrite),
            (Register::X1, OpAccess::ReadWrite),
        ],
    );
    // setp [x2]!, x1!, x0 — the third operand is the plain fill value, read only.
    assert_used(
        0x19C0_0422,
        &[
            (Register::X2, OpAccess::ReadWrite),
            (Register::X1, OpAccess::ReadWrite),
            (Register::X0, OpAccess::Read),
        ],
    );
}

// ---------------------------------------------------------------------------
// General properties.
// ---------------------------------------------------------------------------

#[test]
fn plain_dataproc_has_no_implicit_state() {
    for w in [
        0x8B02_0020u32, // add x0, x1, x2
        0xD280_0020,    // movz x0, #1
        0xF940_0420,    // ldr x0, [x1, #8]
        0xD503_2BFF,    // hint #0x5f
    ] {
        assert_implicit(w, &[]);
    }
}

#[test]
fn accessors_agree_with_the_slice() {
    let insn = decode(0xD503_233F); // paciasp
    let imp = implicit_registers(&insn);
    assert_eq!(imp.len(), imp.as_slice().len());
    assert!(!imp.is_empty());
    assert!(imp.reads(Register::X30) && imp.writes(Register::X30));
    assert!(imp.reads(Register::Sp) && !imp.writes(Register::Sp));
    assert_eq!(imp.access_of(Register::X9), OpAccess::None);
    assert!(!imp.reads(Register::X9) && !imp.writes(Register::X9));
    // The method on `Instruction` is the same computation.
    assert_eq!(insn.implicit_registers().as_slice(), imp.as_slice());
    // `IntoIterator` over a reference yields the same entries.
    assert_eq!((&imp).into_iter().count(), imp.len());
}

#[test]
fn implicit_is_a_subset_of_used_across_a_sweep() {
    // Every implicit access must appear in the full used-register set, with an
    // access that at least covers it. Sweeps a dense slice of the encoding
    // space rather than hand-listing instructions.
    for w in (0u32..=0x00FF_FFFF).step_by(2477) {
        for base in [0x1400_0000u32, 0x8B00_0000, 0xF800_0000, 0xD500_0000] {
            let word = base | w;
            let insn = decode_at(word, 0x1000);
            if insn.is_invalid() {
                continue;
            }
            let imp = implicit_registers(&insn);
            let info = instruction_info(&insn);
            for entry in imp.iter() {
                let found = info
                    .used_registers()
                    .iter()
                    .find(|u| u.register == entry.register);
                let found = match found {
                    Some(f) => f,
                    None => panic!(
                        "implicit {:?} missing from used_registers for {word:#010x} ({:?})",
                        entry,
                        insn.mnemonic()
                    ),
                };
                let covered = found.access == entry.access || found.access == OpAccess::ReadWrite;
                assert!(
                    covered,
                    "implicit {:?} not covered by {:?} for {word:#010x} ({:?})",
                    entry,
                    found,
                    insn.mnemonic()
                );
            }
        }
    }
}
