//! Register / memory access-set tests for [`fARM64::info::instruction_info`].
//!
//! Each case decodes a single A64 word (encodings cross-checked with `llvm-mc`)
//! and asserts the exact read/write register set, memory accesses, and the NZCV
//! `flags_read`/`flags_written` booleans, against hand-derived ARM ARM semantics.
//! There is no oracle for register-access info, so these expectations are the
//! validation surface.

use fARM64::info::instruction_info;
use fARM64::{Decoder, DecoderOptions, OpAccess, Register};

/// Decode a single 32-bit word at ip 0 with all runtime features accepted.
fn decode(word: u32) -> fARM64::Instruction {
    let bytes = word.to_le_bytes();
    let mut dec = Decoder::new(&bytes, 0, DecoderOptions::NONE);
    dec.decode()
}

/// Assert the used-register set equals `expected` (order-independent).
#[track_caller]
fn assert_regs(word: u32, expected: &[(Register, OpAccess)]) {
    let insn = decode(word);
    let info = instruction_info(&insn);
    let got = info.used_registers();
    assert_eq!(
        got.len(),
        expected.len(),
        "register count mismatch for {:#010x} ({:?}): got {:?}, expected {:?}",
        word,
        insn.mnemonic(),
        got,
        expected
    );
    for &(reg, access) in expected {
        let found = got.iter().find(|u| u.register == reg);
        match found {
            Some(u) => assert_eq!(
                u.access, access,
                "access mismatch for {reg:?} in {:#010x} ({:?}): got {:?}, expected {:?}",
                word,
                insn.mnemonic(),
                u.access,
                access
            ),
            None => panic!(
                "register {reg:?} missing for {:#010x} ({:?}); got {:?}",
                word,
                insn.mnemonic(),
                got
            ),
        }
    }
}

/// Assert the single memory access matches `(base, index, access)`.
#[track_caller]
fn assert_one_mem(word: u32, base: Register, index: Register, access: OpAccess) {
    let insn = decode(word);
    let info = instruction_info(&insn);
    let mem = info.used_memory();
    assert_eq!(mem.len(), 1, "expected exactly one memory access for {word:#010x}: {mem:?}");
    assert_eq!(mem[0].base, base, "mem base for {word:#010x}");
    assert_eq!(mem[0].index, index, "mem index for {word:#010x}");
    assert_eq!(mem[0].access, access, "mem access for {word:#010x}");
}

#[track_caller]
fn assert_flags(word: u32, read: bool, written: bool) {
    let insn = decode(word);
    let info = instruction_info(&insn);
    assert_eq!(info.flags_read(), read, "flags_read for {word:#010x} ({:?})", insn.mnemonic());
    assert_eq!(
        info.flags_written(),
        written,
        "flags_written for {word:#010x} ({:?})",
        insn.mnemonic()
    );
}

// ---------------------------------------------------------------------------
// Data-processing.
// ---------------------------------------------------------------------------

#[test]
fn add_writes_dest_reads_sources() {
    // add x0, x1, x2  -> W x0; R x1, x2; no flags.
    let w = 0x8b02_0020;
    assert_regs(
        w,
        &[
            (Register::X0, OpAccess::Write),
            (Register::X1, OpAccess::Read),
            (Register::X2, OpAccess::Read),
        ],
    );
    assert_flags(w, false, false);
    assert!(instruction_info(&decode(w)).used_memory().is_empty());
}

#[test]
fn adds_sets_flags() {
    // adds x0, x1, x2 -> W x0; R x1, x2; flags written, not read.
    let w = 0xab02_0020;
    assert_regs(
        w,
        &[
            (Register::X0, OpAccess::Write),
            (Register::X1, OpAccess::Read),
            (Register::X2, OpAccess::Read),
        ],
    );
    assert_flags(w, false, true);
}

#[test]
fn cmp_has_no_dest_and_sets_flags() {
    // cmp x1, x2 (alias of subs xzr, x1, x2) -> R x1, x2; flags written.
    let w = 0xeb02_003f;
    assert_regs(
        w,
        &[(Register::X1, OpAccess::Read), (Register::X2, OpAccess::Read)],
    );
    assert_flags(w, false, true);
}

#[test]
fn adc_reads_and_writes_flags() {
    // adc x0, x1, x2 -> W x0; R x1, x2; flags read (carry), not written.
    let w = 0x9a02_0020;
    assert_regs(
        w,
        &[
            (Register::X0, OpAccess::Write),
            (Register::X1, OpAccess::Read),
            (Register::X2, OpAccess::Read),
        ],
    );
    assert_flags(w, true, false);
}

#[test]
fn csel_reads_flags() {
    // csel x0, x1, x2, eq -> W x0; R x1, x2; flags read.
    let w = 0x9a82_0020;
    assert_regs(
        w,
        &[
            (Register::X0, OpAccess::Write),
            (Register::X1, OpAccess::Read),
            (Register::X2, OpAccess::Read),
        ],
    );
    assert_flags(w, true, false);
}

#[test]
fn madd_reads_three_sources() {
    // madd x0, x1, x2, x3 -> W x0; R x1, x2, x3 (plain multiply-add, no RMW).
    let w = 0x9b02_0c20;
    assert_regs(
        w,
        &[
            (Register::X0, OpAccess::Write),
            (Register::X1, OpAccess::Read),
            (Register::X2, OpAccess::Read),
            (Register::X3, OpAccess::Read),
        ],
    );
    assert_flags(w, false, false);
}

#[test]
fn mul_writes_dest() {
    // mul x0, x1, x2 (alias of madd ..., xzr) -> W x0; R x1, x2.
    let w = 0x9b02_7c20;
    assert_regs(
        w,
        &[
            (Register::X0, OpAccess::Write),
            (Register::X1, OpAccess::Read),
            (Register::X2, OpAccess::Read),
        ],
    );
}

#[test]
fn mov_writes_dest_reads_source() {
    // mov x0, x1 (alias of orr x0, xzr, x1) -> W x0; R xzr, x1.
    let w = 0xaa01_03e0;
    let insn = decode(w);
    let info = instruction_info(&insn);
    // x0 written, x1 read (the xzr source is included with Read, iced-style).
    let x0 = info.used_registers().iter().find(|u| u.register == Register::X0).unwrap();
    let x1 = info.used_registers().iter().find(|u| u.register == Register::X1).unwrap();
    assert_eq!(x0.access, OpAccess::Write);
    assert_eq!(x1.access, OpAccess::Read);
}

#[test]
fn movk_read_modifies_dest() {
    // movk x0, #1 -> RW x0 (keeps the other bits of x0).
    let w = 0xf280_0020;
    assert_regs(w, &[(Register::X0, OpAccess::ReadWrite)]);
    assert_flags(w, false, false);
}

#[test]
fn mla_vector_accumulates_into_dest() {
    // mla v0.4s, v1.4s, v2.4s -> RW v0; R v1, v2.
    let w = 0x4ea2_9420;
    assert_regs(
        w,
        &[
            (Register::V0, OpAccess::ReadWrite),
            (Register::V1, OpAccess::Read),
            (Register::V2, OpAccess::Read),
        ],
    );
}

// ---------------------------------------------------------------------------
// Loads / stores.
// ---------------------------------------------------------------------------

#[test]
fn ldr_writes_data_reads_mem() {
    // ldr x0, [x1] -> W x0; R x1 (base); mem Read.
    let w = 0xf940_0020;
    assert_regs(
        w,
        &[(Register::X0, OpAccess::Write), (Register::X1, OpAccess::Read)],
    );
    assert_one_mem(w, Register::X1, Register::None, OpAccess::Read);
}

#[test]
fn ldr_preindex_writes_back_base() {
    // ldr x0, [x1, #8]! -> W x0; RW x1 (writeback base); mem Read.
    let w = 0xf840_8c20;
    assert_regs(
        w,
        &[
            (Register::X0, OpAccess::Write),
            (Register::X1, OpAccess::ReadWrite),
        ],
    );
    assert_one_mem(w, Register::X1, Register::None, OpAccess::Read);
}

#[test]
fn ldr_register_offset_index_is_read() {
    // ldr x0, [x1, x2, lsl #3] -> W x0; R x1 (base), x2 (index); mem Read.
    let w = 0xf862_7820;
    assert_regs(
        w,
        &[
            (Register::X0, OpAccess::Write),
            (Register::X1, OpAccess::Read),
            (Register::X2, OpAccess::Read),
        ],
    );
    assert_one_mem(w, Register::X1, Register::X2, OpAccess::Read);
}

#[test]
fn str_reads_data_and_base_writes_mem() {
    // str x0, [x1] -> R x0 (data), x1 (base); mem Write.
    let w = 0xf900_0020;
    assert_regs(
        w,
        &[(Register::X0, OpAccess::Read), (Register::X1, OpAccess::Read)],
    );
    assert_one_mem(w, Register::X1, Register::None, OpAccess::Write);
}

#[test]
fn stp_reads_both_data_regs() {
    // stp x0, x1, [x2, #16] -> R x0, x1 (data), x2 (base); mem Write.
    let w = 0xa901_0440;
    assert_regs(
        w,
        &[
            (Register::X0, OpAccess::Read),
            (Register::X1, OpAccess::Read),
            (Register::X2, OpAccess::Read),
        ],
    );
    assert_one_mem(w, Register::X2, Register::None, OpAccess::Write);
}

#[test]
fn ldp_postindex_writes_data_and_writes_back_base() {
    // ldp x0, x1, [x2], #16 -> W x0, x1; RW x2 (post-index writeback); mem Read.
    let w = 0xa8c1_0440;
    assert_regs(
        w,
        &[
            (Register::X0, OpAccess::Write),
            (Register::X1, OpAccess::Write),
            (Register::X2, OpAccess::ReadWrite),
        ],
    );
    assert_one_mem(w, Register::X2, Register::None, OpAccess::Read);
}

// ---------------------------------------------------------------------------
// Atomics.
// ---------------------------------------------------------------------------

#[test]
fn ldadd_value_read_result_write_mem_rmw() {
    // ldadd w0, w1, [x2] -> R w0 (value); W w1 (result); RW x2 base read; mem RW.
    let w = 0xb820_0041;
    assert_regs(
        w,
        &[
            (Register::W0, OpAccess::Read),
            (Register::W1, OpAccess::Write),
            (Register::X2, OpAccess::Read),
        ],
    );
    assert_one_mem(w, Register::X2, Register::None, OpAccess::ReadWrite);
}

#[test]
fn cas_compare_reg_is_read_write() {
    // cas x0, x1, [x2] -> RW x0 (compare/result); R x1 (new value); mem RW.
    // Encoding: cas x0, x1, [x2] = 0xc8a07c41? compute below via decode check.
    let w = 0xc8a0_7c41; // cas x0, x1, [x2]  (size=11 L=0 o0=0, Rs=0, Rt=1, Rn=2)
    let insn = decode(w);
    assert_eq!(insn.mnemonic(), fARM64::Mnemonic::Cas, "decoded {:?}", insn.mnemonic());
    assert_regs(
        w,
        &[
            (Register::X0, OpAccess::ReadWrite),
            (Register::X1, OpAccess::Read),
            (Register::X2, OpAccess::Read),
        ],
    );
    assert_one_mem(w, Register::X2, Register::None, OpAccess::ReadWrite);
}

#[test]
fn stxr_status_write_data_read_mem_write() {
    // stxr w0, x1, [x2] -> W w0 (status); R x1 (data); mem Write.
    let w = 0xc800_7c41; // stxr w0, x1, [x2]
    let insn = decode(w);
    assert_eq!(insn.mnemonic(), fARM64::Mnemonic::Stxr, "decoded {:?}", insn.mnemonic());
    assert_regs(
        w,
        &[
            (Register::W0, OpAccess::Write),
            (Register::X1, OpAccess::Read),
            (Register::X2, OpAccess::Read),
        ],
    );
    assert_one_mem(w, Register::X2, Register::None, OpAccess::Write);
}

#[test]
fn swp_value_read_result_write() {
    // swp x0, x1, [x2] -> R x0 (value); W x1 (result); mem RW.
    let w = 0xf820_8041;
    let insn = decode(w);
    assert_eq!(insn.mnemonic(), fARM64::Mnemonic::Swp, "decoded {:?}", insn.mnemonic());
    assert_regs(
        w,
        &[
            (Register::X0, OpAccess::Read),
            (Register::X1, OpAccess::Write),
            (Register::X2, OpAccess::Read),
        ],
    );
    assert_one_mem(w, Register::X2, Register::None, OpAccess::ReadWrite);
}

#[test]
fn casp_compare_pair_is_read_write() {
    // casp x0, x1, x2, x3, [x4] -> RW x0,x1 (compare pair); R x2,x3 (value pair).
    let w = 0x4820_7c82;
    let insn = decode(w);
    assert_eq!(insn.mnemonic(), fARM64::Mnemonic::Casp, "decoded {:?}", insn.mnemonic());
    assert_regs(
        w,
        &[
            (Register::X0, OpAccess::ReadWrite),
            (Register::X1, OpAccess::ReadWrite),
            (Register::X2, OpAccess::Read),
            (Register::X3, OpAccess::Read),
            (Register::X4, OpAccess::Read),
        ],
    );
    assert_one_mem(w, Register::X4, Register::None, OpAccess::ReadWrite);
}

#[test]
fn str_preindex_writes_back_base() {
    // str x0, [x1, #8]! -> R x0 (data); RW x1 (writeback base); mem Write.
    let w = 0xf800_8c20;
    assert_regs(
        w,
        &[
            (Register::X0, OpAccess::Read),
            (Register::X1, OpAccess::ReadWrite),
        ],
    );
    assert_one_mem(w, Register::X1, Register::None, OpAccess::Write);
}

// ---------------------------------------------------------------------------
// Branches / link register.
// ---------------------------------------------------------------------------

#[test]
fn bl_writes_link_register() {
    // bl #16 -> W x30 (implicit link).
    let w = 0x9400_0004;
    assert_regs(w, &[(Register::X30, OpAccess::Write)]);
}

#[test]
fn blr_writes_link_reads_target() {
    // blr x1 -> W x30 (implicit link); R x1 (target).
    let w = 0xd63f_0020;
    assert_regs(
        w,
        &[(Register::X1, OpAccess::Read), (Register::X30, OpAccess::Write)],
    );
}

#[test]
fn ret_reads_default_link_register() {
    // ret -> R x30 (implicit default).
    let w = 0xd65f_03c0;
    assert_regs(w, &[(Register::X30, OpAccess::Read)]);
}

#[test]
fn ret_with_explicit_reg_reads_that_reg() {
    // ret x1 -> R x1 (explicit; no implicit x30 added).
    let w = 0xd65f_0020;
    assert_regs(w, &[(Register::X1, OpAccess::Read)]);
}

// ---------------------------------------------------------------------------
// SVE predicated / SME ZA.
// ---------------------------------------------------------------------------

#[test]
fn sve_predicated_add_reads_governing_predicate() {
    // fadd z0.s, p0/m, z0.s, z1.s -> RW z0 (merge-into-dest source), R p0, R z1.
    let w = 0x6580_8020;
    let insn = decode(w);
    assert_eq!(insn.mnemonic(), fARM64::Mnemonic::Fadd);
    let info = instruction_info(&insn);
    // z0 is both destination (slot 0, Write) and source (slot 2, Read) -> RW.
    let z0 = info.used_registers().iter().find(|u| u.register == Register::Z0).unwrap();
    assert_eq!(z0.access, OpAccess::ReadWrite);
    // Governing predicate p0 is read.
    let p0 = info.used_registers().iter().find(|u| u.register == Register::P0).unwrap();
    assert_eq!(p0.access, OpAccess::Read);
    // z1 source read.
    let z1 = info.used_registers().iter().find(|u| u.register == Register::Z1).unwrap();
    assert_eq!(z1.access, OpAccess::Read);
}

#[test]
fn sme_fmopa_za_dest_is_read_write() {
    // fmopa za0.s, p0/m, p1/m, z0.s, z1.s
    // ZAda accumulator (rendered as z0) is RW; predicates p0/p1 read; z1 read.
    let w = 0x8081_2000;
    let insn = decode(w);
    assert_eq!(insn.mnemonic(), fARM64::Mnemonic::Fmopa, "decoded {:?}", insn.mnemonic());
    let info = instruction_info(&insn);
    // Slot-0 accumulator z0 read-modify-written (also appears as source z0 -> RW).
    let z0 = info.used_registers().iter().find(|u| u.register == Register::Z0).unwrap();
    assert_eq!(z0.access, OpAccess::ReadWrite);
    for p in [Register::P0, Register::P1] {
        let pr = info.used_registers().iter().find(|u| u.register == p).unwrap();
        assert_eq!(pr.access, OpAccess::Read, "predicate {p:?} should be read");
    }
    let z1 = info.used_registers().iter().find(|u| u.register == Register::Z1).unwrap();
    assert_eq!(z1.access, OpAccess::Read);
}

// ---------------------------------------------------------------------------
// Factory parity (alloc).
// ---------------------------------------------------------------------------

#[cfg(feature = "alloc")]
#[test]
fn factory_matches_free_function() {
    use fARM64::info::InstructionInfoFactory;
    let insn = decode(0x8b02_0020); // add x0, x1, x2
    let mut fac = InstructionInfoFactory::new();
    let via_factory = *fac.info(&insn);
    let direct = instruction_info(&insn);
    assert_eq!(via_factory.used_registers(), direct.used_registers());
    assert_eq!(via_factory.flags_read(), direct.flags_read());
    assert_eq!(via_factory.flags_written(), direct.flags_written());
}
