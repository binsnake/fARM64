//! Editing + re-encoding tests: decode an instruction, change its operands,
//! and encode the edited semantics back to a word.
//!
//! The encoder never reads [`fARM64::Instruction::word`], so a word produced
//! after an edit is genuinely derived from the edited operands. Each expected
//! word below is the `llvm-mc` encoding of the *edited* instruction, and most
//! cases additionally re-decode the result and compare the projection, so a
//! wrong expectation cannot pass by accident.

use fARM64::{
    Code, Condition, Decoder, DecoderOptions, EncodeError, MemIndexMode, Mnemonic, OpKind, Operand,
    Register,
};

/// Decode a single word at `ip` with all runtime features accepted.
fn decode_at(word: u32, ip: u64) -> fARM64::Instruction {
    let bytes = word.to_le_bytes();
    let mut dec = Decoder::new(&bytes, ip, DecoderOptions::NONE);
    dec.decode()
}

/// Decode a single word at ip 0.
fn decode(word: u32) -> fARM64::Instruction {
    decode_at(word, 0)
}

/// Assert `insn` encodes to `expected`, and that re-decoding `expected` at the
/// same `ip` yields an equal instruction — i.e. the edit landed in a word that
/// really means what the edited operands say.
#[track_caller]
fn assert_encodes_and_round_trips(insn: &fARM64::Instruction, expected: u32) {
    assert_eq!(
        insn.encode(),
        Ok(expected),
        "encode mismatch for edited {:?}",
        insn.mnemonic()
    );
    let back = decode_at(expected, insn.ip());
    assert_eq!(
        back.code(),
        insn.code(),
        "re-decoded code mismatch for {expected:#010x}"
    );
    assert_eq!(
        back.op_count(),
        insn.op_count(),
        "re-decoded operand count mismatch for {expected:#010x}"
    );
    for i in 0..insn.op_count() {
        assert_eq!(
            back.op(i),
            insn.op(i),
            "re-decoded operand {i} mismatch for {expected:#010x}"
        );
    }
}

// ---------------------------------------------------------------------------
// Registers.
// ---------------------------------------------------------------------------

#[test]
fn replace_register_operands() {
    // add x0, x1, x2  ->  add x5, x1, x7
    let mut insn = decode(0x8B02_0020);
    assert!(!insn.is_modified());
    assert!(insn.set_op_register(0, Register::X5));
    assert!(insn.set_op_register(2, Register::X7));
    assert!(insn.is_modified());
    assert_encodes_and_round_trips(&insn, 0x8B07_0025);
}

#[test]
fn replacing_a_register_keeps_its_decorations() {
    // add x0, x1, x2, lsl #3 — the shift decoration must survive the swap.
    let mut insn = decode(0x8B02_0C20);
    assert!(insn.set_op_register(2, Register::X9));
    match insn.op(2) {
        Operand::Reg { reg, shift, .. } => {
            assert_eq!(reg, Register::X9);
            assert!(shift.is_some(), "shift decoration was dropped");
        }
        other => panic!("unexpected operand shape: {other:?}"),
    }
    assert_encodes_and_round_trips(&insn, 0x8B09_0C20);
}

#[test]
fn register_class_and_width_are_checked() {
    let mut insn = decode(0x8B02_0020); // add x0, x1, x2
    let before = insn;

    // A 32-bit view where a 64-bit one was: the *encoding* carries the size.
    assert!(!insn.set_op_register(0, Register::W5));
    // A vector register where a GP one was.
    assert!(!insn.set_op_register(0, Register::V5));
    // A slot past the operand list.
    assert!(!insn.set_op_register(9, Register::X5));
    // Nothing changed, and the instruction was never marked modified.
    assert_eq!(insn, before);
    assert!(!insn.is_modified());

    // SP and XZR are both 64-bit GP, so swapping between them is allowed.
    assert!(insn.set_op_register(1, Register::Sp));
    assert_eq!(insn.op_register(1), Register::Sp);
}

#[test]
fn unchecked_replacement_plus_set_code_changes_the_operand_width() {
    // add w0, w1, w2  ->  add x0, x1, x2 (a different Code, so set_code too).
    let mut insn = decode(0x0B02_0020);
    insn.set_code(Code::AddShifted64);
    for (slot, reg) in [(0, Register::X0), (1, Register::X1), (2, Register::X2)] {
        assert!(insn.set_op_register_unchecked(slot, reg));
    }
    assert_encodes_and_round_trips(&insn, 0x8B02_0020);
}

#[test]
fn register_lists_are_not_editable_piecemeal() {
    // ld1 { v0.16b }, [x0] — a MultiReg list has no single register to replace.
    let mut insn = decode(0x4C40_7000);
    assert_eq!(insn.op_kind(0), OpKind::MultiReg);
    assert!(!insn.set_op_register(0, Register::V3));
    assert!(!insn.is_modified());
}

// ---------------------------------------------------------------------------
// Immediates.
// ---------------------------------------------------------------------------

#[test]
fn replace_an_unsigned_immediate() {
    // add x0, x1, #1  ->  add x0, x1, #0x20
    let mut insn = decode(0x9100_0420);
    assert!(insn.set_op_immediate(2, 0x20));
    assert_encodes_and_round_trips(&insn, 0x9100_8020);
}

#[test]
fn replace_a_resolved_move_immediate() {
    // `movz x0, #1` is disassembled as the `MOV (wide immediate)` alias, whose
    // immediate is already fully resolved: mov x0, #1 -> mov x0, #0x1234.
    let mut insn = decode(0xD280_0020);
    assert_eq!(insn.mnemonic(), Mnemonic::Mov);
    assert_eq!(insn.op_kind(1), OpKind::ImmSigned);
    assert!(insn.set_op_immediate(1, 0x1234));
    assert_encodes_and_round_trips(&insn, 0xD282_4680);
}

#[test]
fn replace_a_shifted_wide_move_immediate() {
    // movk x0, #0x1234, lsl #16  ->  movk x0, #0x5678, lsl #16. The shift is
    // part of the operand and must survive the edit.
    let mut insn = decode(0xF2A2_4680);
    assert_eq!(insn.op_kind(1), OpKind::ImmShiftedMove);
    assert!(insn.set_op_immediate(1, 0x5678));
    match insn.op(1) {
        Operand::ImmShiftedMove { imm, lsl } => {
            assert_eq!(imm, 0x5678);
            assert_eq!(lsl, 16);
        }
        other => panic!("unexpected operand shape: {other:?}"),
    }
    assert_encodes_and_round_trips(&insn, 0xF2AA_CF00);

    // A value too wide for the 16-bit field is refused outright.
    let before = insn;
    assert!(!insn.set_op_immediate(1, 0x1_0000));
    assert_eq!(insn, before);
}

#[test]
fn replace_a_logical_immediate() {
    // and x0, x1, #0xff  ->  and x0, x1, #0xffff
    let mut insn = decode(0x9240_1C20);
    assert_eq!(insn.op_kind(2), OpKind::ImmLogical);
    assert!(insn.set_op_immediate(2, 0xFFFF));
    assert_encodes_and_round_trips(&insn, 0x9240_3C20);
}

#[test]
fn a_logical_immediate_with_no_encoding_is_rejected_by_the_encoder() {
    // 0x5 is not a representable bitmask immediate; the *edit* still succeeds —
    // validation happens at encode time.
    let mut insn = decode(0x9240_1C20);
    assert!(insn.set_op_immediate(2, 0x5));
    assert_eq!(insn.encode(), Err(EncodeError::InvalidImmediate));
}

#[test]
fn immediate_setters_refuse_the_wrong_operand_kind() {
    let mut insn = decode(0x8B02_0020); // add x0, x1, x2 — no immediate
    let before = insn;
    assert!(!insn.set_op_immediate(2, 7));
    assert!(!insn.set_op_immediate(9, 7));
    assert_eq!(insn, before);
    assert!(!insn.is_modified());
}

// ---------------------------------------------------------------------------
// Memory operands.
// ---------------------------------------------------------------------------

#[test]
fn replace_a_memory_base_and_displacement() {
    // ldr x0, [x1, #8]  ->  ldr x0, [x3, #16]
    let mut insn = decode(0xF940_0420);
    assert!(insn.set_memory_base(Register::X3));
    assert!(insn.set_memory_displacement64(16));
    assert_eq!(insn.memory_base(), Register::X3);
    assert_eq!(insn.memory_displacement64(), 16);
    assert_encodes_and_round_trips(&insn, 0xF940_0860);
}

#[test]
fn a_displacement_the_encoding_cannot_scale_is_rejected_by_the_encoder() {
    // The 64-bit unsigned-offset LDR scales by 8; 12 is not a multiple of 8.
    let mut insn = decode(0xF940_0420);
    assert!(insn.set_memory_displacement64(12));
    assert_eq!(insn.encode(), Err(EncodeError::InvalidImmediate));
}

#[test]
fn replace_a_memory_index() {
    // ldr x0, [x1, x2, lsl #3]  ->  ldr x0, [x1, x5, lsl #3]
    let mut insn = decode(0xF862_7820);
    assert!(insn.set_memory_index(Register::X5));
    assert_eq!(insn.memory_index(), Register::X5);
    assert_eq!(insn.memory_index_scale(), 3);
    assert_encodes_and_round_trips(&insn, 0xF865_7820);
}

#[test]
fn the_addressing_mode_survives_a_base_change() {
    // ldr x0, [x1, #8]! — pre-index writeback must be preserved.
    let mut insn = decode(0xF840_8C20);
    assert!(insn.set_memory_base(Register::X3));
    match insn.op(1) {
        Operand::MemImm { base, mode, .. } => {
            assert_eq!(base, Register::X3);
            assert_eq!(mode, MemIndexMode::PreIndex);
        }
        other => panic!("unexpected operand shape: {other:?}"),
    }
    assert_encodes_and_round_trips(&insn, 0xF840_8C60);
}

#[test]
fn memory_setters_refuse_a_non_memory_instruction() {
    let mut insn = decode(0x8B02_0020); // add x0, x1, x2
    assert!(!insn.set_memory_base(Register::X3));
    assert!(!insn.set_memory_index(Register::X3));
    assert!(!insn.set_memory_displacement64(8));
    assert!(!insn.is_modified());
}

// ---------------------------------------------------------------------------
// Branch targets, ip and relocation.
// ---------------------------------------------------------------------------

#[test]
fn retarget_a_direct_branch() {
    // b 0x1004 @ 0x1000  ->  b 0x1020
    let mut insn = decode_at(0x1400_0001, 0x1000);
    assert_eq!(insn.near_branch_target(), 0x1004);
    assert!(insn.set_near_branch_target(0x1020));
    assert_eq!(insn.near_branch_target(), 0x1020);
    assert_encodes_and_round_trips(&insn, 0x1400_0008);
}

#[test]
fn retarget_a_conditional_branch_and_flip_its_condition() {
    // b.eq 0x1008 @ 0x1000  ->  b.ne 0x1000
    let mut insn = decode_at(0x5400_0040, 0x1000);
    assert_eq!(insn.condition(), Some(Condition::Eq));
    assert!(insn.set_condition(Condition::Ne));
    assert!(insn.set_near_branch_target(0x1000));
    assert_eq!(insn.condition(), Some(Condition::Ne));
    assert_encodes_and_round_trips(&insn, 0x5400_0001);
}

#[test]
fn retarget_an_hbc_conditional_branch() {
    // bc.ne 0x110c @ 0x1000  ->  bc.ne 0x1000
    let mut insn = decode_at(0x5400_0871, 0x1000);
    assert_eq!(insn.code(), Code::BcCond);
    assert_eq!(insn.near_branch_target(), 0x110C);
    assert!(insn.set_near_branch_target(0x1000));
    assert_encodes_and_round_trips(&insn, 0x5400_0011);
}

#[test]
fn an_out_of_range_or_misaligned_target_is_an_encode_error() {
    let mut insn = decode_at(0x1400_0001, 0x1000);
    // `B` reaches +/-128 MiB; 256 MiB away is out of range.
    assert!(insn.set_near_branch_target(0x1000_0000));
    assert_eq!(insn.encode(), Err(EncodeError::InvalidImmediate));

    let mut insn = decode_at(0x1400_0001, 0x1000);
    // A64 branch targets are 4-byte aligned.
    assert!(insn.set_near_branch_target(0x1002));
    assert_eq!(insn.encode(), Err(EncodeError::InvalidImmediate));
}

#[test]
fn set_near_branch_target_only_applies_to_direct_branches() {
    // adr carries a resolved label but is not a branch.
    let mut insn = decode_at(0x1000_0000, 0x1000);
    assert!(!insn.set_near_branch_target(0x2000));
    // `set_label` is the general form and does apply.
    assert!(insn.set_label(0x1010));
    assert_encodes_and_round_trips(&insn, 0x1000_0080);

    // An indirect branch has no label at all.
    let mut br = decode(0xD61F_0020);
    assert!(!br.set_near_branch_target(0x2000));
    assert!(!br.set_label(0x2000));
}

#[test]
fn set_ip_keeps_the_absolute_target() {
    // b 0x1004 @ 0x1000. Moving the instruction to 0x1004 keeps it pointing at
    // 0x1004, which is now a zero displacement.
    let mut insn = decode_at(0x1400_0001, 0x1000);
    insn.set_ip(0x1004);
    assert_eq!(insn.ip(), 0x1004);
    assert_eq!(insn.near_branch_target(), 0x1004);
    assert_encodes_and_round_trips(&insn, 0x1400_0000);
}

#[test]
fn relocate_keeps_the_relative_displacement() {
    // b 0x1004 @ 0x1000 relocated to 0x2000 still jumps +4, so the word is
    // unchanged and the target follows.
    let mut insn = decode_at(0x1400_0001, 0x1000);
    insn.relocate(0x2000);
    assert_eq!(insn.ip(), 0x2000);
    assert_eq!(insn.near_branch_target(), 0x2004);
    assert_encodes_and_round_trips(&insn, 0x1400_0001);
}

#[test]
fn relocate_leaves_a_label_less_instruction_alone_but_moves_it() {
    let mut insn = decode_at(0x8B02_0020, 0x1000);
    insn.relocate(0x2000);
    assert_eq!(insn.ip(), 0x2000);
    assert_encodes_and_round_trips(&insn, 0x8B02_0020);
}

#[test]
fn adrp_rejects_a_target_it_cannot_name() {
    // ADRP addresses a 4 KiB page, so a page-aligned target encodes exactly...
    let mut insn = decode_at(0x9000_0000, 0x1000);
    assert_eq!(insn.op(1), Operand::Label(0x1000));
    assert!(insn.set_label(0x2000));
    assert_encodes_and_round_trips(&insn, 0xB000_0000);

    // ...and one carrying sub-page bits has no encoding at all. It must not
    // silently round down to its containing page, which would re-encode to a
    // different address than the operand says.
    let mut sub = decode_at(0x9000_0000, 0x1000);
    assert!(sub.set_label(0x2004));
    assert_eq!(sub.encode(), Err(EncodeError::InvalidImmediate));
}

#[test]
fn relocating_an_adrp_by_a_sub_page_delta_is_an_encode_error() {
    // A page-aligned move keeps ADRP encodable...
    let mut ok = decode_at(0x9000_0000, 0x1000);
    ok.relocate(0x9000);
    assert_eq!(ok.near_branch_target(), 0);
    assert_eq!(ok.op(1), Operand::Label(0x9000));
    assert_eq!(ok.encode(), Ok(0x9000_0000));

    // ...a finer one leaves a label ADRP cannot name.
    let mut bad = decode_at(0x9000_0000, 0x1000);
    bad.relocate(0x1004);
    assert_eq!(bad.op(1), Operand::Label(0x1004));
    assert_eq!(bad.encode(), Err(EncodeError::InvalidImmediate));
}

#[test]
#[cfg(feature = "sve")]
fn a_predicate_as_counter_slot_only_accepts_p8_to_p15() {
    // whilegt pn14.b, x5, x0, vlx2 — the PNg field is 3 bits wide, so p0..p7
    // are the right class and width but still have no encoding here.
    let mut insn = decode(0x2520_40BE);
    assert_eq!(insn.op_kind(0), OpKind::PredCounter);

    let before = insn;
    assert!(!insn.set_op_register(0, Register::P3));
    assert!(!insn.set_op_register(0, Register::P7));
    assert_eq!(insn, before);
    assert!(!insn.is_modified());

    // p8..p15 are accepted and encode.
    assert!(insn.set_op_register(0, Register::P9));
    assert_encodes_and_round_trips(&insn, 0x2520_40B9);
}

#[test]
fn changing_the_address_only_marks_pc_relative_instructions_modified() {
    // `add x0, x1, x2` encodes identically wherever it sits, so moving it
    // leaves `word()` valid and `is_modified()` false.
    let mut alu = decode_at(0x8B02_0020, 0x1000);
    alu.set_ip(0x2000);
    assert_eq!(alu.ip(), 0x2000);
    assert!(!alu.is_modified());
    assert_eq!(alu.word(), 0x8B02_0020);
    assert_eq!(alu.encode(), Ok(0x8B02_0020));

    alu.relocate(0x3000);
    assert_eq!(alu.ip(), 0x3000);
    assert!(!alu.is_modified());

    // A branch's encoding does depend on `ip`, so moving it does mark it.
    let mut br = decode_at(0x1400_0001, 0x1000);
    br.set_ip(0x2000);
    assert!(br.is_modified());

    let mut br2 = decode_at(0x1400_0001, 0x1000);
    br2.relocate(0x2000);
    assert!(br2.is_modified());
}

#[test]
fn set_condition_refuses_instructions_without_a_condition_operand() {
    let mut insn = decode(0x8B02_0020);
    assert!(!insn.set_condition(Condition::Eq));
    assert!(!insn.is_modified());
}

#[test]
fn replace_the_condition_of_a_conditional_select() {
    // csel x0, x1, x2, ne  ->  csel x0, x1, x2, eq
    let mut insn = decode(0x9A82_1020);
    assert!(insn.set_condition(Condition::Eq));
    assert_encodes_and_round_trips(&insn, 0x9A82_0020);
}

// ---------------------------------------------------------------------------
// Code / mnemonic identity.
// ---------------------------------------------------------------------------

#[test]
fn set_code_moves_to_a_sibling_encoding() {
    // add x0, x1, x2  ->  sub x0, x1, x2
    let mut insn = decode(0x8B02_0020);
    insn.set_code(Code::SubShifted64);
    assert_eq!(insn.mnemonic(), Mnemonic::Sub);
    assert_encodes_and_round_trips(&insn, 0xCB02_0020);
}

#[test]
fn set_code_to_an_incompatible_encoding_fails_at_encode_time() {
    // The operand list of `add x0, x1, x2` does not fit a load/store encoding.
    let mut insn = decode(0x8B02_0020);
    insn.set_code(Code::LdrImmUnsigned64);
    assert!(insn.encode().is_err());
}

// ---------------------------------------------------------------------------
// Operand-list shape.
// ---------------------------------------------------------------------------

#[test]
fn set_op_replaces_and_extends_the_operand_list() {
    let mut insn = decode(0x8B02_0020); // add x0, x1, x2
    assert_eq!(insn.op_count(), 3);

    // Replace an operand wholesale with a different kind.
    assert!(insn.set_op(2, Operand::ImmUnsigned(4)));
    assert_eq!(insn.op_kind(2), OpKind::ImmUnsigned);

    // Writing past the end extends the list.
    assert!(insn.set_op(4, Operand::ShiftAmount(1)));
    assert_eq!(insn.op_count(), 5);
    assert_eq!(insn.op_kind(3), OpKind::None);

    // Past MAX_OPERANDS is refused.
    assert!(!insn.set_op(fARM64::MAX_OPERANDS, Operand::ImmUnsigned(0)));
}

#[test]
fn push_and_truncate_the_operand_list() {
    let mut insn = decode(0x8B02_0020);
    assert!(insn.push_op(Operand::ShiftAmount(2)));
    assert_eq!(insn.op_count(), 4);

    assert!(insn.set_op_count(2));
    assert_eq!(insn.op_count(), 2);
    // Truncated slots are cleared, not merely hidden.
    assert!(insn.set_op_count(4));
    assert_eq!(insn.op(2), Operand::None);
    assert_eq!(insn.op(3), Operand::None);

    assert!(!insn.set_op_count(fARM64::MAX_OPERANDS + 1));

    // A full operand list cannot be pushed to.
    assert!(insn.set_op_count(fARM64::MAX_OPERANDS));
    assert!(!insn.push_op(Operand::ShiftAmount(1)));
}

// ---------------------------------------------------------------------------
// The modified flag and `re_encode`.
// ---------------------------------------------------------------------------

#[test]
fn re_encode_commits_the_edit_into_the_raw_word() {
    let mut insn = decode(0xF940_0420); // ldr x0, [x1, #8]
    assert_eq!(insn.word(), 0xF940_0420);
    assert!(!insn.is_modified());

    assert!(insn.set_memory_base(Register::X3));
    assert!(insn.is_modified());
    // `word()` still reports what was decoded.
    assert_eq!(insn.word(), 0xF940_0420);

    assert_eq!(insn.re_encode(), Ok(0xF940_0420 | (3 << 5)));
    assert_eq!(insn.word(), 0xF940_0460);
    assert!(!insn.is_modified());
}

#[test]
fn a_failed_re_encode_changes_nothing() {
    let mut insn = decode(0x9240_1C20); // and x0, x1, #0xff
    assert!(insn.set_op_immediate(2, 0x5)); // not a valid bitmask immediate
    assert_eq!(insn.re_encode(), Err(EncodeError::InvalidImmediate));
    // The original word and the modified flag both survive the failure.
    assert_eq!(insn.word(), 0x9240_1C20);
    assert!(insn.is_modified());
}

#[test]
fn encode_bytes_is_the_little_endian_form() {
    let insn = decode(0x8B02_0020);
    assert_eq!(insn.encode_bytes(), Ok(0x8B02_0020u32.to_le_bytes()));
}

// ---------------------------------------------------------------------------
// A broader sweep.
// ---------------------------------------------------------------------------

#[test]
fn register_renaming_round_trips_across_the_dataproc_space() {
    // For every instruction in a dense slice of the data-processing space that
    // the encoder already round-trips, rename each plain 64-bit GP operand to a
    // different register and require the edited word to decode back to exactly
    // the edited operands. This exercises the edit path over far more encodings
    // than the hand-written cases above.
    let mut checked = 0usize;
    for w in (0u32..=0x00FF_FFFF).step_by(1039) {
        for base in [0x8B00_0000u32, 0xCB00_0000, 0x9A00_0000, 0xD100_0000] {
            let word = base | w;
            let insn = decode(word);
            if insn.is_invalid() || insn.encode() != Ok(word) {
                continue;
            }
            for slot in 0..insn.op_count() {
                let old = insn.op_register(slot);
                // Only plain, non-special 64-bit GP registers: SP/ZR carry
                // encoding-defined meaning and the alias the decoder picked can
                // depend on them.
                if old.class() != fARM64::RegClass::Gp
                    || old.width_bits() != 64
                    || old == Register::Sp
                    || old == Register::Xzr
                {
                    continue;
                }
                let new = if old == Register::X7 {
                    Register::X9
                } else {
                    Register::X7
                };
                let mut edited = insn;
                if !edited.set_op_register(slot, new) {
                    continue;
                }
                let encoded = match edited.encode() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                // The rename must actually change the encoding, and the new
                // word must be self-consistent.
                assert_ne!(
                    encoded, word,
                    "renaming slot {slot} of {word:#010x} left the word unchanged"
                );
                let back = decode(encoded);
                assert_eq!(
                    back.code(),
                    edited.code(),
                    "renaming slot {slot} of {word:#010x} changed the encoding identity"
                );
                assert_eq!(
                    back.encode(),
                    Ok(encoded),
                    "the word produced by renaming slot {slot} of {word:#010x} does not round-trip"
                );
                // A rename can change which preferred-disassembly alias the
                // decoder picks, which re-shapes the operand list; compare the
                // slot directly when the shape is unchanged, and otherwise just
                // require the new register to be referenced.
                if back.op_count() == edited.op_count()
                    && back.op_kind(slot) == edited.op_kind(slot)
                {
                    assert_eq!(
                        back.op(slot),
                        edited.op(slot),
                        "renaming slot {slot} of {word:#010x} did not land in the word"
                    );
                } else {
                    assert!(
                        (0..back.op_count()).any(|i| back.op_register(i) == new),
                        "renaming slot {slot} of {word:#010x} lost {new:?}: {back:?}"
                    );
                }
                checked += 1;
            }
        }
    }
    assert!(
        checked > 500,
        "sweep covered only {checked} renames; the filter is too tight to be meaningful"
    );
}
