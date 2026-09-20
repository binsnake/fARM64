# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `implicit` module: implicit register reads/writes — the architectural state an
  instruction touches without naming it in an operand. `implicit_registers()`
  (and `Instruction::implicit_registers()`) return a fixed-capacity,
  allocation-free `ImplicitRegisters` list covering the link register of
  calls/returns, the `X30`/`SP`/`X16`/`X17` of the pointer-authentication hint
  and FEAT_PAuth_LR forms, the eight-register FEAT_LS64 `LD64B`/`ST64B` group,
  the SVE `FFR`, `PC` for PC-relative address generation, and `NZCV`. The
  streaming-mode transition of `SMSTART`/`SMSTOP` zeroes every `Z`/`P`
  register; that is documented as out of scope rather than reported.
- Four implicit-state pseudo-registers on `Register` — `Nzcv`, `Ffr`, `Za`,
  `Pc` — in a new `RegClass::Special`, plus `Register::is_pseudo()`. They are
  never produced by the decoder as operands and never rendered by a formatter.
- In-place instruction editing, so a decoded `Instruction` can be re-encoded
  with different operands: `set_op`, `set_op_count`, `push_op`,
  `set_op_register` (class- and width-checked), `set_op_register_unchecked`,
  `set_op_immediate`, `set_condition`, `set_near_branch_target`, `set_label`,
  `set_memory_base`, `set_memory_index`, `set_memory_displacement64`,
  `set_code`, `set_mnemonic`, `set_ip`, and `relocate`. Every setter is total.
  `set_op_register` also enforces the operand shape's own range, so the 3-bit
  predicate-as-counter `PNg` field rejects `P0`..`P7`.
- `Instruction::is_modified()`, `Instruction::encode_bytes()`, and
  `Instruction::re_encode()` (encode and commit into `word()`; unchanged on
  failure). `set_ip`/`relocate` mark an instruction modified only when its
  encoding actually depends on `ip`, i.e. when it carries a label.

### Changed

- `InstructionInfo::used_registers()` now merges the implicit registers in, so
  `NZCV` appears as `Register::Nzcv` and a register that is both named and
  implicitly touched reports one combined access. `flags_read()` /
  `flags_written()` are unchanged and remain the scalar view of the same fact.
- More accurate access classification: SVE/SME governing predicates are read
  rather than written on loads; the MOPS `CPY*`/`SET*` writeback operands are
  read-modified (and `SET*`'s fill value is read); `ST64BV`/`ST64BV0` write only
  their status register; the `PAC*`/`AUT*`/`XPAC*` data-processing forms and
  `CHKFEAT` read-modify their destination; `RMIF`/`SETF8`/`SETF16`/`WRFFR` and
  the FEAT_PAuth_LR `AUT*SPPCR` forms have no destination register; and an SME
  ZA tile / tile-slice / ZA-array operand reports the `ZA` pseudo-register
  instead of a same-numbered SVE `Z` register.
- `Instruction::set_flags()` reports `FlagEffect::Sets` for the flag-manipulation
  forms `RMIF`, `SETF8`/`SETF16`, `CFINV`, `AXFLAG`/`XAFLAG`, and for the SVE
  `RDFFRS`, which previously reported `FlagEffect::None`.
- `Instruction::flow_control()` classifies the FEAT_PAuth_LR
  `RETAASPPC`/`RETABSPPC`/`RETAASPPCR`/`RETABSPPCR` as `FlowControl::Return`.
- `RegClass` is now `#[non_exhaustive]`.
- `MAX_USED_MEM` is now documented as exact at two (the MOPS copy family carries
  both a destination and a source memory operand), and a sweep of the encoding
  space guards both inline access-list capacities against silent truncation.

### Fixed

- `Instruction::flow_control()` returned `FlowControl::Next` for the FEAT_HBC
  hinted conditional branch `BC.<cond>`, which also made
  `Instruction::near_branch_target()` return `0` for it despite documenting the
  opposite. It is now a `FlowControl::ConditionalBranch` with a resolved target.
- The `ADRP` encoder masked its target down to the containing 4 KiB page. A
  page-aligned target is unaffected (the decoder only ever produces one), but a
  target carrying sub-page bits now reports `EncodeError::InvalidImmediate`
  instead of silently encoding a different address than the operand names.
- Merging predication (`Pg/M`) leaves the destination's unselected elements at
  their previous value, so the destination is now `OpAccess::ReadWrite` rather
  than `OpAccess::Write` — `ABS Zd.B, Pg/M, Zn.B` and the SME `MOVA` tile-slice
  forms preserve what the predicate does not select. Zeroing predication
  (`Pg/Z`) is unchanged.

## [0.0.2] - 2026-07-14

### Added

- Complete, allocation-free `Code::values()` and `Mnemonic::values()` iterators
  with the same exact-size, double-ended, fused contract as iced enums.
- Safe, constant-time `from_u16()` conversion and iced-compatible
  `TryFrom<usize>` implementations for both public instruction enums.
- A zero-sized, `no_std` `EnumValueError` for rejected integer conversions.

### Changed

- Enum metadata tests now consume the generated complete catalogs instead of a
  manually maintained partial `Code` list.

## [0.0.1] - 2026-07-13

Initial crates.io release.

### Added

- A pure-Rust AArch64 decoder with a `no_std`, allocation-free default build.
- Fixed-buffer Arm UAL formatting and an opt-in, UAL-equivalent GNU adapter.
- Runtime architecture-feature selection and Cargo compile-out features.
- Instruction metadata and encoding support.
- Optional `alloc` and `std` convenience APIs.
- Differential, round-trip, portability, MSRV, and allocation-audit tests.

[Unreleased]: https://github.com/binsnake/fARM64/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/binsnake/fARM64/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/binsnake/fARM64/releases/tag/v0.0.1
