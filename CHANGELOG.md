# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
