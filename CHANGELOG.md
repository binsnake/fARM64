# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2026-07-13

Initial crates.io release.

### Added

- A pure-Rust AArch64 decoder with a `no_std`, allocation-free default build.
- Fixed-buffer Arm UAL formatting and an opt-in, UAL-equivalent GNU adapter.
- Runtime architecture-feature selection and Cargo compile-out features.
- Instruction metadata and encoding support.
- Optional `alloc` and `std` convenience APIs.
- Differential, round-trip, portability, MSRV, and allocation-audit tests.

[0.0.1]: https://github.com/binsnake/fARM64/releases/tag/v0.0.1
