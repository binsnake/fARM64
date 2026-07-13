# Extension status and known gaps

This file is a public status summary, not an exhaustive claim that every A64
encoding in every Arm revision is implemented. The checked-in tests and optional
differential sweeps are the source of truth for the current revision.

## Implemented surface

The decoder and semantic encoder cover the base A64 groups, scalar floating
point and Advanced SIMD, cryptographic instructions, SVE/SVE2, SME/SME2, and a
broad set of later architectural extensions. These include LSE/LSE128, PAuth and
PAuth_LR, MTE, MOPS, CSSC, RCPC3, D128, THE, LSUI, SVE2.1–2.3 additions, FP8,
FPRCVT, LSFE, CPA, CMPBR, GCS, TME, and newer system instructions represented by
the public `Feature` enum.

The repository also supports two Apple implementation-defined families:

- Apple AMX, using operation names and encodings documented by the public
  [`corsix/amx`](https://github.com/corsix/amx) reverse-engineering project.
- Apple GXF (`GENTER` and `GEXIT`), referencing Asahi Linux's
  [Apple Proprietary Instructions](https://asahilinux.org/docs/hw/cpu/apple-instructions/)
  encoding documentation and Sven Peter's [GXF background](https://blog.svenpeter.dev/posts/m1_sprr_gxf/).

These are not Arm architectural extensions. They are rejected by
`FeatureSet::BASE` and require `Feature::AppleAmx` or `Feature::Gxf` (or the
permissive `FeatureSet::ALL`). Focused tests pin their accepted and reserved
forms and semantic encode/decode round trips.

## Compile-time and runtime gates

Cargo features and runtime features serve different purposes:

- `sve` and `sme` compile the corresponding large decoder and encoder modules.
- `crypto` compiles the Advanced SIMD crypto decoder. Crypto `Code`/`Mnemonic`
  variants and encoder support remain part of the public build without it.
- `full` enables those three Cargo features.
- `FeatureSet` is the fine-grained runtime admission layer. FP16, BF16, LSE,
  PAuth, MTE, and most other entries in `Feature` are always compiled and have
  no same-named Cargo feature.

Enabling a runtime feature cannot restore an SVE, SME, or crypto decoder module
that was omitted at compile time.

## Known limitations

- `GnuFormatter` currently delegates to the UAL formatter. It is an API
  compatibility adapter, not yet a distinct GNU/objdump rendering policy.
- The encoder operates on one `Instruction`; block relocation and a
  `BlockEncoder` are outside the current API.
- FEAT_LS64WB writeback forms are not currently claimed as supported.
- Very new Arm architecture revisions can add encodings faster than this list
  is updated. Absence from this summary is not a compatibility guarantee.

## Validation

Normal CI exercises formatting, linting, documentation, supported feature
combinations, tests, portability builds, and package construction. Optional
ignored tests can compare against a locally supplied Vector 35 corpus or an
installed LLVM toolchain. Those external inputs are not included in the crate,
so the public documentation intentionally does not publish a fixed corpus
coverage percentage or test count.

When adding an extension, include focused valid, invalid/reserved, runtime-gate,
formatter, and semantic round-trip cases. If an external oracle is available,
use it as an additional cross-check rather than as the architectural authority.
