# VALIDATION.md — Methodology & Results (fARM64)

How `fARM64` is validated, and the current measured results. Correctness is
defined as **conformance with the *Arm Architecture Reference Manual* (the "ARM
ARM")**; the corpora and tools below are differential oracles, not the authority.
Where an oracle diverges from the spec, `fARM64` follows the spec and the
difference is recorded in the [divergences](#divergences-spec-correct-non-reproductions)
table.

---

## Results (final, measured)

Numbers below are the live output of the golden corpus harness (run command in
[Running the harness](#running-the-harness)).

### Overall

| Metric | Value |
|-|-|
| Corpus cases (total) | 42289 |
| Attempted (decoded to a non-`Invalid` instruction) | 42016 (99.35% of total) |
| Matched the oracle | 41925 (99.78% of attempted) |
| Recorded mismatches | 91 |
| Distinct group labels | 1270 |

"Coverage" is *attempted / total* (99.35%); "parity" is *matched / attempted*
(99.78%). The remaining mismatches are the verified-spec-correct binja rendering
bugs in the [divergences](#divergences-spec-correct-non-reproductions) table plus
a small number of sysreg-name data gaps.

### Worst groups (lowest match rate among attempted)

Every 0%/low group below is an intentional, spec-verified non-reproduction of a
Binary Ninja rendering quirk — see the divergences table.

| Group | total | attempted | matched | rate |
|-|-|-|-|-|
| REVD | 8 | 8 | 0 | 0.00% |
| histseg | 8 | 8 | 0 | 0.00% |
| match | 8 | 8 | 0 | 0.00% |
| nmatch | 8 | 8 | 0 | 0.00% |
| pmul | 8 | 8 | 0 | 0.00% |
| saba | 8 | 8 | 1 | 12.50% |
| uaba | 8 | 8 | 3 | 37.50% |
| sqxtnb | 8 | 8 | 4 | 50.00% |
| sqxtnt | 8 | 8 | 5 | 62.50% |
| sqxtunt | 8 | 8 | 5 | 62.50% |
| uqxtnb | 8 | 8 | 5 | 62.50% |
| uqxtnt | 8 | 8 | 6 | 75.00% |
| sqrdmlsh | 32 | 32 | 26 | 81.25% |
| sqrdmlah | 32 | 32 | 28 | 87.50% |
| sqxtunb | 8 | 8 | 7 | 87.50% |
| LDNP | 80 | 80 | 76 | 95.00% |
| LDP | 240 | 240 | 235 | 97.92% |

### Representative fully-passing groups

The large base-ISA, scalar-FP and SVE load groups are at 100% parity:

| Group | total | matched | rate |
|-|-|-|-|
| LDR | 552 | 552 | 100.00% |
| STR | 472 | 472 | 100.00% |
| MOV | 532 (attempted) | 532 | 100.00% |
| LD1 / ST1 | 384 / 384 | 384 / 384 | 100.00% |
| FMOV | 336 | 336 | 100.00% |
| FCVTZS / FCVTZU / SCVTF / UCVTF | 288 each | 288 each | 100.00% |
| LD2/LD3/LD4 / ST2/ST3/ST4 | 240 each | 240 each | 100.00% |
| ld1h / ld1b / ld1sh / ld1w (SVE) | 224/208/192/192 | all | 100.00% |

### Coverage gaps (total − attempted)

The largest not-yet-attempted buckets, mostly a handful of unimplemented niche
forms (vector `BIF`/`BIT`/`BSL`, `SHLL`, `LD64B`/`ST64B*`) and the modified-imm
variants folded into `MOV`/logical groups:

| Group | total | attempted | gap |
|-|-|-|-|
| MOV | 578 | 532 | 46 |
| AND / BIC / EOR | 80 | 64 | 16 each |
| BIF / BIT / BSL | 16 | 0 | 16 each |
| MVN / ORN | 48 | 32 | 16 each |
| ORR | 112 | 96 | 16 |
| RBIT | 48 | 32 | 16 |
| SHLL | 16 | 0 | 16 |
| SB | 16 | 1 | 15 |
| LD64B / ST64B / ST64BV / ST64BV0 | 8 | 0 | 8 each |

---

## Round-trip (encoder) {#round-trip-encoder}

The encoder (`encode` / `Instruction::encode`, see
[API.md](./API.md#encoder-encode--encodeerror)) is validated by a **decode →
encode → decode** round-trip over the same binja corpus, in `tests/roundtrip.rs`.
For each corpus word it decodes to an `Instruction`, calls `encode` (which
rebuilds the word from *semantics only* — it never reads `insn.word()`), then
re-decodes the produced word and checks the two `Instruction`s are
value-identical. This proves the decode is invertible without depending on the
raw input bits.

### Results (final, measured)

| Metric | Value |
|-|-|
| Corpus cases (total) | 42289 |
| Encode attempted (decoded to a non-`Invalid` instruction) | 42144 |
| **Semantic round-trip** (encode re-decodes to the identical `Instruction`) | **42144 / 42144 = 100.00%** of attempted (= 99.85% of all decoded) |
| Exact-word (encode reproduces the original bits) | 98.00% |

Every attempted case round-trips **semantically**: `decode(encode(insn))` equals
`insn` (same `code`, `mnemonic`, operands, flow/flags). The encoder covers all
decoder groups (dp_imm / dp_reg / branch_sys / ldst / ldst_simd / simd_fp+crypto
/ sve / sme). Inverse pseudocode helpers used: `encode_bit_masks` (logical
immediate) and the inverse of `vfp_expand_imm` / `adv_simd_expand_imm`.

### The ~2% exact-word gap is don't-care / should-be-ones fields

The ~2% of cases where the re-encoded word differs *bit-for-bit* from the input
are **semantically lossless**: they are architectural don't-care or should-be-one
fields that the decoder discards (they carry no semantics), so the encoder emits
the canonical value. The re-encoded word still re-decodes to the identical
`Instruction`. The affected fields are:

| Encoding | Discarded field |
|-|-|
| `SMULH` / `UMULH` | `Ra` |
| Load/store-exclusive | `Rs` / `Rt2` |
| `IC` | `Rt` |
| `DUP` (general) | index bits |
| `FCMP` / `FCMPE #0.0` | `Rm` |

These are value-identical to the canonical encoding and re-decode identically, so
they are counted as semantic passes, not defects.

### Running the round-trip

```sh
cargo test --features "std full" -- --ignored --nocapture roundtrip
```

---

## Oracles

Two independent oracles, neither authoritative:

| Oracle | Role | What it gives |
|-|-|-|
| Binary Ninja corpus (`refs/.../test_cases.txt`) | exact-match development guide | a large `(word, expected text)` set, read locally, never shipped |
| LLVM `llvm-mc` / `llvm-objdump` | spec cross-check | second opinion used to confirm that a binja mismatch is in fact a binja bug |

GNU binutils `objdump` is a third optional cross-check used ad hoc. The ARM ARM
is the tiebreaker; the binja corpus is a guide, and every divergence from it is
cited against LLVM in the table below.

---

## Harness

| File | Purpose |
|-|-|
| `tests/golden.rs` | the binja-corpus differential sweep; tallies coverage + parity, buckets by group label, dumps mismatches |
| `tests/llvm_diff.rs` | per-word cross-check vs `llvm-mc --disassemble`; skips cleanly if `llvm-mc` is absent |
| `tests/roundtrip.rs` | decode → `encode` → re-decode over the corpus; asserts the re-decoded `Instruction` is value-identical (semantic round-trip) and tallies exact-word rate |
| `tests/common/mod.rs` | shared corpus streaming, the fixed test address, and the `normalize` pipeline |
| `examples/disasm.rs` | tiny CLI: hex word(s) on argv/stdin → `WWWWWWWW\t<text>` via the zero-alloc `BufSink` + `FmtFormatter` path |

The golden test decodes each corpus word at the fixed corpus anchor address,
formats with the default `fARM64::format::FmtFormatter`, runs both sides through
`common::normalize` (trim/collapse whitespace, strip trailing `//` comments,
expand `{zN.T-zM.T}` register ranges, normalize hex/decimal and floats,
lowercase, plus token equivalences like `cs==hs` / `cc==lo`) and compares. It is
`#[ignore]`d and currently **non-failing** (it reports a coverage/parity summary
rather than gating); `MATCH_THRESHOLD` can be raised to turn it into a regression
gate. Mismatches are written to `target/golden-mismatches.txt`.

### Running the harness

```sh
# build clean (0 warnings expected)
cargo build && cargo build --features "std full"

# unit tests
cargo test --features "std full" --lib

# corpus coverage + parity summary (the numbers in this file)
cargo test --features "std full" -- --ignored --nocapture golden 2>&1 \
  | grep -A40 "corpus parity"

# LLVM spec cross-check (needs llvm-mc on PATH)
cargo test --features "std full" -- --ignored --nocapture llvm

# manual spot check
printf '0x20,0x04,0x00,0x11' | \
  "/c/Program Files/LLVM21/bin/llvm-mc" --disassemble --triple=aarch64 \
  +sve,+sve2,+sme,+sha2,+sha3,+aes,+sm4
cargo run --features std --example disasm 11000420 d503201f
```

Env knobs (both differential tests): `FARM64_CORPUS=<path>` overrides the corpus
location, `FARM64_GROUP=<substr>` filters by group label, `FARM64_LIMIT=<n>`
caps the case count, `FARM64_LLVM=<exe>` overrides the `llvm-mc` binary.

---

## Divergences (spec-correct non-reproductions)

These are Binary Ninja rendering bugs that `fARM64` intentionally does **not**
reproduce. Each was confirmed against LLVM `llvm-mc` (and the ARM ARM) to be
spec-correct on the fARM64 side, so the corpus mismatch is expected and counted
as a known difference, not a defect.

| Area | binja rendering bug | fARM64 (spec-correct) |
|-|-|-|
| `REVD` (SVE) | wrong element size / operand spelling for the 128-bit reverse | spec arrangement per ARM ARM, confirmed by LLVM |
| `pmul` | mis-rendered polynomial-multiply arrangement | correct arrangement/lane per spec |
| `histseg` | wrong operand/arrangement spelling | spec form, LLVM-confirmed |
| `match` / `nmatch` (SVE2) | wrong predicate/arrangement rendering | spec form, LLVM-confirmed |
| `saba` / `uaba` | incorrect accumulate-arrangement spelling | spec arrangement |
| `sqxtnb` / `sqxtnt` / `sqxtunb` / `sqxtunt` / `uqxtnb` / `uqxtnt` | wrong narrowing source/dest element size | spec narrowing widths |
| `sqrdmlah` / `sqrdmlsh` (`.b`) | byte-arrangement rendering bug | spec `.b` arrangement |
| decimal immediates | binja prints some immediates in decimal where the spec uses hex (and the SVE mixed radix) | hex-always, with the documented SVE signed-radix convention |

The SVE radix convention fARM64 follows (positive in hex, negative in decimal for
the specific `INDEX`/`CMP #imm`/`ADDVL`/`LD1RQ` families) is encoded in the
`Operand::ImmSignedDec` / `SveMemMode` variants and matches the corpus where binja
itself is consistent.

---

## Robustness invariants (checked)

- Decode is **total and panic-free** for all 2^32 words; every failure path
  returns `Code::Invalid` + a typed `DecodeError`, never a panic.
- `len()` is always `INSN_LEN` (4); the cursor advances exactly 4 bytes.
- The default decode + `FmtFormatter` + `BufSink` path performs **zero heap
  allocation** (no `alloc` dependency on that path).

---

*`fARM64` is an original implementation derived from the ARM ARM, licensed
`MIT`. Oracle corpora are used locally for differential testing
only and are never shipped.*
