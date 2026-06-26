# fARM64 — Remaining extensions / gaps (next-session handoff)

Snapshot after Batch F (commit `33be3ca`, 2026-06-26). This is the prioritized worklist for closing the
rest of the AArch64 surface. It is generated from the **LLVM 21 differential** (`tests/llvm_diff.rs`), which
is the oracle now that the Binary Ninja corpus is 100% decoded.

## Current state

Metric | Value
|-|-|
Binja corpus decode | **100.00% coverage**, 99.75% text parity (residual = binja rendering bugs; fARM64 is spec-correct vs LLVM)
Encoder | **100.00% semantic round-trip** over the corpus
LLVM **GAPS** (LLVM decodes, fARM64 returns `Invalid`) | **6,282** (was 46,811 at the start of this session)
LLVM **DISAGREEMENTS** (both decode, text differs) | 14,863 (mostly alias/radix style noise; a few real — see §3)
LLVM **REVERSE** (fARM64 decodes, LLVM rejects = over-decode) | 168,480 (correctness debt — see §2)
Tests / clippy | 345 lib tests, 0 clippy, all configs 0-warning
Sample size | ~5.04M words (random + structured), `--mattr=+all`

Regenerate this data any time:
```
cargo test --features "std full" -- --ignored --nocapture llvm_diff   # writes target/llvm-diff.txt
# section (a)=GAPS  (b)=DISAGREEMENTS  (c)=REVERSE, each "count<TAB>mnemonic<TAB>word<TAB>llvm-text"
```

## 1. Missing instructions — the 6,282 GAPs (priority worklist)

Grouped by ARM feature + the fARM64 module area they belong in (for the parallel-worktree model in §4).
Counts are differential-sample words (relative magnitude, not exact instruction counts). Each row has an
example word you can feed to `llvm-mc --disassemble --triple=aarch64 --mattr=+all`.

### Group G1 — `simd_fp` area (NEON + SVE matrix/dot/mlal)
Family | Mnemonics | ~count | FEAT | example word | notes
|-|-|-|-|-|-|
Integer/FP matrix-mul | `smmla` `ummla` `usmmla` (NEON), `fmmla` (SVE), `smmla/ummla/usmmla` (SVE) | ~560 | I8MM / F32MM / F64MM | `4E80A45F` smmla v31.4s,v2.16b,v0.16b | NEON `.4s,.16b,.16b`; SVE forms too
SVE FP8 widening mlal (z-form) | `fmlalb` `fmlalt` `fmlallbb` `fmlallbt` `fmlalltb` `fmlalltt` | ~400 | FP8 (SVE) | `6420518E` fmlalb z14.h,z12.b,z0.b[0] | z-reg analogue of the NEON forms done in Batch D
SVE BF16 mul-add (indexed) | `bfmla` `bfmls` `bfmul` | ~360 | SVE_BF16 | `64220960` bfmla z0.h,z11.h,z2.h[0] | indexed `z.h[i]`
SVE multi-vector FMUL | `fmul` (`{z..},{z..},{z..}`) | ~140 | SME2/SVE2 | `C160E798` fmul {z24.h,z25.h},{z28.h,z29.h},{z0.h,z1.h} | multi-vector, reuses SveVecGroup

### Group G2 — `sve` area (SVE2.1 quadword + misc)
Family | Mnemonics | ~count | FEAT | example word | notes
|-|-|-|-|-|-|
SVE2.1 quadword (.q) load/store | `ld1q` `st1q` `ld2q`–`ld4q` `st2q`–`st4q`, `ld1w/d` `.q` | ~1,340 | SVE2p1 | `A5208101` ld3q {z1.q-z3.q},p0/z,[x8,x0,lsl #4] | `.q` element + gather `ld1q [z.d, x]`
SVE `REVD` zeroing | `revd` (`z.q, p/z, z.q`) | ~150 | SVE2p1 | `052EA020` revd z0.q,p0/z,z1.q | merging form exists; `/z` deferred. Also fix the merging size!=00 over-decode (REVERSE)
SVE2.1 predicate-pair / counter WHILE | `whilehi/lo/...` predicate-pair `{p0,p1}` and `pn` `vlx2` forms | ~150 | SVE2p1 | `25225911` whilehi {p0.b,p1.b},x8,x2 | new predicate-pair result + `vlx2/vlx4`
SVE FP unary-pred convert (`/z`) | frintn/p/m/z/a/x/i, frecpx, fsqrt, fcvt, fcvtx, scvtf/ucvtf, fcvtzs/zu, bfcvt, flogb | (in gap) | SVE2p1 | top-byte 0x64 sel=101 | the zeroing convert/round family deferred in F (different opcode layout than the merging path)

### Group G3 — `sme` area (SME2 LUT + ZA move)
Family | Mnemonics | ~count | FEAT | example word | notes
|-|-|-|-|-|-|
SME2 LUT (ZT0 table) | `luti2` `luti4` (`{z..}, zt0, z[i]`) | ~865 | LUT (SME2) | `C08C407A` luti2 {z26.b,z27.b},zt0,z3[0] | needs the ZT0 table register operand
SME2 ZA tile move | `mov` / `movaz` (`za0h.b[w12,6:7], {z..}` and reverse) | ~140 | SME2 | `C0040183` mov za0h.b[w12,6:7],{z12.b,z13.b} | ZA horizontal/vertical slice + group

### Group G4 — `ldst` area (FP atomics)
Family | Mnemonics | ~count | FEAT | example word | notes
|-|-|-|-|-|-|
FP atomic memory ops | `ldfadd` `ldfmax` `ldfmin` `ldfmaxnm` `ldfminnm` (+ `a`/`l`/`al` ordering, h/s/d) | ~600 | LSFE | `7C20039A` ldfadd h0,h26,[x28] | full LSE-style family but FP; mirror the integer LSE atomic decoder

## 2. Over-decoding (REVERSE = 168,480) — correctness debt, NOT missing instructions

fARM64 currently decodes many encodings that LLVM (`--mattr=+all`) rejects as UNDEFINED — i.e. it is too
permissive (doesn't enforce some reserved-bit / size / register constraints). Biggest offenders:

count | mnemonic | example | likely cause
|-|-|-|-|
20,030 | `fmops` | `808000F7` | SME outer-product not validating reserved/size bits (accepts undefined)
17,568 | `fmopa` | `8080032F` | same family — together ~37k
11,202 | `mov` | `05082005` | SVE mov/dup form accepting undefined sub-encodings
11,114 | `ext` | `050107F6` | SVE EXT over-broad
5,346 | `mova` | `C0000111` | SME MOVA
5,077 / 4,976 / 4,827 | `eor`/`orr`/`and` | — | SVE logical (imm/reg) missing reserved checks
4,566 / 3,011 / 2,883 / 2,870 | `add`/`sub`/`adds`/`subs` | — | SVE/base missing constraint checks
4,127 / 2,227 | `fmlsl`/`fmlsl2` | `0EE0EC2E` | NEON FMLAL/FMLSL accept `size<0>==1` which is undefined (known, pre-existing)

This is a separate, large hardening effort: walk the top REVERSE mnemonics, confirm via the ARM ARM which
bit-patterns are genuinely UNDEFINED, and add the guards — **without** regressing the binja corpus (the
corpus golden run is the gate: it must stay 100% coverage / 99.75% parity, since every corpus word is a
valid encoding). Highest value: the `fmopa`/`fmops` pair (~37k) and the SVE `mov`/`ext`/logical/add-sub set.

## 3. DISAGREEMENTS (14,863) — mostly noise, some real

Most are cosmetic (fARM64 UAL/binja style vs LLVM: hex-vs-decimal immediates, brace spacing, alias choice)
and are intentional. Real ones to look at: the FMLAL/FMLSL `size<0>` and FCVTXN `size==00` over-decodes
(also surface in REVERSE), and any case where the **mnemonic** (first token) differs — grep section (b) of
`target/llvm-diff.txt` and filter out pure radix/brace differences.

## 4. How to continue (the proven parallel recipe)

Each batch = N family agents in **isolated git worktrees** (concurrent, private build dirs → no contention),
then a sequential **integrator** that merges and unions the additive catalog edits. Proven on Batches E & F.

Per worktree agent:
- It is a fresh checkout of HEAD and LACKS the gitignored `refs/` corpus → set
  `FARM64_CORPUS=D:/binsnake/farm/refs/arch-arm64-master/disassembler/test_cases.txt` for golden/roundtrip.
- Implement decode (`src/decode/*`) + encode (`src/encode/*`); add `Code`/`Mnemonic` via the `codes!` macro +
  `tables/names.rs` in lockstep; new `Feature` variants are runtime-gated and in `FeatureSet::ALL`.
- Validate (FIX until green): `cargo build` + `--features "std full"` (0 warn), `cargo clippy --all-targets
  --features "std full"` (0), `cargo test --features "std full" --lib`, golden (must stay 100%/99.75%),
  roundtrip (100% semantic), and a targeted `llvm-mc` check on its mnemonics.
- Commit on a branch `wt/<batch>-<family>`. **Never** run `git checkout/reset/restore/clean` (it has
  destroyed uncommitted sibling work). Only `git add` + `git switch -C` + `git commit`.

Integration (sequential, in main tree which HAS `refs/`): `git merge --no-edit wt/<batch>-*` one at a time;
conflicts are PURELY ADDITIVE in the catalog files (`mnemonic.rs`, `tables/names.rs`, `operand.rs`,
`format/fmt_writer.rs`, `features.rs`, `encode/mod.rs`, `decode/mod.rs`) — resolve by UNIONING (keep both,
dedup true duplicates). Build after each merge. Then full verify + LLVM differential. Finally clean up
(`git worktree remove --force` + `git branch -D wt/*`; `.claude/` is gitignored) and `git reset --soft
<pre-batch-commit> && git commit` to squash to one tidy commit per batch.

**Suggested next batches** (by module area, so decode files stay disjoint): **G** = G1 (simd_fp matrix/
mlal/dot) + G2 (sve quadword/REVD/while) + G3 (sme LUT/MOVA) + G4 (ldst LSFE) — i.e. one more parallel
batch closes most of the 6,282 GAPs. Then a dedicated **hardening** pass for the REVERSE over-decodes (§2),
starting with `fmopa`/`fmops`.
