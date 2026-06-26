# fARM64 — Remaining extensions / gaps (next-session handoff)

Snapshot after the **G + extra + H** extension batches (2026-06-26). The §1 GAP worklist from the
previous handoff is essentially closed; this revision records what landed and re-prioritises the
remaining tail from a **fresh LLVM-22 differential survey** (`clang -c .inst` + `llvm-objdump
--mattr=+all`, used as the oracle on this machine since `llvm-mc` isn't installed).

## How this session validated (no corpus on this box)
The gitignored binja corpus (`refs/test_cases.txt`) is absent here, so `tests/golden.rs` /
`roundtrip.rs` cannot run. Everything below was validated **against LLVM** instead, via a
clang+objdump oracle wrapper (`H:/projects/farm64/oracle.py`: `dec 0xWORD…` / `enc "mnem ops"…`)
and per-feature integration tests. Each batch landed only after build + clippy (0 warnings, all
cargo-feature configs) + lib tests + a region differential showing 0 valid/invalid + 0 mnemonic
disagreements (over-decode fixes additionally proven 0-regression via pre/post diffs).

## Progress this session (all on `main`, pushed)

Commit | What
|-|-|
`b98b6d8` | **G4** FEAT_LSFE atomic FP in-memory (`LDF*/STF*/LDBF*/STBF*`, H/S/D+BF16, a/l/al)
`09e87a1` | **G1** NEON int matrix-mul (SMMLA/UMMLA/USMMLA), SVE FP8 MLAL z-form, SVE BF16 indexed, multi-vec FMUL
`d6903e1` | **G2** SVE2.1 `.q` ld/st (ld1q/st1q gather + ld2q–4q/st2q–4q), REVD `/z` (+merging size guard), WHILE pred-pair/`pn`, SVE FP unary `/z` convert
`aabe20e` | **G3** SME2 LUTI2/4 via ZT0 (consecutive+strided dests, pair source; new `Register::Zt0`), SME2 multi-vector ZA tile-slice MOV/MOVAZ
`c75c7d6` | NEON FP/BF16/FP8 **FMMLA/BFMMLA** (F16F32MM/F16MM/F8F16MM/F8F32MM) + fixes the FCADD `lo=111011` over-decode
`7b828d3` | **H1** FEAT_HBC `BC.<cond>` (was mis-rendered `b.<cond>`, ~1200) + FEAT_PAuth_LR `*SPPC` branches
`6b59eb8` | **H2** SME outer-product rewrite: over-decode + mnemonic repair (bmopa/bfmopa/umopa…) + FEAT_SME_MOP4
`28ac334` | **H3** SVE: `.q` single-reg ld1w/d/st1w/d, BF16 predicated (bf{mla,mls,mul,add,sub,clamp}+f clamp), PSEL (kills dup-shadow), FP8 MLAL **vector**+FMMLA+BFMMLA+full FDOT (bit23 fix), bfmlslb/t, sabal/uabal, frint32/64 z/x `/z`, lastp/firstp
`ff7bab4` | **H4** over-decode hardening: add/sub extended `opt!=00`, NEON ld/st-structure `.1d`/reserved, FP16-MLAL `size<0>`; + FEAT_RPRFM

### Measured impact (identical 614,400-word random+structured sample, `--mattr=+all`)
Metric | Before (prev handoff start) | After this session
|-|-|-|
LLVM **GAPS** (LLVM decodes, fARM64 `Invalid`) | 662 / 87 mnem | **205 / 54 mnem**  (−69%)
LLVM **REVERSE** (fARM64 decodes, LLVM rejects = over-decode) | 19,199 / 264 | **9,199 / 228**  (−52%)
**DISAGREEMENTS** (mnemonic differs) | 1,653 / 52 | **69 / 14**  (−96%)

(Sample is ~1/8 the size of the 5M `llvm_diff.rs` run, so treat counts as relative magnitudes.)
Regenerate: dump fARM64 over a sweep, run the oracle on the same words, bucket by mnemonic
(scratch `survey.py`).

## 1. Remaining GAPs (missing — 205 in-sample, prioritised)
Family | Example word | LLVM | area | notes
|-|-|-|-|-|
NEON FP16 2-way `fdot` | `0E85FF12` | `fdot v18.2s, v24.4h, v5.4h` | simd_fp | FEAT_FP16FML-style 2-way; vector + by-element (H4 deferred)
SVE2.1 `addqp`/`addsubp` | `043779B1`/`04E27D59` | `addqp z.b,z.b,z.b` | sve | quadword pair add / add-sub
SME2 `luti6` | `4526AFD1` | `luti6 z.b, {z,z}, z` | sme/sve_lut | G3 deferred
SVE FP8 `fmmla` (.s,.h,.h) | `6430E7F3` | `fmmla z19.s, z31.h, z16.h` | sve | a remaining FP8/FP16 SVE matrix form
SVE `udot`/`sdot` (.h,.b,.b) | `44560750`/`4453012F` | `udot z.h,z.b,z.b` | sve | 2-way byte dot to `.h`
SVE `sqabs`/`sqneg` (pred) | `440ABD9E`/`44CBA737` | `sqabs z.b,p/z,z.b` | sve
SVE CPA `madpt`/`mlapt`/`subp` | `44D7DA82`/`44CFD317`/`4490A1E9` | — | sve | FEAT_CPA `.d` predicated/MAC
SVE `famax`/`famin` (pred) | `658E8BF9`/`658F96F1` | `famax z.s,p/m,z.s,z.s` | sve | FAMINMAX SVE
SVE `frint64z`(merging) | `6516A0FB` | `frint64z z.d,p/m,z.d` | sve | the `/m` FRINTTS forms (we added `/z`)
SVE multi-vec `fmul` 1-mult | `C1F8E8C6` | `fmul {z.d,z.d},{z.d,z.d},z.d` | sve/sme2 | single-multiplier form (H3 deferred)
FEAT_LRCPC3 `ldapp`/`ldap`/`stlp` | `D9527BB3`/`D94A5981`/`D9125A54` | `ldapp x,x,[x]` | ldst | ordered ld/st-pair forms
scalar `fcvtms`(h→s)… | `1EF40243` | `fcvtms s3, h18` | simd_fp | a few scalar FP convert width pairs
SVE `aesdimc`/AES-q | `4533EE14` | `aesdimc {z,z},{z,z},z.q[i]` | sve | FEAT_SVE_AES2 multi-vector
…~40 more distinct, ≤3 each (long tail).

## 2. Remaining over-decodes (REVERSE — 9,199 in-sample, prioritised)
Dominated now by **SVE/SME load-store reserved encodings** (the base/NEON/SME-outer-product
offenders were fixed in H2/H4). Each row = LLVM rejects, fARM64 still decodes:

count | mnem | example | likely cause
|-|-|-|-|
752 / 749 | `ld1q`/`st1q` | `E14CDA26` `ld1q z0v.q[w14],…` | **SME ZA-array** ld1q/st1q `.q` slice — reserved sub-forms accepted (H2 left as secondary)
488 / 485 | `ld1d`/`ldff1d` | `85BBCD88` `ld1d {z.s},p/z,[z.s,#imm]` | SVE 32-bit-gather vector+imm: reserved msz/offset combos
343 / 336 / 153 / 151 | `st1b`/`st1d`/`st1h`/`st1w` | `E03779B1` `st1b z0h.b[w15,#1],…` | **SME ZA-array** store: reserved
337 / 336 | `str`/`ldr` | `E13779B1` `str za[w15,#1]`, `85921796` `ldr p6,[…]` | SME ZA / SVE predicate ld/st reserved
197 / 195 / 185 / 180 | `prfh`/`prfd`/`prfb`/`prfw` | `84623A9B` gather-prefetch | SVE gather prefetch: reserved msz/`#imm`
189 / 188 / 114 | `ldff1sw`/`ld1sw`/`ldnt1sw` | `85623A9B` | SVE gather: reserved
185 | `mov` | `05156075` `mov z.b,p/m,#0x300` | SVE CPY/MOV-imm: out-of-range/reserved imm
156 / 154 / 152 | `ld1b`/`ld1h`/`ld1w` | `A47F528D` `ld1b {z.d},p/z,[x,xzr]` | **SVE contiguous with `Xm==xzr`** is the imm form → UNDEFINED (same class as the `.q` xzr guard added in H3; apply broadly)
… | `ext`,`cmp*`,`and/orr/eor`-imm, structured `mul vl`, `ld1rq*` | — | smaller SVE reserved-field gaps

Recipe: for each family, sweep the field that LLVM rejects through the oracle, add the guard, and
prove 0-regression with a pre/post diff (build the prior commit in a throwaway worktree, dump the
region with both, classify each changed word vs LLVM — the method used to land H3/H4 cleanly).
Biggest single wins: the SME ZA-array ld1q/st1q/st1b/st1d/str/ldr block (~2.85k) and the SVE
gather/prefetch block (~2.2k), then the SVE-contiguous-`xzr` guard (~0.6k).

## 3. DISAGREEMENTS (69 in-sample, 14 distinct) — mostly intentional
`mova` vs `mov` and `sxtl/uxtl` vs `sshll/ushll #0` and `bfc` vs `bfi …,wzr` are deliberate alias
choices (binja/UAL style); the `z<n>` vs `za<n>` ZA-tile prefix is the intentional binja spelling.
Real ones worth fixing: single-vector `mova`→`movaz` (`C0420228`, H2 left it), and the SVE2.1
multi-vector narrowing shifts where fARM64 emits the single `sqshrnb`/`uqrshrnb`/`shrnb` instead of
the `{z,z}` two-vector `sqshrn`/`uqrshrn`/`uqshrn` (`45BC0316` etc.), and `sqincp`→`cntp pn,vlx`.

## 4. How to continue (proven recipe)
Parallel family agents in isolated git worktrees (one decode module each, so files stay disjoint;
the catalog files `mnemonic.rs`/`tables/names.rs`/`features.rs`/`operand.rs` union-merge), then a
sequential integrator that re-validates each branch against the LLVM oracle (independent
differential — this caught real misses the agents missed) before squash-merging + pushing. The
agent playbook is `H:/projects/farm64/AGENT_GUIDE.md`; the oracle is `H:/projects/farm64/oracle.py`.
**Suggested next batch (I):** I1 = SME ZA-array ld/st over-decode; I2 = SVE gather/prefetch
over-decode + SVE-contiguous-`xzr` guard; I3 = the GAP tail (fdot, addqp/addsubp, luti6, SVE
udot/sdot/sqabs/sqneg/famax/famin, LRCPC3 ldapp/stlp, CPA madpt/mlapt). I3 is additive (low risk);
I1/I2 are over-decode hardening (prove 0-regression).
