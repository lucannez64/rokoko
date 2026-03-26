# Project: cowboys-and-aliens (rokoko)

## Build & Run
- `cargo build --release --features unsafe-sumcheck,p-26,incomplete-rexl`
- `cargo run --release --features unsafe-sumcheck,p-26,incomplete-rexl`
- `cargo test --release --features unsafe-sumcheck,p-26,incomplete-rexl -- --test-threads=1 --nocapture`
- 77 tests pass; 1 known failure: `common::matrix::tests::test_index_out_of_bounds` (doesn't panic in release)

## What This Is
Lattice-based proof system (SALSAA protocol). Recursive sumcheck rounds proving properties of a committed witness:
binariness (round 0), L2 norm (rounds 1+), VDF chain, projection consistency.

## Ring Arithmetic
- DEGREE=128 coefficients per ring element
- MOD_Q=1125899906839937 (~2^50)
- Representation::IncompleteNTT (primary working representation)
- `RingElement` supports `*=`, `+=`, conjugate, etc. The `*= (&a, &b)` pattern does `self = a * b`.

## Key File: src/salsaa/executor.rs (~2700 lines)
This is the main file. Contains everything: config, prover, verifier, proof structs.

### Core Types
- `RoundConfig` — enum: `Intermediate { common, decomposition_base_log, next }` | `Last { common }`
- `SalsaaProof` — enum: `Intermediate { common, new_claims, decomposed_split_commitment, next }` | `Last { common, folded_witness, projected_witness }`
- Both use `Deref` to their `Common` structs for transparent field access
- `RoundConfigCommon` fields: witness_length, exact_binariness, vdf, l2, projection_ratio, main_witness_columns, projection_prefix, main_witness_prefix, inner_evaluation_claims
- `SalsaaProofCommon` fields: projection_commitment, sumcheck_transcript, ip_l2_claim, ip_linf_claim, claims, claim_over_projection

### Round Structure
- `build_round_config()` recursively builds the config chain
- Round 0: 2 columns, VDF + exact_binariness, projection_ratio=2
- Rounds 1-N: 8 columns, L2 norm, projection_ratio=8
- Last round: no decomposition/split, sends folded_witness + projected_witness directly
- Currently produces 6 rounds (R0 first, R5 last) for p-26 config

### Prover Flow (prover_round)
1. Project witness → projection_commitment
2. Sample batching challenges, VDF challenge
3. Build extended_witness via paste_by_prefix
4. Load sumcheck context, run sumcheck (poly generation + partial evaluation loop)
5. Compute claims = <witness_col_i, eval_points_inner> for each column
6. Sample folding challenges → fold witness into 1 column
7. **Intermediate**: split → decompose (base_log=8) → commit → compute new_claims → recurse
8. **Last**: return folded_witness.data + projected_witness.data directly

### Verifier Flow (verifier_round)
1. Replay Fiat-Shamir (projection matrix, batching challenges, VDF challenge)
2. Verify norm claims (L2/Linf constant terms)
3. Replay sumcheck transcript → recover evaluation_points_ring
4. Compute folded_claim = Σ folding_challenges[i] * claims[(0,i)]
5. **Intermediate**: recompose claims from decomposition, check commitments, verify sumcheck eval, recurse
6. **Last**: evaluate folded_witness directly at eval points, check commitments via commit_basic, verify sumcheck eval

### Sumcheck Architecture
- `ProverSumcheckContext` / `VerifierSumcheckContext` — recursive (has `next: Option<Box<...>>`)
- `init_prover_sumcheck` / `init_verifier_sumcheck` — match on config variant for recursion
- `ElephantCell<T>` = Rc<RefCell<T>> (safe) or Rc<UnsafeCell<T>> (unsafe mode)
- Combiner batches multiple sumchecks; RingToFieldCombiner maps ring→field for Fiat-Shamir

### Sumcheck Types
- **Type1** (inner evaluation): Product(Product(inner_eval, outer_eval), main_witness) — verifies MLE claims
- **Type3** (projection): Diff(LHS_product, RHS_product) — verifies projection consistency
- **L2**: Product(witness, conjugated_witness) — proves ||w||^2
- **Linf**: Product((1-w), conjugated_w) — proves exact binariness
- **VDF**: Product(Product(step_powers, batched_row), main_witness) — proves VDF chain

### Witness Layout
- VerticallyAlignedMatrix: column-major, witness.col(i) gives column i
- Extended witness = 2x original (via paste_by_prefix with main_witness_prefix and projection_prefix)
- Prefix bits: MSB selects main_witness (0) vs projection (main_witness_columns)

### Commitment
- `commit_basic(crs, witness_matrix, rank) -> BasicCommitment` (= HorizontallyAlignedMatrix<RingElement>)
- `commit_basic_internal(eval_points, witness_matrix, rank) -> HorizontallyAlignedMatrix<RingElement>`
- RANK = 8 for salsaa

### Other Files (separate protocol)
- `src/protocol/` — older sumcheck protocol (different from salsaa), has its own executor, prover, verifier
- `src/protocol/config.rs` — `SizeableProof` trait, `SumcheckConfig`, `SimpleConfig`
- `src/protocol/parties/executor.rs` — the old executor using `Config` enum
- Don't confuse the two protocols!
