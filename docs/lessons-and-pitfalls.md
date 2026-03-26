# Lessons Learned & Common Pitfalls

## Evaluation Points: The "+1 Skip" Trap
**Bug**: When the last round has no split (no layer variable), the verifier must use `evaluation_points_ring[outer_points_len..]` (NO +1 skip) to evaluate the folded witness. Intermediate rounds skip +1 because the layer variable selects top/bottom halves of the split. The last round has no split, so all inner variables must be included.

**Rule**: The prover computes claims using `evaluation_points[outer_points_len..]` from the CURRENT round's sumcheck. The verifier must use the same points. For intermediate rounds, `new_evaluation_points_inner = eval_points[outer_points_len+1..]` because one variable is consumed by the split. For the last round, use `eval_points[outer_points_len..]`.

## Decomposition Base Log Must Fit i16
**Bug**: decomposition_base_log=12 caused i16 overflow in witness decomposition.
**Fix**: Changed to decomposition_base_log=8 (values fit in i16 range ±128).

## Deref Pattern for Enum Variants
When refactoring a struct into an enum with shared fields, use the `Deref` pattern:
- Extract shared fields into `FooCommon` struct
- Implement `Deref<Target = FooCommon>` for the enum
- Variant-specific fields (like `decomposition_base_log`, `next`) require explicit `match`
- This avoids changing hundreds of `config.field` call sites

## The `temp *= (&a, &b)` Idiom
This is NOT temp *= a * b. It's `temp = a * b` using the MulAssign<(&A, &B)> impl. The pattern for inner products is:
```rust
let mut temp = RingElement::zero(Representation::IncompleteNTT);
let mut result = RingElement::zero(Representation::IncompleteNTT);
for (w, r) in data.iter().zip(points.iter()) {
    temp *= (w, r);   // temp = w * r
    result += &temp;   // result += w * r
}
```

## PreprocessedRow vs Raw Evaluation Points
- `evaluation_point_to_structured_row(&points)` → `StructuredRow` (tensor-factored form)
- `PreprocessedRow::from_structured_row(&structured_row)` → expanded MLE evaluation vector
- The `.preprocessed_row` field has `2^n` elements (the full MLE evaluated at the point)
- Use this for inner products with witness data

## Two Separate Protocols
The codebase has TWO protocol implementations:
1. **salsaa** (`src/salsaa/executor.rs`): The active one. Recursive rounds, VDF, enum configs.
2. **protocol** (`src/protocol/`): Older implementation. Has its own executor, prover, verifier, config.
Don't confuse them — they have different types and flows.

## const DEBUG: bool
Line 46 of executor.rs. Set to `true` for extensive assertions during development (costs performance). Set to `false` for production runs.

## VDF Sumcheck Structure
- VDF matrix M has structure: g on diagonal, a on sub-diagonal (both are 64-element vectors)
- Batched with challenge powers: (d' ⊗ h)^T · w = -y_0 + c^{2K} · y_t
- h = g + c·a (64 elements), d' = (1, c, c^2, ..., c^{2K-1})
- d'_sumcheck: `LinearSumcheck::new_with_prefixed_sufixed_data(2K, 1, 6)`
- h_sumcheck: `LinearSumcheck::new_with_prefixed_sufixed_data(64, total_vars - 6, 0)`
- Verifier evaluates MLE[d'](x) = ∏_i ((1-x_i) + x_i · c^{2^i}) via tensor structure

## Proof Size
Current p-26 config: ~3695 KB total proof
- Rounds 0-8 (intermediate): ~42-81 KB each (dominated by decomposed_split_commitment ~48 KB)
- Last round: ~3093 KB (dominated by folded_witness + projected_witness ~1536 KB each)
- The last round witness data is the proof size bottleneck

## Test Expectations
- 77/78 tests pass (1 known failure: `test_index_out_of_bounds` in release mode)
- Full end-to-end: prover ~12.4s, verifier ~9.5ms (p-26 config, release)
- Per-round verifier timing: intermediate rounds ~360-414µs, last round ~7.6ms
