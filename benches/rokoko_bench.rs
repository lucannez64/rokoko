//! Rokoko ZKP benchmarks — vector commitment proof, analogous to VELA authentication.
//!
//! Cyclo (VELA auth) proves knowledge of one short ring element `sk` such that
//! the Ajtai commitment `A·sk = pk` holds in R_q (N=128, q=2^50-111).
//!
//! Rokoko proves the same conceptual relation — knowledge of a short witness
//! matrix whose structured commitment matches a public value — using a
//! multi-round sumcheck protocol over the same ring R_q[X]/(X^128+1).
//! The default p-28 witness is 2^14×2^8 ring elements; use `--features p-26`
//! for a 4× smaller/faster run when comparing against Cyclo.
//!
//! ## Timing scope
//!
//! `init_sumcheck` / `init_verifier` build large precomputed tables tied to
//! the CRS.  In production these are one-time amortized costs (created once,
//! shared across many proofs).  They are therefore placed **outside** the
//! timed region — only `commit + prover_round` and `verifier_round` are
//! measured, matching the structure of the cyclo bench (which only times the
//! FFI call).
//!
//! ## Pool warmup
//!
//! Rokoko uses a global `RingElement` pool.  On pool miss the code prints a
//! diagnostic line AND allocates fresh memory.  Running one untimed warmup
//! iteration fills the pool so the actual benchmark iterations see hits
//! instead of misses and diagnostic I/O.
//!
//! Build:
//!   cargo bench --bench rokoko_bench                  # p-28 (default)
//!   cargo bench --bench rokoko_bench --features p-26  # recommended for comparison

use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use num::range;
use rokoko::{
    common::{
        pool::{drain_pool, load_and_preallocate, save_access_stats},
        ring_arithmetic::{Representation, RingElement},
        structured_row::StructuredRow,
    },
    protocol::{
        config::{Config, SumcheckConfig, CONFIG},
        crs::CRS,
        open::evaluation_point_to_structured_row,
        params::{decompose_witness, witness_sampler},
        parties::{commiter::commit, prover::prover_round, verifier::verifier_round},
        sumcheck::init_sumcheck,
        sumchecks::builder_verifier::init_verifier,
    },
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn top_config() -> &'static SumcheckConfig {
    match &*CONFIG {
        Config::Sumcheck(c) => c,
        _ => panic!("expected SumcheckConfig at top level"),
    }
}

fn eval_inner(log: u32) -> Vec<StructuredRow> {
    vec![evaluation_point_to_structured_row(
        &range(0, log as usize)
            .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2))
            .collect::<Vec<_>>(),
    )]
}

fn eval_outer(log: u32) -> Vec<StructuredRow> {
    vec![evaluation_point_to_structured_row(
        &range(0, log as usize)
            .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2))
            .collect::<Vec<_>>(),
    )]
}

const POOL_STATS_FILE: &str = "target/rokoko_bench_pool_stats.txt";

/// Warmup: run one full prove+verify to discover every pool size the protocol
/// needs, save the access counts to a file, then reload them so the pool is
/// full for the first benchmark iteration.
///
/// Before each subsequent iteration the bench loop calls `refill_pool()` to
/// restore the same counts, ensuring every iteration starts with pool hits and
/// never triggers the diagnostic "pool miss" println.
fn warmup(
    crs: &CRS,
    config: &SumcheckConfig,
    witness_decomposed: &rokoko::common::matrix::VerticallyAlignedMatrix<RingElement>,
    ep_inner: &Vec<StructuredRow>,
    ep_outer: &Vec<StructuredRow>,
) {
    let mut ctx = init_sumcheck(crs, config);
    let mut vctx = init_verifier(crs, config);
    let (cwa, rc) = commit(crs, config, witness_decomposed);
    let (proof, claims) = prover_round(
        crs, config, &cwa, witness_decomposed, ep_inner, ep_outer, &mut ctx, true, None,
    );
    verifier_round(crs, config, &rc, &proof, ep_inner, ep_outer, &claims.unwrap(), &mut vctx, None);

    // Persist the access pattern so each bench iteration can reload it.
    save_access_stats(POOL_STATS_FILE).expect("failed to save pool stats");
    // Refill the pool for the first iteration.
    refill_pool();
}

/// Restore the pool to the state it had at the start of a fresh prove+verify.
/// Drains first so repeated calls don't accumulate memory unboundedly.
/// Called before each timed iteration so misses (and their diagnostic prints)
/// never appear during measurement.
fn refill_pool() {
    drain_pool();
    load_and_preallocate(POOL_STATS_FILE).expect("failed to load pool stats");
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

fn bench_prove(c: &mut Criterion) {
    let config = top_config();
    let crs = CRS::gen_crs(config.composed_witness_length, config.basic_commitment_rank + 2);

    let witness = witness_sampler();
    let witness_decomposed = decompose_witness(&witness);

    let ep_inner = eval_inner(witness_decomposed.height.ilog2());
    let ep_outer = eval_outer(witness_decomposed.width.ilog2());

    warmup(&crs, config, &witness_decomposed, &ep_inner, &ep_outer);

    let mut group = c.benchmark_group("rokoko");
    group.sample_size(10);

    group.bench_function("prove/vela_auth", |b| {
        // iter_custom: init_sumcheck (amortized one-time cost) is outside the
        // timed region; only commit + prover_round are measured.
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                refill_pool();
                let mut ctx = init_sumcheck(&crs, config);
                let start = std::time::Instant::now();
                let (cwa, _rc) = commit(&crs, config, &witness_decomposed);
                let (proof, _claims) =
                    prover_round(&crs, config, &cwa, &witness_decomposed, &ep_inner, &ep_outer, &mut ctx, true, None);
                total += start.elapsed();
                drop(proof);
                drop(cwa);
                drop(ctx);
            }
            total
        });
    });

    group.finish();
}

fn bench_verify(c: &mut Criterion) {
    let config = top_config();
    let crs = CRS::gen_crs(config.composed_witness_length, config.basic_commitment_rank + 2);

    let witness = witness_sampler();
    let witness_decomposed = decompose_witness(&witness);

    let ep_inner = eval_inner(witness_decomposed.height.ilog2());
    let ep_outer = eval_outer(witness_decomposed.width.ilog2());

    warmup(&crs, config, &witness_decomposed, &ep_inner, &ep_outer);

    // Generate one proof outside the loop (same pattern as cyclo_bench).
    let mut ctx = init_sumcheck(&crs, config);
    let (cwa, rc_commitment) = commit(&crs, config, &witness_decomposed);
    let (proof, claims) = prover_round(
        &crs, config, &cwa, &witness_decomposed, &ep_inner, &ep_outer, &mut ctx, true, None,
    );
    let claims = claims.unwrap();

    let mut group = c.benchmark_group("rokoko");
    group.sample_size(10);

    group.bench_function("verify/vela_auth", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                refill_pool();
                let mut vctx = init_verifier(&crs, config);
                let start = std::time::Instant::now();
                verifier_round(
                    &crs, config, &rc_commitment, &proof, &ep_inner, &ep_outer, &claims,
                    &mut vctx, None,
                );
                total += start.elapsed();
                drop(vctx);
            }
            total
        });
    });

    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let config = top_config();
    let crs = CRS::gen_crs(config.composed_witness_length, config.basic_commitment_rank + 2);

    let witness = witness_sampler();
    let witness_decomposed = decompose_witness(&witness);

    let ep_inner = eval_inner(witness_decomposed.height.ilog2());
    let ep_outer = eval_outer(witness_decomposed.width.ilog2());

    warmup(&crs, config, &witness_decomposed, &ep_inner, &ep_outer);

    let mut group = c.benchmark_group("rokoko");
    group.sample_size(10);

    group.bench_function("roundtrip/vela_auth", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                refill_pool();
                let mut ctx = init_sumcheck(&crs, config);
                let mut vctx = init_verifier(&crs, config);
                let start = std::time::Instant::now();
                let (cwa, rc) = commit(&crs, config, &witness_decomposed);
                let (proof, claims) = prover_round(
                    &crs, config, &cwa, &witness_decomposed, &ep_inner, &ep_outer,
                    &mut ctx, true, None,
                );
                verifier_round(
                    &crs, config, &rc, &proof, &ep_inner, &ep_outer, &claims.unwrap(),
                    &mut vctx, None,
                );
                total += start.elapsed();
                drop(proof);
                drop(cwa);
                drop(ctx);
                drop(vctx);
            }
            total
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_prove, bench_verify, bench_roundtrip,
}
criterion_main!(benches);
