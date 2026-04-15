//! Cyclo ZKP benchmarks — modelled on VELA authentication.
//!
//! VELA uses Cyclo to authenticate devices.  During login the client proves,
//! in zero-knowledge, knowledge of a private key `cyclo_sk` (N=128 ring
//! coefficients) whose Ajtai commitment matches the enrolled `cyclo_pk` stored
//! on the server, bound to the current session challenge via a committed SHA-256
//! hash.
//!
//! Public inputs  (132 u64s):
//!   [0..128]   cyclo_pk — 128 LE-encoded u64 ring-element coefficients
//!   [128..132] committed_hash — 32-byte SHA-256 output reinterpreted as 4 u64s
//!
//! Private inputs (128 u64s):
//!   [0..128]   cyclo_sk — the prover's secret key
//!
//! The benchmarks exercise:
//!   - `prove`  — lattice-based ZK proof generation (~133 KB proof, ~2–3 s)
//!   - `verify` — proof verification (much cheaper than proving)
//!
//! All field elements are sampled uniformly in Z_q, q = 1125899906839937.
//!
//! Build:
//!   cargo bench --bench cyclo_bench --features cyclo

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

// ── Raw FFI (mirrors cyclo.h / vela_crypto::cyclo::ffi) ──────────────────────

mod ffi {
    use std::ffi::c_int;

    #[link(name = "cyclo", kind = "static")]
    #[link(name = "ntt_shim", kind = "static")]
    extern "C" {
        /// Returns positive proof byte-length on success; 1–6 on error.
        pub fn cyclo_prove(
            public_inputs: *const u64,
            public_len: usize,
            private_inputs: *const u64,
            private_len: usize,
            proof_out: *mut u8,
            proof_out_size: usize,
        ) -> c_int;

        /// Returns 1 if valid, 0 if invalid, negative on internal error.
        /// `private_len` must equal the count passed to `cyclo_prove`.
        pub fn cyclo_verify(
            public_inputs: *const u64,
            public_len: usize,
            private_len: usize,
            proof: *const u8,
            proof_len: usize,
        ) -> c_int;
    }
}

// ── Safe wrappers (same contract as vela_crypto::cyclo) ──────────────────────

/// Error code ceiling — return values above this encode the proof byte-length.
const MAX_ERR: i32 = 6;
/// Maximum proof buffer (512 KB matches the Zig-side constant).
const PROOF_BUF: usize = 512 * 1024;

fn prove(public: &[u64], private: &[u64]) -> Vec<u8> {
    let mut buf = vec![0u8; PROOF_BUF];
    let ret = unsafe {
        ffi::cyclo_prove(
            public.as_ptr(),
            public.len(),
            private.as_ptr(),
            private.len(),
            buf.as_mut_ptr(),
            buf.len(),
        )
    };
    assert!(ret > MAX_ERR, "cyclo_prove error code {ret}");
    buf.truncate(ret as usize);
    buf
}

fn verify(public: &[u64], private_len: usize, proof: &[u8]) -> bool {
    let ret = unsafe {
        ffi::cyclo_verify(
            public.as_ptr(),
            public.len(),
            private_len,
            proof.as_ptr(),
            proof.len(),
        )
    };
    assert!(ret >= 0, "cyclo_verify internal error {ret}");
    ret == 1
}

// ── Input construction ────────────────────────────────────────────────────────

/// Cyclo field modulus: a 50-bit prime used by the VELA PRESET_128.
const Q: u64 = 1125899906839937;
/// Ring degree N (VELA PRESET_128).
const N: usize = 128;
/// SHA-256 output as u64s (32 bytes / 8).
const HASH_U64S: usize = 4;

/// Deterministic LCG field-element sampler — reproducible without a heavy PRNG.
fn lcg_field_elems(seed: u64, count: usize) -> Vec<u64> {
    let mut s = seed;
    (0..count)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            s % Q
        })
        .collect()
}

/// VELA-auth input vectors:
///   public  = cyclo_pk (N u64s) || committed_hash (HASH_U64S u64s)
///   private = cyclo_sk (N u64s)
fn vela_inputs() -> (Vec<u64>, Vec<u64>) {
    let cyclo_pk       = lcg_field_elems(0xDEAD_BEEF_1234_5678, N);
    let committed_hash = lcg_field_elems(0xCAFE_BABE_8765_4321, HASH_U64S);
    let cyclo_sk       = lcg_field_elems(0xFEED_FACE_ABCD_EF01, N);

    let mut public = Vec::with_capacity(N + HASH_U64S);
    public.extend_from_slice(&cyclo_pk);
    public.extend_from_slice(&committed_hash);

    (public, cyclo_sk)
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

fn bench_prove(c: &mut Criterion) {
    let (public, private) = vela_inputs();

    let mut group = c.benchmark_group("cyclo");
    group.sample_size(10); // proving is expensive; 10 samples is sufficient

    group.bench_function("prove/vela_auth", |bencher| {
        bencher.iter(|| prove(black_box(&public), black_box(&private)));
    });

    group.finish();
}

fn bench_verify(c: &mut Criterion) {
    let (public, private) = vela_inputs();
    let proof = prove(&public, &private); // one-time setup outside the loop

    let mut group = c.benchmark_group("cyclo");
    group.sample_size(10); // verify is ~5 s; 10 samples keeps total runtime manageable

    group.bench_function("verify/vela_auth", |bencher| {
        bencher.iter(|| verify(black_box(&public), black_box(N), black_box(&proof)));
    });

    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let (public, private) = vela_inputs();

    let mut group = c.benchmark_group("cyclo");
    group.sample_size(10);

    group.bench_function("roundtrip/vela_auth", |bencher| {
        bencher.iter(|| {
            let proof = prove(black_box(&public), black_box(&private));
            black_box(verify(black_box(&public), black_box(N), &proof))
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
