use std::{hint::black_box, time::Duration};
use pilvi_tfhe::rings::*;
use pilvi_tfhe::codegen::*;
use criterion::{criterion_group, criterion_main, Criterion};
use pilvi_tfhe::pilvi::*;
use pilvi_tfhe::helpers::*;



fn bench_hexl(c: &mut Criterion) {
    
}


criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_hexl
}
criterion_main!(benches);