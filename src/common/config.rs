use std::sync::LazyLock;

pub static DEGREE: usize = 128;
pub static HALF_DEGREE: usize = 64;
pub static MOD_Q: u64 = 1125899906839937;

pub static ADDITION_SUBTRACTION_BUDGET: LazyLock<u64> = LazyLock::new(|| u64::MAX / (MOD_Q * 2)); // if we start from number HALF_WAY_MOD_Q how many additions/subtractions (with elements in [0,MOD_Q)) can we do without overflowing u64?

pub static HALF_WAY_MOD_Q: LazyLock<u64> = LazyLock::new(|| {
    let budget = u64::MAX / (MOD_Q * 2);
    budget * MOD_Q
});
pub static NOF_BATCHES: usize = 2;
