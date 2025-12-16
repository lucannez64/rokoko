use crate::common::{
    power_series::PowerSeries,
    ring_arithmetic::{incomplete_ntt_multiplication, RingElement},
    sampling::sample_random_vector,
    witness::WitnessMatrix,
};

/// Struct representing the Common Reference String (CRS) for cryptographic operations.
pub struct CRS {
    pub(crate) ck: Vec<PowerSeries>,
}

/// Generates a Common Reference String (CRS).
///
/// # Returns
///
/// A `CRS` containing commitment keys (`ck`) a randomly sampled vector (`a`), and a challenge set.
impl CRS {
    pub fn gen_crs(wit_dim: usize, module_size: usize) -> CRS {
        let v_module = sample_random_vector(module_size);

        let ck = compute_commitment_keys(v_module, wit_dim);

        CRS { ck }
    }
}

/// Computes commitment keys by raising the given module to successive powers.
///
/// # Arguments
///
/// * `module` - A vector of `RingElement`
/// * `wit_dim` - The witness dimension (width).
///
/// # Returns
///
/// A vector of vectors representing the computed commitment keys.
pub fn compute_commitment_keys(module: Vec<RingElement>, wit_dim: usize) -> Vec<PowerSeries> {
    module
        .iter()
        .map(|mut m| {
            let mut row = Vec::with_capacity(wit_dim);
            let mut power = m.clone();
            let mut result =
                RingElement::new(crate::common::ring_arithmetic::Representation::IncompleteNTT);
            row.push(m.clone());
            for _ in 1..wit_dim {
                incomplete_ntt_multiplication(&mut result, &mut power, &mut m);
                row.push(power.clone());
            }
            let mut ps = PowerSeries {
                full_layer: row.clone(),
                tensors: WitnessMatrix::new((wit_dim / 2) - 1, 0),
            };
            let mut current_dim = wit_dim;

            while current_dim % 2 == 0 {
                current_dim /= 2;
                let mut one = RingElement::one();
                one.from_even_odd_coefficients_to_incomplete_ntt_representation();
                let mut new_row = vec![RingElement::one(), row[current_dim - 1].clone()];
                ps.tensors.push_col(&mut new_row);
            }
            while current_dim % 2 == 0 {
                current_dim /= 2;

                let mut one = RingElement::one();
                one.from_even_odd_coefficients_to_incomplete_ntt_representation();

                let mut new_col = Vec::with_capacity(ps.tensors.height);

                for r in 0..ps.tensors.height {
                    let value = if r == 0 {
                        one.clone()
                    } else if r == 1 {
                        row[current_dim - 1].clone()
                    } else {
                        RingElement::zero(
                            crate::common::ring_arithmetic::Representation::IncompleteNTT,
                        )
                    };

                    new_col.push(value);
                }

                ps.tensors.push_col(&mut new_col);
            }
            ps
        })
        .collect()
}
