use crate::common::{
    power_series,
    ring_arithmetic::{
        addition, addition_in_place, incomplete_ntt_multiplication, Representation, RingElement,
    },
    witness::WitnessMatrix,
};

#[derive(Debug, Clone)]
pub struct PowerSeries {
    pub full_layer: Vec<RingElement>,
    pub tensors: WitnessMatrix<RingElement>,
}

pub fn dot_series_matrix(
    power_series: &[PowerSeries],
    matrix: &WitnessMatrix<RingElement>,
) -> WitnessMatrix<RingElement> {
    let n_series = power_series.len();
    let height = matrix.height;
    let width = matrix.width;

    let mut result = WitnessMatrix::new(height, n_series);

    let mut tmp = RingElement::zero(Representation::IncompleteNTT);

    for (ps_i, series) in power_series.iter().enumerate() {
        let layer = &series.full_layer[..width];

        for c in 0..width {
            let layer_c = &layer[c];
            let matrix_col = matrix.col(c);

            for r in 0..height {
                result[(ps_i, r)] = RingElement::zero(Representation::IncompleteNTT);
                incomplete_ntt_multiplication(&mut tmp, layer_c, &matrix_col[r]);
                addition_in_place(&mut result[(ps_i, r)], &tmp);
            }
        }
    }

    result
}
