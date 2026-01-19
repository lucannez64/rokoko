use std::ops::Index;

use crate::common::{
    hash::HashWrapper,
    matrix::{VerticallyAlignedMatrix, ZeroNew},
};

static PROJECTION_BASE_HEIGHT: usize = 128;

// Static storage for return values to avoid returning references to temporaries
static FALSE_FALSE: (bool, bool) = (false, false);
static FALSE_TRUE: (bool, bool) = (false, true);
static TRUE_FALSE: (bool, bool) = (true, false);
static TRUE_TRUE: (bool, bool) = (true, true);

#[derive(Clone)]
pub struct ProjectionSquare {
    // Row-wise aligned: each byte stores 8 columns of one row
    pub data_sign: [u8; PROJECTION_BASE_HEIGHT * PROJECTION_BASE_HEIGHT / 8],
    pub data_value: [u8; PROJECTION_BASE_HEIGHT * PROJECTION_BASE_HEIGHT / 8],
}
pub struct ProjectionMatrix {
    pub projection_height: usize,
    pub projection_width: usize,
    pub projection_ratio: usize,
    pub projection_data: VerticallyAlignedMatrix<ProjectionSquare>,
}

impl ProjectionMatrix {
    pub fn new(projection_ratio: usize, projection_height: usize) -> Self {
        debug_assert!(
            projection_height % PROJECTION_BASE_HEIGHT == 0,
            "projection_height must be multiple of PROJECTION_BASE_HEIGHT"
        );
        ProjectionMatrix {
            projection_height,
            projection_width: projection_height * projection_ratio,
            projection_ratio,
            projection_data: VerticallyAlignedMatrix::new_zero(
                projection_height / PROJECTION_BASE_HEIGHT,
                projection_height * projection_ratio / PROJECTION_BASE_HEIGHT,
                &ProjectionSquare {
                    data_sign: [0u8; PROJECTION_BASE_HEIGHT * PROJECTION_BASE_HEIGHT / 8],
                    data_value: [0u8; PROJECTION_BASE_HEIGHT * PROJECTION_BASE_HEIGHT / 8],
                },
            ),
        }
    }

    /// Get masks for 8 consecutive columns at a single row
    /// Returns (k_pos, k_inc) where each bit represents one column
    /// This directly reads the natural storage format (8 columns per byte)
    #[inline]
    pub fn get_row_masks_u8(&self, row: usize, col_base: usize) -> (u8, u8) {
        debug_assert!(col_base % 8 == 0, "col_base must be aligned to 8");
        let square_row = row / PROJECTION_BASE_HEIGHT;
        let square_col = col_base / PROJECTION_BASE_HEIGHT;
        let inner_row = row % PROJECTION_BASE_HEIGHT;
        let inner_col_base = col_base % PROJECTION_BASE_HEIGHT;

        let square = &self.projection_data[(square_row, square_col)];

        // Each byte stores 8 consecutive columns for one row
        // byte_index = row * (bytes_per_row) + which_byte_in_row
        let bytes_per_row = PROJECTION_BASE_HEIGHT / 8;
        let byte_index = inner_row * bytes_per_row + inner_col_base / 8;
        let k_pos = square.data_sign[byte_index];
        let k_inc = square.data_value[byte_index];

        (k_pos, k_inc)
    }

    #[cfg(test)]
    pub fn from_i8(data: Vec<Vec<i8>>) -> Self {
        let projection_height = data.len();
        let projection_width = data[0].len();
        let projection_ratio = projection_width / projection_height;
        let mut projection_data = VerticallyAlignedMatrix::new_zero(
            projection_height / PROJECTION_BASE_HEIGHT,
            projection_width / PROJECTION_BASE_HEIGHT,
            &ProjectionSquare {
                data_sign: [0u8; PROJECTION_BASE_HEIGHT * PROJECTION_BASE_HEIGHT / 8],
                data_value: [0u8; PROJECTION_BASE_HEIGHT * PROJECTION_BASE_HEIGHT / 8],
            },
        );
        for outer_col in 0..projection_data.width {
            for row in 0..projection_data.height {
                let mut square = ProjectionSquare {
                    data_sign: [0u8; PROJECTION_BASE_HEIGHT * PROJECTION_BASE_HEIGHT / 8],
                    data_value: [0u8; PROJECTION_BASE_HEIGHT * PROJECTION_BASE_HEIGHT / 8],
                };
                for inner_row in 0..PROJECTION_BASE_HEIGHT {
                    for inner_col in 0..PROJECTION_BASE_HEIGHT {
                        let value = data[row * PROJECTION_BASE_HEIGHT + inner_row]
                            [outer_col * PROJECTION_BASE_HEIGHT + inner_col];
                        let (is_positive, is_non_zero) = match value {
                            0 => (false, false),
                            1 => (true, true),
                            -1 => (false, true),
                            _ => panic!("Invalid value in projection matrix"),
                        };
                        // Row-wise storage: each row spans multiple bytes (PROJECTION_BASE_HEIGHT / 8 bytes per row)
                        let bytes_per_row = PROJECTION_BASE_HEIGHT / 8;
                        let byte_index = inner_row * bytes_per_row + inner_col / 8;
                        let bit_offset = inner_col % 8;
                        if is_positive {
                            square.data_sign[byte_index] |= 1 << bit_offset;
                        }
                        if is_non_zero {
                            square.data_value[byte_index] |= 1 << bit_offset;
                        }
                    }
                }
                projection_data[(row, outer_col)] = square;
            }
        }
        ProjectionMatrix {
            projection_height,
            projection_width,
            projection_ratio,
            projection_data,
        }
    }

    pub fn sample(&mut self, hash_wrapper: &mut HashWrapper) {
        for square in self.projection_data.data.iter_mut() {
            hash_wrapper.fill_from_xof(b"projection-square-sign", &mut square.data_sign);
            hash_wrapper.fill_from_xof(b"projection-square-value", &mut square.data_value);
        }
    }
}

impl Index<(usize, usize)> for ProjectionSquare {
    type Output = (bool, bool);

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        // Row-wise storage: each row spans multiple bytes (PROJECTION_BASE_HEIGHT / 8 bytes per row)
        let bytes_per_row = PROJECTION_BASE_HEIGHT / 8;
        let byte_index = row * bytes_per_row + col / 8;
        let bit_offset = col % 8;
        let is_positive = (self.data_sign[byte_index] >> bit_offset) & 1 == 1;
        let is_non_zero = (self.data_value[byte_index] >> bit_offset) & 1 == 1;
        match (is_positive, is_non_zero) {
            (false, false) => &FALSE_FALSE,
            (false, true) => &FALSE_TRUE,
            (true, false) => &TRUE_FALSE,
            (true, true) => &TRUE_TRUE,
        }
    }
}

impl Index<(usize, usize)> for ProjectionMatrix {
    // { -1, 0, 1 } is represented as (is_positive, is_non_zero), which automatically imposes a desired bias towards 0
    type Output = (bool, bool);

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        let square_row = row / PROJECTION_BASE_HEIGHT;
        let square_col = col / PROJECTION_BASE_HEIGHT;
        let inner_row = row % PROJECTION_BASE_HEIGHT;
        let inner_col = col % PROJECTION_BASE_HEIGHT;
        &self.projection_data[(square_row, square_col)][(inner_row, inner_col)]
    }
}

#[cfg(test)]
mod tests {
    use crate::common::ring_arithmetic::{Representation, RingElement};

    use super::*;

    #[test]
    fn test_stability_of_sampling() {
        let mut hash_wrapper = HashWrapper::new();
        let mut projection_matrix_1 = ProjectionMatrix::new(4, PROJECTION_BASE_HEIGHT * 4);
        println!(
            "Matrix 1: height={}, width={}, data.height={}, data.width={}",
            projection_matrix_1.projection_height,
            projection_matrix_1.projection_width,
            projection_matrix_1.projection_data.height,
            projection_matrix_1.projection_data.width
        );
        projection_matrix_1.sample(&mut hash_wrapper);

        let mut hash_wrapper_2 = HashWrapper::new();
        let mut projection_matrix_2 = ProjectionMatrix::new(4, PROJECTION_BASE_HEIGHT * 4);
        projection_matrix_2.sample(&mut hash_wrapper_2);

        for outer_col in 0..PROJECTION_BASE_HEIGHT * 16 {
            for row in 0..PROJECTION_BASE_HEIGHT * 4 {
                debug_assert_eq!(
                    projection_matrix_1[(row, outer_col)],
                    projection_matrix_2[(row, outer_col)]
                );
            }
        }
    }

    #[test]
    fn test_instability_with_different_transcript() {
        let mut hash_wrapper = HashWrapper::new();
        let mut projection_matrix_1 = ProjectionMatrix::new(4, PROJECTION_BASE_HEIGHT * 4);
        projection_matrix_1.sample(&mut hash_wrapper);

        let mut hash_wrapper_2 = HashWrapper::new();
        hash_wrapper_2
            .update_with_ring_element(&RingElement::constant(42, Representation::IncompleteNTT));
        let mut projection_matrix_2 = ProjectionMatrix::new(4, PROJECTION_BASE_HEIGHT * 4);
        projection_matrix_2.sample(&mut hash_wrapper_2);

        let mut differences_found = 0;
        for col in 0..PROJECTION_BASE_HEIGHT * 4 * 4 {
            for row in 0..PROJECTION_BASE_HEIGHT {
                if projection_matrix_1[(row, col)] != projection_matrix_2[(row, col)] {
                    differences_found += 1;
                }
            }
        }
        debug_assert!(differences_found > 0);
    }

    #[test]
    fn test_indexing() {
        let mut data = vec![vec![0i8; PROJECTION_BASE_HEIGHT * 4]; PROJECTION_BASE_HEIGHT];
        data[0][0] = 1;
        data[3][1] = -1;
        data[1][4] = 1;
        data[2][3] = 0;
        let projection_matrix = ProjectionMatrix::from_i8(data);
        debug_assert_eq!(projection_matrix[(0, 0)], (true, true));
        debug_assert_eq!(projection_matrix[(3, 1)], (false, true));
        debug_assert_eq!(projection_matrix[(1, 4)], (true, true));
        debug_assert_eq!(projection_matrix[(2, 3)], (false, false));
    }
}
