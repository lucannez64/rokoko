use std::ops::IndexMut;

use crate::{
    common::{
        arithmetic::evaluation_point_to_structured_row,
        matrix::{HorizontallyAlignedMatrix, VerticallyAlignedMatrix, ZeroNew},
        ring_arithmetic::{Representation, RingElement},
        sampling::sample_random_vector,
        structured_row::{self, PreprocessedRow, StructuredRow},
    },
    protocol::crs::CRS,
};

pub struct Opening {
    // TODO: add recursive layers of commitments
    pub rhs: HorizontallyAlignedMatrix<RingElement>,
    pub evaluations: Vec<RingElement>,
    pub evaluation_points_inner: Vec<PreprocessedRow>,
    pub evaluation_points_outer: Vec<PreprocessedRow>,
}

pub fn open_at(
    witness: &VerticallyAlignedMatrix<RingElement>,
    evaluation_point_inner: &Vec<Vec<RingElement>>,
    evaluation_point_outer: &Vec<Vec<RingElement>>,
) -> Opening {
    assert_eq!(
        evaluation_point_inner[0].len(),
        witness.height.ilog2() as usize
    );
    assert_eq!(
        evaluation_point_outer[0].len(),
        witness.width.ilog2() as usize
    );

    let nof_evaluation_points = evaluation_point_inner.len();
    assert_eq!(evaluation_point_outer.len(), nof_evaluation_points);

    let structured_points_inner = evaluation_point_inner
        .iter()
        .map(|ep| evaluation_point_to_structured_row(ep))
        .collect::<Vec<StructuredRow>>();

    let preprocessed_points_inner = structured_points_inner
        .into_iter()
        .map(|sr| PreprocessedRow::from_structured_row(sr))
        .collect::<Vec<PreprocessedRow>>();

    let structured_points_outer = evaluation_point_outer
        .iter()
        .map(|ep| evaluation_point_to_structured_row(ep))
        .collect::<Vec<StructuredRow>>();

    let preprocessed_points_outer = structured_points_outer
        .into_iter()
        .map(|sr| PreprocessedRow::from_structured_row(sr))
        .collect::<Vec<PreprocessedRow>>();

    let mut rhs = HorizontallyAlignedMatrix::new_zero(
        evaluation_point_inner.len(),
        witness.width,
        &RingElement::zero(Representation::IncompleteNTT),
    );

    for (i, preprocessed_row_inner) in preprocessed_points_inner.iter().enumerate() {
        let mut temp = RingElement::zero(Representation::IncompleteNTT);
        for col in 0..witness.width {
            for (elem, w_elem) in preprocessed_row_inner
                .preprocessed_row
                .iter()
                .zip(witness.col(col).iter())
            {
                temp *= (elem, w_elem);
                *rhs.index_mut((i, col)) += &temp;
            }
        }
    }

    let mut evaluations = Vec::with_capacity(evaluation_point_outer.len());

    for (i, preprocessed_row_outer) in preprocessed_points_outer.iter().enumerate() {
        let mut temp = RingElement::zero(Representation::IncompleteNTT);
        for (elem, w_elem) in preprocessed_row_outer
            .preprocessed_row
            .iter()
            .zip(rhs.row(i).iter())
        {
            temp *= (elem, w_elem);
        }
        evaluations.push(temp);
    }

    Opening {
        rhs,
        evaluations,
        evaluation_points_inner: preprocessed_points_inner,
        evaluation_points_outer: preprocessed_points_outer,
    }
}
