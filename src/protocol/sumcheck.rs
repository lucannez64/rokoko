use core::hash;
use std::cell::RefCell;

use num::range;

use crate::{
    common::{
        arithmetic::inner_product,
        decomposition::{
            compose_from_decomposed, get_composer_offset, get_decomposed_offset_scaled,
        },
        hash::HashWrapper,
        projection_matrix::{self, ProjectionMatrix},
        ring_arithmetic::{Representation, RingElement},
    },
    protocol::{
        commitment::Prefix,
        config::{slice_by_prefix, Config},
        crs,
        sumcheck_utils::{
            common::HighOrderSumcheckData, diff::DiffSumcheck, linear::LinearSumcheck,
            polynomial::Polynomial, product::ProductSumcheck, selector_eq::SelectorEq,
        },
    },
};

// Computes the tensor product of two vectors of RingElements.
// e.g. (a0, a1) ⊗ (b0, b1) = (a0*b0, a0*b1, a1*b0, a1*b1)
fn tensor_product(a: &Vec<RingElement>, b: &Vec<RingElement>) -> Vec<RingElement> {
    let mut result = Vec::with_capacity(a.len() * b.len());
    for a_elem in a.iter() {
        for b_elem in b.iter() {
            result.push(a_elem * b_elem);
        }
    }
    result
}

// This is domain-specific sumcheck executor implementations
// for various sumcheck protocols used in the main protocol.

// SUMCHECK
// we want to check that
// ck \cdot folded_witness - commitment \cdot fold_challenge = 0
// outer_evaluation_points \cdot folded_witness - opening \cdot fold_challenge = 0
// <opening, inner_evaluation_points> - evaluations = 0
// I \otimes projection_matrix \cdot folded_witness - projection_image \cdot fold_challenge = 0
// rc_projection_image, rc_opening, rc_commitment are well-formed
// <w, conj(w)> + <y, conj(y)> - t = 0

pub fn sumcheck(
    crs: &crs::CRS,
    config: &Config,
    combined_witness: &Vec<RingElement>,
    projection_matrix: &ProjectionMatrix,
    folding_challenges: &Vec<RingElement>,
    hash_wrapper: &mut HashWrapper,
) {
    // TODO sumcheck protocol implementation

    // Let's check first ck \cdot folded_witness - commitment \cdot fold_challenge = 0

    let ck = crs
        .ck_for_wit_dim(config.witness_height)
        .iter()
        .take(config.basic_commitment_rank)
        .collect::<Vec<_>>();

    let composition_basis_basic_commitment =
        range(0, config.commitment_recursion.decomposition_chunks)
            .map(|i| {
                RingElement::constant(
                    1u64 << (config.commitment_recursion.decomposition_base_log as u64 * i as u64),
                    Representation::IncompleteNTT,
                )
            })
            .collect::<Vec<RingElement>>();

    let composition_basis_witness = range(0, config.witness_decomposition_chunks)
        .map(|i| {
            // Basis element corresponding to 2^{base_log * i}
            RingElement::constant(
                1u64 << (config.witness_decomposition_base_log as u64 * i as u64),
                Representation::IncompleteNTT,
            )
        })
        .collect::<Vec<RingElement>>();

    let row_zero_commitment_prefix = Prefix {
        prefix: config.commitment_recursion.prefix.prefix * config.basic_commitment_rank, // because we have rank 2
        length: config.commitment_recursion.prefix.length
            + config.basic_commitment_rank.ilog2() as usize, // if the rank is 2, then 2.ilog(2) = 1, so we add 1 to length
    };

    let mut left_2 = inner_product(
        &tensor_product(&ck[0].preprocessed_row, &composition_basis_witness),
        &slice_by_prefix(&combined_witness, &config.folded_witness_prefix),
    );

    let left_offset = ck[0].preprocessed_row.iter().fold(
        RingElement::zero(Representation::IncompleteNTT),
        |acc, x| {
            &acc + &(x * &RingElement::all(
                get_composer_offset(
                    config.witness_decomposition_base_log as u64,
                    config.witness_decomposition_chunks,
                ),
                Representation::IncompleteNTT,
            ))
        },
    );

    left_2 -= &left_offset;

    let mut right_2 = inner_product(
        &tensor_product(&folding_challenges, &composition_basis_basic_commitment),
        &slice_by_prefix(&combined_witness, &row_zero_commitment_prefix),
    );

    let right_offset = folding_challenges.iter().fold(
        RingElement::zero(Representation::IncompleteNTT),
        |acc, x| {
            &acc + &(x * &RingElement::all(
                get_composer_offset(
                    config.commitment_recursion.decomposition_base_log as u64,
                    config.commitment_recursion.decomposition_chunks,
                ),
                Representation::IncompleteNTT,
            ))
        },
    );

    right_2 -= &right_offset;

    assert_eq!(left_2, right_2);

    assert_eq!(config.composed_witness_length, combined_witness.len());
    // finally we write if as a sumcheck claim

    let mut combined_witness_sumcheck = RefCell::new(LinearSumcheck::<RingElement>::new(
        config.composed_witness_length,
    ));

    combined_witness_sumcheck
        .borrow_mut()
        .load_from(combined_witness);

    let mut witness_selector_sumcheck = RefCell::new(SelectorEq::<RingElement>::new(
        config.folded_witness_prefix.prefix,
        config.folded_witness_prefix.length,
        config.composed_witness_length.ilog2() as usize,
    ));

    let mut witness_combiner_sumcheck = RefCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
            composition_basis_witness.len(),
            config.composed_witness_length.ilog2() as usize
                - composition_basis_witness.len().ilog2() as usize,
            0,
        ),
    );

    witness_combiner_sumcheck
        .borrow_mut()
        .load_from(&composition_basis_witness);

    let mut commitment_zero_row_sumcheck = RefCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
            config.witness_height,
            config.composed_witness_length.ilog2() as usize
                - config.witness_height.ilog2() as usize
                - composition_basis_witness.len().ilog2() as usize,
            composition_basis_witness.len().ilog2() as usize,
        ),
    );

    commitment_zero_row_sumcheck
        .borrow_mut()
        .load_from(&ck[0].preprocessed_row);

    let mut constant_sumcheck = RefCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
            1,
            config.composed_witness_length.ilog2() as usize,
            0,
        ),
    );

    constant_sumcheck
        .borrow_mut()
        .load_from(&vec![RingElement::all(
            get_decomposed_offset_scaled(
                config.witness_decomposition_base_log as u64,
                config.witness_decomposition_chunks,
            ),
            // 0,
            Representation::IncompleteNTT,
        )]);

    let mut left_sumcheck_3 = RefCell::new(ProductSumcheck::new(
        &combined_witness_sumcheck,
        &witness_combiner_sumcheck,
    ));

    let mut left_sumcheck_2 = RefCell::new(DiffSumcheck::new(&left_sumcheck_3, &constant_sumcheck));

    let mut left_sumcheck_1 = RefCell::new(ProductSumcheck::new(
        &left_sumcheck_2,
        &commitment_zero_row_sumcheck,
    ));

    let mut left_sumcheck_0 = RefCell::new(ProductSumcheck::new(
        &witness_selector_sumcheck,
        &left_sumcheck_1,
    ));

    let mut poly = Polynomial::<RingElement>::new(1);

    left_sumcheck_0
        .borrow_mut()
        .univariate_polynomial_into(&mut poly);

    assert_eq!(&poly.at_zero() + &poly.at_one(), left_2);
    assert_eq!(poly.num_coefficients, 3);
}
