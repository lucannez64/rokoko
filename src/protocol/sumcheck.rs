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
        commitment::{self, Prefix},
        config::{slice_by_prefix, Config},
        crs::{self, CRS},
        sumcheck_utils::{
            common::HighOrderSumcheckData, diff::DiffSumcheck, linear::LinearSumcheck,
            polynomial::Polynomial, product::ProductSumcheck, selector_eq::SelectorEq,
        },
    },
};

fn composition_sumcheck(
    base_log: u64,
    chunks: usize,
    total_vars: usize,
) -> (
    RefCell<LinearSumcheck<RingElement>>,
    RefCell<LinearSumcheck<RingElement>>,
) {
    let conmposition_basis = range(0, chunks)
        .map(|i| {
            // Basis element corresponding to 2^{base_log * i}
            RingElement::constant(
                1u64 << (base_log as u64 * i as u64),
                Representation::IncompleteNTT,
            )
        })
        .collect::<Vec<RingElement>>();
    let combiner_sumcheck = RefCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
            conmposition_basis.len(),
            total_vars - conmposition_basis.len().ilog2() as usize,
            0,
        ),
    );

    combiner_sumcheck
        .borrow_mut()
        .load_from(&conmposition_basis);

    let mut witness_combiner_constant_sumcheck = RefCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(1, total_vars, 0),
    );

    witness_combiner_constant_sumcheck
        .borrow_mut()
        .load_from(&vec![RingElement::all(
            get_decomposed_offset_scaled(base_log as u64, chunks),
            // 0,
            Representation::IncompleteNTT,
        )]);

    (combiner_sumcheck, witness_combiner_constant_sumcheck)
}

fn sumcheck_from_prefix(prefix: &Prefix, total_vars: usize) -> RefCell<SelectorEq<RingElement>> {
    RefCell::new(SelectorEq::<RingElement>::new(
        prefix.prefix,
        prefix.length,
        total_vars,
    ))
}

fn ck_sumcheck(
    crs: &CRS,
    total_vars: usize,
    wit_dim: usize,
    i: usize,
    sufix: usize,
) -> RefCell<LinearSumcheck<RingElement>> {
    let ck = crs.ck_for_wit_dim(wit_dim);

    let mut sumcheck = RefCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
            wit_dim,
            total_vars - wit_dim.ilog2() as usize - sufix,
            sufix,
        ),
    );

    sumcheck.borrow_mut().load_from(&ck[i].preprocessed_row);

    sumcheck
}

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
    let total_vars = config.composed_witness_length.ilog2() as usize;

    let mut combined_witness_sumcheck = RefCell::new(LinearSumcheck::<RingElement>::new(
        config.composed_witness_length,
    ));

    combined_witness_sumcheck
        .borrow_mut()
        .load_from(combined_witness);

    let mut folded_witness_selector_sumcheck =
        sumcheck_from_prefix(&config.folded_witness_prefix, total_vars);

    let (mut folded_witness_combiner_sumcheck, mut witness_combiner_constant_sumcheck) =
        composition_sumcheck(
            config.witness_decomposition_base_log as u64,
            config.witness_decomposition_chunks,
            config.composed_witness_length.ilog2() as usize,
        );

    let mut commitment_key_zero_row_sumcheck = ck_sumcheck(
        crs,
        total_vars,
        config.witness_height,
        0,
        config.witness_decomposition_chunks.ilog2() as usize,
    );

    let mut left_sumcheck_3 = RefCell::new(ProductSumcheck::new(
        &combined_witness_sumcheck,
        &folded_witness_combiner_sumcheck,
    ));

    let mut left_sumcheck_2 = RefCell::new(DiffSumcheck::new(
        &left_sumcheck_3,
        &witness_combiner_constant_sumcheck,
    ));

    let mut left_sumcheck_1 = RefCell::new(ProductSumcheck::new(
        &left_sumcheck_2,
        &commitment_key_zero_row_sumcheck,
    ));

    let mut left_sumcheck_0 = RefCell::new(ProductSumcheck::new(
        &folded_witness_selector_sumcheck,
        &left_sumcheck_1,
    ));

    let mut poly = Polynomial::<RingElement>::new(1);

    left_sumcheck_0
        .borrow_mut()
        .univariate_polynomial_into(&mut poly);

    assert_eq!(poly.num_coefficients, 3);

    // Now, we proceed with rhs

    let mut folding_challenges_sumcheck = RefCell::new(
        LinearSumcheck::<RingElement>::new_with_prefixed_sufixed_data(
            config.witness_width,
            total_vars
                - config.witness_width.ilog2() as usize
                - config.commitment_recursion.decomposition_chunks.ilog2() as usize,
            config.commitment_recursion.decomposition_chunks.ilog2() as usize,
        ),
    );
    folding_challenges_sumcheck
        .borrow_mut()
        .load_from(&folding_challenges);

    let commitment_zero_row_prefix = Prefix {
        prefix: config.commitment_recursion.prefix.prefix * config.basic_commitment_rank,
        length: config.commitment_recursion.prefix.length
            + config.basic_commitment_rank.ilog2() as usize,
    };

    let commitment_zero_row_sumcheck =
        sumcheck_from_prefix(&commitment_zero_row_prefix, total_vars);

    let (mut commitment_combiner_sumcheck, mut commitment_combiner_constant_sumcheck) =
        composition_sumcheck(
            config.commitment_recursion.decomposition_base_log as u64,
            config.commitment_recursion.decomposition_chunks,
            config.composed_witness_length.ilog2() as usize,
        );

    let mut right_sumcheck_3 = RefCell::new(ProductSumcheck::new(
        &combined_witness_sumcheck,
        &commitment_combiner_sumcheck,
    ));

    let mut right_sumcheck_2 = RefCell::new(DiffSumcheck::new(
        &right_sumcheck_3,
        &commitment_combiner_constant_sumcheck,
    ));

    let mut right_sumcheck_1 = RefCell::new(ProductSumcheck::new(
        &right_sumcheck_2,
        &folding_challenges_sumcheck,
    ));

    let mut right_sumcheck_0 = RefCell::new(ProductSumcheck::new(
        &commitment_zero_row_sumcheck,
        &right_sumcheck_1,
    ));

    right_sumcheck_0
        .borrow_mut()
        .univariate_polynomial_into(&mut poly);

    let constraint_sumcheck = RefCell::new(DiffSumcheck::new(&left_sumcheck_0, &right_sumcheck_0));

    constraint_sumcheck
        .borrow_mut()
        .univariate_polynomial_into(&mut poly);

    assert_eq!(
        &poly.at_zero() + &poly.at_one(),
        RingElement::zero(Representation::IncompleteNTT)
    );
}
