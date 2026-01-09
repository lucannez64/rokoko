use std::{process::exit, sync::LazyLock};

use num::range;

use crate::{
    common::{
        decomposition::decompose,
        hash::HashWrapper,
        matrix::{new_vec_zero_preallocated, HorizontallyAlignedMatrix, VerticallyAlignedMatrix},
        projection_matrix::ProjectionMatrix,
        ring_arithmetic::{Representation, RingElement},
        sampling::sample_random_short_vector,
        structured_row::{self, PreprocessedRow, StructuredRow},
    },
    protocol::{
        commitment::{
            commit_basic, commit_basic_internal, recursive_commit, BasicCommitment, Prefix,
            RecursionConfig, RecursiveCommitment,
        },
        crs::{CK, CRS},
        fold::fold,
        open::{evaluation_point_to_structured_row, open_at, Opening},
        project::project,
    },
};

pub struct RoundOutput {
    folded_witness: VerticallyAlignedMatrix<RingElement>,
    projection_image: VerticallyAlignedMatrix<RingElement>,
    opening: Opening,
}

pub static CONFIG: LazyLock<Config> = LazyLock::new(|| Config {
    witness_height: 512,
    witness_width: 16,
    projection_ratio: 32,
    basic_commitment_rank: 2,
    nof_openings: 1,

    commitment_recursion: RecursionConfig {
        decomposition_radix_log: 15,
        decomposition_chunks: 4,
        rank: 1,
        next: None,
        prefix: Prefix {
            prefix: 0b1100,
            length: 4,
        }, // 2048 / 2^4 = 128
    },
    opening_recursion: RecursionConfig {
        decomposition_radix_log: 15,
        decomposition_chunks: 4,
        rank: 1,
        next: None,
        prefix: Prefix {
            prefix: 0b11010,
            length: 5,
        }, // 2048 / 2^5 = 64
    },
    projection_recursion: RecursionConfig {
        decomposition_radix_log: 15,
        decomposition_chunks: 2,
        rank: 1,
        next: None,
        prefix: Prefix {
            prefix: 0b10,
            length: 2,
        }, // 2048 / 2^2 = 512
    },

    folded_witness_prefix: Prefix {
        prefix: 0b0,
        length: 1,
    }, // 2048 / 2^1 = 1024
    witness_decomposition_chunks: 2,
    witness_decomposition_radix_log: 15,

    // committed basic_commitment_len = basic_commitment_rank * witness_width * commitment_recursion.decomposition_chunks = 2 * 16 * 4 = 128

    // committed projection_image_len = witness_height * witness_width / projection_ratio  * projection_recursion.decomposition_chunks = (512 * 16 / 32) * 2 = 512

    // committed opening_len = nof_openings * witness_width * opening_recursion.decomposition_chunks = 1 * 16 * 4 = 64

    // folded_witness len is witness_height * witness_decomposition_chunks = 512 * 2 = 1024

    // in total, we fit into 2048 elements per round
    composed_witness_length: 2048,

    next: None, // for multiple rounds
});

pub struct Config {
    witness_height: usize,
    witness_width: usize,
    projection_ratio: usize, // shall be likely the witness_height
    commitment_recursion: RecursionConfig,
    opening_recursion: RecursionConfig,
    projection_recursion: RecursionConfig,
    nof_openings: usize,

    witness_decomposition_radix_log: usize,
    witness_decomposition_chunks: usize,
    folded_witness_prefix: Prefix,

    basic_commitment_rank: usize,
    composed_witness_length: usize,

    next: Option<Box<Config>>, // for multiple rounds
}

// pub fn init_empty_recursive_commitment(config: &Vec<RecursionConfig>) -> RecursiveCommitment {

// }

fn paste_by_prefix(dest: &mut Vec<RingElement>, src: &Vec<RingElement>, prefix: &Prefix) {
    assert_eq!(src.len(), 1 << dest.len().ilog2() as usize - prefix.length);
    // e.g. if dest.len() = 2048, prefix.length = 4, prefix.prefix = 9 (0b1001)
    // then start = 9 << (11 - 4) = 9 << 7 = 1152 = 10010000000 index to start pasting
    let start = prefix.prefix << (dest.len().ilog2() as usize - prefix.length);
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dest.as_mut_ptr().add(start), src.len());
    }
}

fn paste_recursive_commitment(
    dest: &mut Vec<RingElement>,
    commitment: &RecursiveCommitment,
    config: &RecursionConfig,
) {
    paste_by_prefix(dest, &commitment.committed_data, &config.prefix);

    if let (Some(next_commitment), Some(next_config)) = (&commitment.next, &config.next) {
        paste_recursive_commitment(dest, next_commitment, next_config);
    }
}

pub fn prover_round(
    crs: &CRS,
    rc_commitment: &RecursiveCommitment,
    witness: &VerticallyAlignedMatrix<RingElement>,
    evaluation_points_inner: &Vec<Vec<RingElement>>,
    evaluation_points_outer: &Vec<Vec<RingElement>>,
) -> RoundOutput {
    let mut hash_wrapper = HashWrapper::new();

    hash_wrapper.update_with_ring_element_slice(&rc_commitment.most_inner_commitment());

    let opening = open_at(&witness, &evaluation_points_inner, &evaluation_points_outer);

    let rc_opening = recursive_commit(crs, &CONFIG.opening_recursion, &opening.rhs.data);

    hash_wrapper.update_with_ring_element_slice(&opening.rhs.data);

    let mut projection_matrix = ProjectionMatrix::new(CONFIG.projection_ratio);

    projection_matrix.sample(&mut hash_wrapper);

    let projection_image = project(&witness, &projection_matrix);

    let rc_projection_image =
        recursive_commit(&crs, &CONFIG.projection_recursion, &projection_image.data);

    hash_wrapper.update_with_ring_element_slice(&rc_projection_image.most_inner_commitment());

    let mut fold_challenge = vec![RingElement::zero(Representation::IncompleteNTT); witness.width];

    hash_wrapper.sample_biased_ternary_ring_element_vec_into(&mut fold_challenge);

    let folded_witness = fold(&witness, &fold_challenge);

    let mut next_round_data = new_vec_zero_preallocated(CONFIG.composed_witness_length);

    let folded_witness_decomposed = decompose(
        &folded_witness.data,
        CONFIG.witness_decomposition_radix_log as u64,
        CONFIG.witness_decomposition_chunks,
    );

    // TODO: can we avoid those copies?
    paste_by_prefix(
        &mut next_round_data,
        &folded_witness_decomposed,
        &CONFIG.folded_witness_prefix,
    );

    paste_recursive_commitment(
        &mut next_round_data,
        &rc_projection_image,
        &CONFIG.projection_recursion,
    );

    paste_recursive_commitment(&mut next_round_data, &rc_opening, &CONFIG.opening_recursion);

    paste_recursive_commitment(
        &mut next_round_data,
        &rc_commitment,
        &CONFIG.commitment_recursion,
    );

    // SUMCHECK
    // we want to check that
    // ck \cdot folded_witness - commitment \cdot fold_challenge = 0
    // outer_evaluation_points \cdot folded_witness - opening \cdot fold_challenge = 0
    // <opening, inner_evaluation_points> - evaluations = 0
    // I \otimes projection_matrix \cdot folded_witness - projection_image \cdot fold_challenge = 0
    // rc_projection_image, rc_opening, rc_commitment are well-formed
    // <w, conj(w)> + <y, conj(y)> - t = 0

    RoundOutput {
        folded_witness,
        projection_image,
        opening,
    }
}

pub fn execute() {
    let crs = CRS::gen_crs(CONFIG.witness_height * 2, 2);

    let witness = VerticallyAlignedMatrix {
        height: CONFIG.witness_height,
        width: CONFIG.witness_width,
        data: sample_random_short_vector(
            CONFIG.witness_height * CONFIG.witness_width,
            10,
            Representation::IncompleteNTT,
        ),
    };

    let ck = &crs.ck_for_wit_dim(witness.height);

    let basic_commitment = commit_basic(&crs, &witness, CONFIG.basic_commitment_rank);

    let rc_commitment =
        recursive_commit(&crs, &CONFIG.commitment_recursion, &basic_commitment.data);

    let evaluation_points_inner = vec![range(0, witness.height.ilog2() as usize)
        .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2))
        .collect::<Vec<RingElement>>()];

    let evaluation_points_outer = vec![range(0, witness.width.ilog2() as usize)
        .map(|_| RingElement::random_bounded(Representation::IncompleteNTT, 2))
        .collect::<Vec<RingElement>>()];

    let round_output = prover_round(
        &crs,
        &rc_commitment,
        &witness,
        &evaluation_points_inner,
        &evaluation_points_outer,
    );
}
