use crate::{common::{hash::HashWrapper, matrix::VerticallyAlignedMatrix, projection_matrix::ProjectionMatrix, ring_arithmetic::{Representation, RingElement}, sampling::sample_random_short_vector, structured_row::StructuredRow}, protocol::{commitment::{self, BasicCommitment, commit_basic}, crs::CRS, params::{decompose_witness, witness_sampler}, project::{prepare_i16_witness, project}}};


const WITNESS_DIM: usize = 2usize.pow(16);
const WITNESS_WIDTH: usize = 2usize; 
const RANK: usize = 8;

pub struct SalsaaProof {

}

pub fn prover_round(
    crs: &CRS,
    commitmens: &BasicCommitment,
    witness: &VerticallyAlignedMatrix<RingElement>,
    evaluation_points_inner: &Vec<StructuredRow>,
    exact_binariness: bool, // whether the proof should be for exact binariness. If not l2 norm of the witness is given by the proof
    hash_wrapper: &mut HashWrapper,
) -> SalsaaProof {

    let witness_16 = prepare_i16_witness(witness);

    let mut projection_matrix =
        ProjectionMatrix::new(witness.width, 256);

    projection_matrix.sample(hash_wrapper);

    project(&witness_16, &projection_matrix);

    panic!("Not implemented yet");
}



pub fn binary_witness_sampler() -> VerticallyAlignedMatrix<RingElement> {
    VerticallyAlignedMatrix {
        height: WITNESS_DIM,
        width: WITNESS_WIDTH,
        data: sample_random_short_vector(
            WITNESS_DIM * WITNESS_WIDTH,
            2,
            Representation::IncompleteNTT,
        ),
        used_cols: WITNESS_WIDTH,
    }
}

pub fn execute() {
    println!("Generating CRS...");

    let crs = CRS::gen_crs(
        WITNESS_DIM,
        8,
    );


    let witness = binary_witness_sampler();

    println!("===== COMMITTING WITNESS =====");
    let start = std::time::Instant::now();

    let commitment = commit_basic(
        &crs,
        &witness,
        RANK,
    );

    let commit_duration = start.elapsed().as_nanos();
    println!("TOTAL Commit time: {:?} ns", commit_duration);

    




    

}