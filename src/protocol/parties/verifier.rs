use crate::{
    common::{hash::HashWrapper, ring_arithmetic::RingElement, structured_row::StructuredRow},
    protocol::{
        config::{Config, NextRoundCommitment, RoundProof, SumcheckConfig, SumcheckRoundProof},
        crs::CRS,
        open::{evaluation_point_to_structured_row, evaluation_point_to_structured_row_conjugate},
        sumchecks::{
            context_verifier::VerifierSumcheckContext, runner_verifier::sumcheck_verifier,
        },
    },
};

pub fn verifier_round(
    crs: &CRS,
    config: &SumcheckConfig,
    rc_commitment: &Vec<RingElement>,
    round_proof: &SumcheckRoundProof,
    evaluation_points_inner: &Vec<StructuredRow>,
    evaluation_points_outer: &Vec<StructuredRow>,
    claims: &Vec<RingElement>,
    sumcheck_context_verifier: &mut VerifierSumcheckContext,
) {
    let start = std::time::Instant::now();
    let mut hash_wrapper_verifier = HashWrapper::new();

    let evaluation_points = sumcheck_verifier(
        &config,
        sumcheck_context_verifier,
        &rc_commitment,
        &round_proof,
        &evaluation_points_inner,
        &evaluation_points_outer,
        &claims,
        &mut hash_wrapper_verifier,
    );
    let elapsed = start.elapsed().as_nanos();
    println!("Verifier: {} ns", elapsed);
    match &round_proof.next {
        Some(next_round_proof) => {
            let next_round_commitment =
                round_proof
                    .next_round_commitment
                    .as_ref()
                    .unwrap_or_else(|| {
                        panic!(
                        "Next round commitment must be present when next round proof is present."
                    )
                    });
            match next_round_proof.as_ref() {
                RoundProof::Sumcheck(next_sumcheck_round_proof) => {
                    let next_sumcheck_config = match &config.next {
                        Some(next_config) => match next_config.as_ref() {
                            Config::Sumcheck(next_sumcheck_config) => next_sumcheck_config,
                            _ => panic!("Expected sumcheck config for next round."),
                        },
                        None => panic!("Next sumcheck config must be present."),
                    };

                    let (new_evaluation_points_outer, new_evaluation_points_inner) =
                        evaluation_points
                            .split_at(next_sumcheck_config.witness_width.ilog2() as usize);

                    let next_round_commiments_recursive = match &next_round_commitment {
                        NextRoundCommitment::Recursive(rc) => rc,
                        _ => panic!("Expected recursive commitment for next round."),
                    };

                    verifier_round(
                        crs,
                        &next_sumcheck_config,
                        &next_round_commiments_recursive,
                        next_sumcheck_round_proof,
                        &vec![
                            evaluation_point_to_structured_row(
                                &new_evaluation_points_inner.to_vec(),
                            ),
                            evaluation_point_to_structured_row_conjugate(
                                &new_evaluation_points_inner.to_vec(),
                            ),
                        ],
                        &vec![
                            evaluation_point_to_structured_row(
                                &new_evaluation_points_outer.to_vec(),
                            ),
                            evaluation_point_to_structured_row_conjugate(
                                &new_evaluation_points_outer.to_vec(),
                            ),
                        ],
                        &vec![
                            round_proof.claim_over_witness.clone(),
                            round_proof.claim_over_witness_conjugate.conjugate(),
                        ],
                        sumcheck_context_verifier.next.as_mut().unwrap(),
                    );
                }
                RoundProof::Simple(_) => {
                    // TODO: implement simple next round verifier
                }
            }
        }
        None => {}
    }
}
