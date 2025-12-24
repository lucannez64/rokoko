use crate::common::ring_arithmetic::{Representation, RingElement};

#[derive(Debug, Clone)]
pub struct StructuredRow {
    // Each layer corresponds to one dimension in the tensor product structure.
    // Each layer has two elements, corresponding to the two choices (0 or 1) at that dimension.
    // For example, for a 3-layer structured row, the tensor_layers might look like:
    // [
    //   [a0, a1],  // Layer 0
    //   [b0, b1],  // Layer 1
    //   [c0, c1],  // Layer 2
    // ]
    // Then, the entry at position 5 (binary 100) would be computed as:
    // a0 * b0 * c1.
    // Notably, the order of layers corresponds to the inverse order of bits in the index,
    // (i.e. with the first layer corresponding to the least significant bit.)
    pub tensor_layers: Vec<[RingElement; 2]>,
}

impl StructuredRow {
    pub fn at(&self, pos: usize) -> RingElement {
        let mut result = RingElement::one(Representation::IncompleteNTT);
        let mut index = pos;
        for layer in &self.tensor_layers {
            let bit = index & 1;
            index >>= 1;
            result *= &layer[bit];
        }
        result
    }
}

pub struct PreprocessedRow {
    pub structured_row: StructuredRow,
    pub preprocessed_row: Vec<RingElement>,
}

impl PreprocessedRow {
    pub fn from_structured_row(structured_row: StructuredRow) -> Self {
        let mut result = Vec::with_capacity(2usize.pow(structured_row.tensor_layers.len() as u32));
        result.push(RingElement::one(Representation::IncompleteNTT));

        for layer in &structured_row.tensor_layers {
            let mut new_entries: Vec<RingElement> = Vec::with_capacity(result.len());
            for r in &mut result {
                let r1 = &*r * &layer[1];
                new_entries.push(r1);
                *r *= &layer[0];
            }
            for e in new_entries {
                result.push(e);
            }
        }
        PreprocessedRow {
            structured_row,
            preprocessed_row: result,
        }
    }
}

#[test]
fn test_structured_row() {
    let tensor_layers = vec![
        [
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(2, Representation::IncompleteNTT),
        ],
        [
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(4, Representation::IncompleteNTT),
        ],
        [
            RingElement::constant(5, Representation::IncompleteNTT),
            RingElement::constant(6, Representation::IncompleteNTT),
        ],
    ];
    let structured_row = StructuredRow { tensor_layers };

    assert_eq!(
        structured_row.at(0),
        RingElement::constant(1 * 3 * 5, Representation::IncompleteNTT)
    );
    assert_eq!(
        structured_row.at(1),
        RingElement::constant(2 * 3 * 5, Representation::IncompleteNTT)
    );
    assert_eq!(
        structured_row.at(2),
        RingElement::constant(1 * 4 * 5, Representation::IncompleteNTT)
    );
    assert_eq!(
        structured_row.at(3),
        RingElement::constant(2 * 4 * 5, Representation::IncompleteNTT)
    );
    assert_eq!(
        structured_row.at(4),
        RingElement::constant(1 * 3 * 6, Representation::IncompleteNTT)
    );
    assert_eq!(
        structured_row.at(5),
        RingElement::constant(2 * 3 * 6, Representation::IncompleteNTT)
    );
    assert_eq!(
        structured_row.at(6),
        RingElement::constant(1 * 4 * 6, Representation::IncompleteNTT)
    );
    assert_eq!(
        structured_row.at(7),
        RingElement::constant(2 * 4 * 6, Representation::IncompleteNTT)
    );
}

#[test]
fn test_preprocessed_row() {
    let tensor_layers = vec![
        [
            RingElement::constant(1, Representation::IncompleteNTT),
            RingElement::constant(2, Representation::IncompleteNTT),
        ],
        [
            RingElement::constant(3, Representation::IncompleteNTT),
            RingElement::constant(4, Representation::IncompleteNTT),
        ],
        [
            RingElement::constant(5, Representation::IncompleteNTT),
            RingElement::constant(6, Representation::IncompleteNTT),
        ],
    ];
    let structured_row = StructuredRow { tensor_layers };

    let structured_row_unused = structured_row.clone();

    let preprocessed_row = PreprocessedRow::from_structured_row(structured_row_unused);

    for i in 0..preprocessed_row.preprocessed_row.len() {
        assert_eq!(preprocessed_row.preprocessed_row[i], structured_row.at(i));
    }
}
