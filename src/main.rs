use cowboys_and_aliens::common::ring_arithmetic::*;
use cowboys_and_aliens::common::init_common;

fn main() {
    init_common();
    let mut a = RingElement::new_random(Representation::Coefficients);
    let mut b = RingElement::new_random(Representation::Coefficients);
    let mut c = RingElement::new(Representation::Coefficients);

    naive_polynomial_multiplication(&mut c, &a, &b);

    a.from_coefficients_to_even_odd_coefficients();
    b.from_coefficients_to_even_odd_coefficients();

    a.from_even_odd_coefficients_to_incomplete_ntt_representation();
    b.from_even_odd_coefficients_to_incomplete_ntt_representation();

    let mut d = RingElement::new(Representation::IncompleteNTT);

    let start_time = std::time::Instant::now();
    incomplete_ntt_multiplication(&mut d, &a, &b);
    let duration = start_time.elapsed();
    println!("NTT multiplication took: {:?}", duration);

    d.from_incomplete_ntt_to_even_odd_coefficients();
    d.from_even_odd_coefficients_to_coefficients();

    assert_eq!(c.v, d.v);
    println!("Result matches naive multiplication!");

    // b.from_incomplete_ntt_to_even_odd_coefficients();
    // b.from_even_odd_coefficients_to_coefficients();
    // println!("b back = {:?}", b.v);

}
