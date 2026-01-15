const PROJECTION_BASE_HEIGHT: usize = 8;
const SIZE: usize = PROJECTION_BASE_HEIGHT * PROJECTION_BASE_HEIGHT / 8;

fn main() {
    println!("PROJECTION_BASE_HEIGHT = {}", PROJECTION_BASE_HEIGHT);
    println!("SIZE = {}", SIZE);
    let arr: [u8; SIZE] = [0; SIZE];
    println!("Array length: {}", arr.len());
}
