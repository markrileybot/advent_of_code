use crate::add::add;
use crate::day1::{day1_1, day1_2};

mod add;
mod day1;
mod utils;

fn main() {
    // add().expect("oof");
    day1_1().expect("ooof");
    day1_2().expect("ooof");
}