use anyhow::Result;

use crate::day1::{day1_1, day1_2};
use crate::day2::{day2_1, day2_2};
use crate::utils::Ctx;

mod add;
mod day1;
mod utils;
mod day2;

#[macro_export]
macro_rules! run {
    ($( $x:expr ),*) => {
        let ctx = Ctx::new()?;
        $(
            print!("{} ", stringify!($x));
            $x(&ctx)?;
        )*
    };
}

fn main() -> Result<()> {
    run!(
        day1_1,
        day1_2,
        day2_1,
        day2_2
    );
    Ok(())
}