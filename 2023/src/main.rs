use std::io::Write;
use anyhow::Result;

use crate::day1::day1;
use crate::day2::day2;
use crate::day3::day3;
use crate::day4::day4;
use crate::day5::day5;
use crate::utils::Ctx;

mod add;
mod day1;
mod utils;
mod day2;
mod day3;
mod day4;
mod day5;

#[macro_export]
macro_rules! run {
    ($( $x:ident ),*) => {
        let ctx = Ctx::new()?;
        $(
            print!("{} ", stringify!($x));
            std::io::stdout().flush()?;
            let res: (u32, u32) = $x(&ctx).expect(format!("{} failed!", stringify!($x)).as_str());
            println!(" p1={} p2={}", res.0, res.1);
        )*
    };
}

fn main() -> Result<()> {
    run!(
        day1,
        day2,
        day3,
        day4,
        day5
    );
    Ok(())
}