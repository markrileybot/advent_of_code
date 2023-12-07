use std::str::FromStr;
use anyhow::Result;
use cust::launch;
use cust::memory::CopyDestination;
use cust::prelude::SliceExt;

use aoc_2023_cudalib::day4::{Card, MY_LEN};

use crate::utils::Ctx;

static INPUT: &str = include_str!("../input/day4.txt");

pub fn day4(ctx: &Ctx) -> Result<(u32, u32)> {
    let cards = INPUT.lines()
        .map(|l| Card::from_str(l).unwrap())
        .collect::<Vec<Card>>();

    let len = cards.len() * MY_LEN;
    let stream = &ctx.stream;
    let (day4_kernel, block_size) = ctx.load_kernel("day4")?;
    let grid_size = (len as u32 + block_size - 1) / block_size;
    let mut results = vec![0u32; len];

    print!(
        "using {} blocks and {} threads per block for {} elts ",
        grid_size, block_size, len);

    let cards_buf = cards.as_slice().as_dbuf()?;
    let results_buf = results.as_slice().as_dbuf()?;

    unsafe {
        launch!(
            day4_kernel<<<grid_size, block_size, 0, stream>>>(
                cards_buf.as_device_ptr(), // games
                cards_buf.len(), // number of numbers
                results_buf.as_device_ptr() // result
            )
        )?;
    }

    stream.synchronize()?;

    results_buf.copy_to(&mut results)?;
    Ok((results[0], results[1]))
}