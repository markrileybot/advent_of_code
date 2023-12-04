use std::str::FromStr;

use anyhow::Result;
use cust::launch;
use cust::memory::{CopyDestination, UnifiedBuffer};
use cust::prelude::SliceExt;

use aoc_2023_cudalib::day2::{Game, Rgb};

use crate::utils::Ctx;

static INPUT: &str = include_str!("../input/day2.txt");

pub fn day2_1(ctx: &Ctx) -> Result<()> {
    let (count, _power) = day2(ctx)?;
    println!("{}", count);
    Ok(())
}

pub fn day2_2(ctx: &Ctx) -> Result<()> {
    let (_count, power) = day2(ctx)?;
    println!("{}", power);
    Ok(())
}

fn day2(ctx: &Ctx) -> Result<(u32, u32)> {
    let games_vec = INPUT.lines()
        .map(|l| {
            l.split_once(":").unwrap().1.trim()
                .split(";")
                .map(|s| Rgb::from_str(s).unwrap())
                .collect::<Vec<Rgb>>()

        }).collect::<Vec<Vec<Rgb>>>();

    let buffers = games_vec.iter()
        .map(|s| s.as_slice().as_unified_buf().unwrap())
        .collect::<Vec<UnifiedBuffer<Rgb>>>();
    let games = buffers.iter()
        .map(|s| Game {sets: s.as_ref()})
        .collect::<Vec<Game>>();

    let stream = &ctx.stream;
    let (day2_kernel, block_size) = ctx.load_kernel("day2")?;
    let grid_size = (games.len() as u32 + block_size - 1) / block_size;
    let mut results = vec![0u32; games.len() * 2];

    print!(
        "using {} blocks and {} threads per block for {} games ",
        grid_size, block_size, games.len());

    let games_buf = games.as_slice().as_dbuf()?;
    let results_buf = results.as_slice().as_dbuf()?;

    unsafe {
        launch!(
            day2_kernel<<<grid_size, block_size, 0, stream>>>(
                games_buf.as_device_ptr(), // games
                games_buf.len(), // number of numbers
                Rgb::from((12u32, 13u32, 14u32)),
                results_buf.as_device_ptr() // result
            )
        )?;
    }

    stream.synchronize()?;

    results_buf.copy_to(&mut results)?;

    Ok((results[0], results[games.len()]))
}