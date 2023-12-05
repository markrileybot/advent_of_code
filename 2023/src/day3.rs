use anyhow::Result;
use cust::launch;
use cust::memory::CopyDestination;
use cust::prelude::SliceExt;

use aoc_2023_cudalib::day3::Element;

use crate::utils::Ctx;

static INPUT: &str = include_str!("../input/day3.txt");

pub fn day3(ctx: &Ctx) -> Result<(u32, u32)> {
    let mut grid_size = (0usize, 0usize);
    let elements = INPUT.lines().enumerate()
        .flat_map(|(y, l) | {
            let l2 = format!("{}.", l); // this makes are cuda code simpler
            l2.chars().enumerate().map(move |(x, c) | {
                Element {x, y, value: c}
            }).collect::<Vec<Element>>()
        })
        .collect::<Vec<Element>>();

    for x in &elements {
        grid_size.0 = grid_size.0.max(x.x);
        grid_size.1 = grid_size.1.max(x.y);
    }
    grid_size.0 += 1;
    grid_size.1 += 1;

    let stream = &ctx.stream;
    let (day3_kernel, block_size) = ctx.load_kernel("day3")?;
    let gs = (elements.len() as u32 + block_size - 1) / block_size;
    let mut results = vec![-1i32; elements.len() * 2];

    print!(
        "using {} blocks and {} threads per block for {} elts ",
        grid_size.0, grid_size.1, block_size);

    let elements_buf = elements.as_slice().as_dbuf()?;
    let results_buf = results.as_slice().as_dbuf()?;

    unsafe {
        launch!(
            day3_kernel<<<gs, block_size, 0, stream>>>(
                elements_buf.as_device_ptr(), // games
                elements_buf.len(), // number of numbers
                grid_size.0, grid_size.1,
                results_buf.as_device_ptr() // result
            )
        )?;
    }

    stream.synchronize()?;

    results_buf.copy_to(&mut results)?;
    Ok((results[0] as u32, results[1] as u32))
}