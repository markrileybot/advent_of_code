use std::mem::swap;
use anyhow::Result;
use cust::launch;
use cust::memory::CopyDestination;
use cust::prelude::SliceExt;
use crate::utils::{Ctx,};

static INPUT: &str = include_str!("../input/day1.txt");

pub(crate) fn day1() -> Result<()> {
    let mut digits = INPUT.lines()
        .map(|l| {
            let c0 = l.chars()
                .filter(|c| c.is_digit(10))
                .next().unwrap().to_digit(10).unwrap();
            let c1 = l.chars()
                .filter(|c| c.is_digit(10))
                .last().unwrap().to_digit(10).unwrap();
            c0 * 10 + c1
        }).collect::<Vec<u32>>();

    let ctx = Ctx::new()?;
    let stream = &ctx.stream;
    let (day1_kernel, block_size) = ctx.load_kernel("day1")?;
    let grid_size = ((digits.len() / 10) as u32 + block_size - 1) / block_size;
    let expected = digits.iter().fold(0, |d0, d1| d0 + d1);

    println!(
        "using {} blocks and {} threads per block for {} digits expected {}",
        grid_size, block_size, digits.len(), expected
    );

    let mut digits_buf = digits.as_slice().as_dbuf()?;

    unsafe {
        launch!(
            // slices are passed as two parameters, the pointer and the length.
            day1_kernel<<<grid_size, block_size, 0, stream>>>(
                10usize, // chunk size
                digits_buf.len(), // number of numbers
                digits_buf.as_device_ptr() // numbers
            )
        )?;
    }

    stream.synchronize()?;

    digits_buf.copy_to(&mut digits)?;

    println!("{:?}", digits[0]);
    Ok(())
}