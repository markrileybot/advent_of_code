use anyhow::Result;
use cust::launch;
use cust::memory::CopyDestination;
use cust::prelude::SliceExt;

use crate::utils::Ctx;

static INPUT: &str = include_str!("../input/day1.txt");


pub(crate) fn day1_2() -> Result<()> {
    let words = vec!["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
    let mut digits = INPUT.lines()
        .map(|l| {
            // digits
            let mut c0 = l.chars()
                .enumerate()
                .filter(|c| c.1.is_digit(10))
                .map(|c| (c.0, c.1.to_digit(10).unwrap()))
                .next().unwrap_or((usize::max_value(), 0));
            let mut c1 = l.chars()
                .enumerate()
                .filter(|c| c.1.is_digit(10))
                .map(|c| (c.0, c.1.to_digit(10).unwrap()))
                .last().unwrap_or((0, 0));

            // words
            for (widx, word) in words.iter().enumerate() {
                if let Some(index) = l.find(word) {
                    if index < c0.0 {
                        c0.1 = widx as u32;
                        c0.0 = index;
                    }
                }
                if let Some(index) = l.rfind(word) {
                    if index > c1.0 {
                        c1.1 = widx as u32;
                        c1.0 = index;
                    }
                }
            }

            c0.1 * 10 + c1.1
        }).collect::<Vec<u32>>();
    day1(&mut digits)
}

pub(crate) fn day1_1() -> Result<()> {
    let mut digits = INPUT.lines()
        .map(|l| {
            let c0 = l.chars()
                .filter(|c| c.is_digit(10))
                .next().unwrap_or('0').to_digit(10).unwrap();
            let c1 = l.chars()
                .filter(|c| c.is_digit(10))
                .last().unwrap_or('0').to_digit(10).unwrap();
            c0 * 10 + c1
        }).collect::<Vec<u32>>();
    day1(&mut digits)
}

fn day1(mut digits: &mut Vec<u32>) -> Result<()> {
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