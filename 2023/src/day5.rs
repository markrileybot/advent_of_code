use std::time::SystemTime;
use anyhow::Result;
use cust::launch;
use cust::memory::UnifiedBuffer;
use cust::memory::GpuBuffer;
use cust::prelude::{CopyDestination, DeviceCopyExt, SliceExt};
use aoc_2023_cudalib::day5::{Id, Mapping, Mappings, Type};

use crate::utils::Ctx;

static INPUT: &str = include_str!("../input/day5.txt");

pub fn day5(ctx: &Ctx) -> Result<(u32, u32)> {
    let (seeds, mappings) = parse_input();

    // p1 don't need no gpu
    let p1 = seeds.iter()
        .map(|s| mappings.map(*s, Type::Seed))
        .min()
        .unwrap();

    // p2 do
    let num_seeds = seeds.chunks(2)
        .flat_map(|s| (s[0]..(s[0] + s[1])).into_iter())
        .into_iter()
        .count();
    let mut counter = 0;
    let mut seeds_chunk = UnifiedBuffer::new(&0u32, 1000000).unwrap();
    let mut result = [0u32; 1];
    let mut p2 = Vec::with_capacity((num_seeds / seeds_chunk.len()) + 1);

    let stream = &ctx.stream;
    let (day5_kernel, block_size) = ctx.load_kernel("day5")?;
    let grid_size = (seeds_chunk.len() as u32 + block_size - 1) / block_size;
    let device_mappings = mappings.as_dbox().unwrap();

    let mut launch_count = 0;
    for id in seeds.chunks(2)
        .flat_map(|s| (s[0]..(s[0] + s[1])).into_iter())
        .into_iter() {
        seeds_chunk[counter] = id as u32;
        counter += 1;

        if counter == seeds_chunk.len() {
            result[0] = u32::max_value();
            let result_buf = result.as_slice().as_dbuf().unwrap();
            let time = SystemTime::now();
            unsafe {
                launch!(
                    day5_kernel<<<grid_size, block_size, 0, stream>>>(
                        device_mappings.as_device_ptr(), // games
                        seeds_chunk.as_device_ptr(), // result
                        seeds_chunk.len(), // number of numbers
                        result_buf.as_device_ptr()
                    )
                )?;
            }
            launch_count += 1;
            counter = 0;
            stream.synchronize()?;
            result_buf.copy_to(&mut result)?;
            p2.push(result[0]);
        }
    }

    Ok((p1 as u32, *p2.iter().min().unwrap()))
}

fn parse_input() -> (Vec<Id>, Mappings) {
    let mut seeds = Vec::new();
    let mut mappings = Mappings::new();
    let mut current_mapping_source = Type::Seed;
    let mut current_mapping_count = 0;
    for x in INPUT.lines() {
        if x.starts_with("seeds:") {
            for x in x.split(" ") {
                if let Ok(num) = x.trim().parse::<Id>() {
                    seeds.push(num);
                }
            }
        } else if x.starts_with("seed-to-soil") {
            current_mapping_source = Type::Seed;
            current_mapping_count = 0;
        } else if x.starts_with("soil-to-fertilizer") {
            current_mapping_source = Type::Soil;
            current_mapping_count = 0;
        } else if x.starts_with("fertilizer-to-water") {
            current_mapping_source = Type::Fertilizer;
            current_mapping_count = 0;
        } else if x.starts_with("water-to-light") {
            current_mapping_source = Type::Water;
            current_mapping_count = 0;
        } else if x.starts_with("light-to-temperature") {
            current_mapping_source = Type::Light;
            current_mapping_count = 0;
        } else if x.starts_with("temperature-to-humidity") {
            current_mapping_source = Type::Temp;
            current_mapping_count = 0;
        } else if x.starts_with("humidity-to-location") {
            current_mapping_source = Type::Humidity;
            current_mapping_count = 0;
        } else if !x.trim().is_empty() {
            let mut map = x.split(" ")
                .map(|s| s.trim().parse::<Id>().unwrap())
                .collect::<Mapping>();
            map.source_type = current_mapping_source;
            map.dest_type = map.source_type.dest().unwrap();
            mappings.mappings[map.source_type.ord()][current_mapping_count] = map;
            current_mapping_count += 1;
        }
    }
    (seeds, mappings)
}