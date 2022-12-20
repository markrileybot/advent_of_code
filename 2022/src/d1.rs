use std::cmp::Ordering;
use std::io::BufRead;

fn parse<T:BufRead>(inputs: T) -> Vec<u32> {
    let mut elf_totals = Vec::new();
    for x in inputs.lines().map(|v| v.unwrap()) {
        if x.is_empty() || elf_totals.is_empty() {
            elf_totals.push(0);
        } else {
            let elf_idx = &elf_totals.len() - 1;
            elf_totals[elf_idx] += x.parse::<u32>().unwrap();
        }
    }
    elf_totals
}

pub fn p1<T:BufRead>(inputs: T) {
    let elf_totals = parse(inputs);
    match elf_totals.iter().max() {
        Some(max) => println!("Max value: {}", max),
        None => println!("Vector is empty"),
    }
}

pub fn p2<T:BufRead>(inputs: T) {
    let mut elf_totals = parse(inputs);
    elf_totals.sort_by(|a, b|
        if b > a { Ordering::Greater } else if a > b { Ordering::Less } else { Ordering::Equal });
    println!("{}", elf_totals[0] + elf_totals[1] + elf_totals[2]);
    for x in elf_totals {
        println!("{}", x);
    }
}