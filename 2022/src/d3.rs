use std::collections::HashSet;
use std::iter::FromIterator;
use std::io::BufRead;

fn priority(c: char) -> u8 {
    let mut pri = c as u8;
    if pri > 96 {
        pri -= 96;
    } else {
        pri -= 38;
    }
    return pri;
}

pub fn p1<T:BufRead>(inputs: T) {
    let mut total: u32 = 0;
    for x in inputs.lines().map(|f| f.unwrap()) {
        let (c0, c1) = x.split_at(x.len() / 2);
        let c0: HashSet<u8> = HashSet::from_iter(c0.chars().map(|c| priority(c)));
        let c1: HashSet<u8> = HashSet::from_iter(c1.chars().map(|c| priority(c)));
        for i in c0 {
            if c1.contains(&i) {
                total += i as u32;
            }
        }
    }
    println!("{}", total);
}

pub fn p2<T:BufRead>(inputs: T) {
    let mut total: u32 = 0;
    for group in inputs.lines().map(|f| f.unwrap()).collect::<Vec<String>>().chunks_exact(3) {
        let e0: HashSet<u8> = HashSet::from_iter(group[0].chars().map(|c| priority(c)));
        let e1: HashSet<u8> = HashSet::from_iter(group[1].chars().map(|c| priority(c)));
        let e2: HashSet<u8> = HashSet::from_iter(group[2].chars().map(|c| priority(c)));
        for i in e0 {
            if e1.contains(&i) && e2.contains(&i) {
                total += i as u32;
            }
        }
    }
    println!("{}", total);
}