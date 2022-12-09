use std::collections::HashSet;
use std::iter::FromIterator;

fn p0(inputs: &Vec<String>, marker: usize) {
    for x in inputs {
        for i in 0..x.len() {
            let magic: HashSet<char> = HashSet::from_iter(x[i..i+&marker].chars());
            if magic.len() == marker {
                println!("{}", i+marker);
                break;
            }
        }
    }
}

pub fn p1(inputs: &Vec<String>) {
    p0(inputs, 4);
}

pub fn p2(inputs: &Vec<String>) {
    p0(inputs, 14);
}