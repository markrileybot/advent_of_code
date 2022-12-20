use std::collections::HashSet;
use std::iter::FromIterator;
use std::io::BufRead;

fn p0<T:BufRead>(inputs: T, marker: usize) {
    for x in inputs.lines().map(|f| f.unwrap()) {
        for i in 0..x.len() {
            let magic: HashSet<char> = HashSet::from_iter(x[i..i+&marker].chars());
            if magic.len() == marker {
                println!("{}", i+marker);
                break;
            }
        }
    }
}

pub fn p1<T:BufRead>(inputs: T) {
    p0(inputs, 4);
}

pub fn p2<T:BufRead>(inputs: T) {
    p0(inputs, 14);
}