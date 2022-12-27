use std::fs::File;
use std::io::{BufRead, BufReader};

use anyhow::{bail, Result};
use clap::{App, Arg};

mod d1;
mod d2;
mod d3;
mod d4;
mod d5;
mod d6;
mod d7;
mod d8;
mod d9;
mod d10;
mod d11;
mod d12;
mod d13;
mod d14;
mod d15;

trait Problem<T: BufRead> {
    fn solve(&self, input: T);
}

struct FuncyProblem<T: BufRead> {
    func: fn(T)
}
impl <T:BufRead> FuncyProblem<T> {
    fn new(func: fn(T)) -> Self {
        Self {
            func
        }
    }
}
impl <T:BufRead> Problem<T> for FuncyProblem<T> {
    fn solve(&self, input: T) {
        (self.func)(input)
    }
}

fn main() -> Result<()> {

    let args = App::new("Advent of Code 2021")
        .version("1.0")
        .arg(Arg::with_name("input")
            .required(true)
            .short("i")
            .long("input")
            .value_name("FILE")
            .help("Input file")
            .takes_value(true))
        .arg(Arg::with_name("day")
            .required(true)
            .short("d")
            .long("day")
            .value_name("DAY")
            .help("Day")
            .takes_value(true))
        .arg(Arg::with_name("problem")
            .required(true)
            .short("p")
            .long("problem")
            .value_name("PROBLEM")
            .help("Problem")
            .takes_value(true))
        .get_matches();

    let day = args.value_of("day").unwrap().parse::<usize>().unwrap() - 1;
    let problem = args.value_of("problem").unwrap().parse::<usize>().unwrap() - 1;
    let input_file_name = args.value_of("input").unwrap();

    let problems: Vec<Vec<FuncyProblem<BufReader<File>>>> = vec![
        vec![FuncyProblem::new(d1::p1), FuncyProblem::new(d1::p2)],
        vec![FuncyProblem::new(d2::p1), FuncyProblem::new(d2::p2)],
        vec![FuncyProblem::new(d3::p1), FuncyProblem::new(d3::p2)],
        vec![FuncyProblem::new(d4::p1), FuncyProblem::new(d4::p2)],
        vec![FuncyProblem::new(d5::p1), FuncyProblem::new(d5::p2)],
        vec![FuncyProblem::new(d6::p1), FuncyProblem::new(d6::p2)],
        vec![FuncyProblem::new(d7::p1), FuncyProblem::new(d7::p2)],
        vec![FuncyProblem::new(d8::p1), FuncyProblem::new(d8::p2)],
        vec![FuncyProblem::new(d9::p1), FuncyProblem::new(d9::p2)],
        vec![FuncyProblem::new(d10::p1), FuncyProblem::new(d10::p2)],
        vec![FuncyProblem::new(d11::p1), FuncyProblem::new(d11::p2)],
        vec![FuncyProblem::new(d12::p1), FuncyProblem::new(d12::p2)],
        vec![FuncyProblem::new(d13::p1), FuncyProblem::new(d13::p2)],
        vec![FuncyProblem::new(d14::p1), FuncyProblem::new(d14::p2)],
        vec![FuncyProblem::new(d15::p1), FuncyProblem::new(d15::p2)],
    ];

    if let Some(day) = problems.get(day) {
        if let Some(problem) = day.get(problem) {
            problem.solve(BufReader::new(File::open(input_file_name)?));
            return Ok(())
        } else {
            bail!("Invalid problem {}", problem + 1);
        }
    } else {
        bail!("Invalid day {}", day + 1);
    }
}
