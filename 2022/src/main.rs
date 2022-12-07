use std::fs::File;
use std::io::{BufRead, BufReader};

use anyhow::{bail, Result};
use clap::{App, Arg};

mod d1;
mod d2;
mod d3;
mod d4;

trait DayRunnable {
    fn run(&self, file_name: &str, problem_id: usize) -> Result<()>;
}

trait InputParser<T> {
    fn parse(&self, line: String, inputs: &mut Vec<T>);
}

struct LineParser<T> {
    line_parser: fn(String) -> T,
}

impl <T> LineParser<T> {
    fn new(line_parser: fn(String) -> T) -> LineParser<T> {
        return LineParser {
            line_parser
        }
    }
}

impl <T> InputParser<T> for LineParser<T> {
    fn parse(&self, line: String, inputs: & mut Vec<T>) {
        inputs.push((self.line_parser)(line));
    }
}

struct Day<T> {
    input_parser: Box<dyn InputParser<T>>,
    problems: Vec<fn(&Vec<T>)>,
}

impl <T: 'static> Day<T> {
    fn new(problems: Vec<fn(&Vec<T>)>, input_parser: Box<dyn InputParser<T>>) -> Day<T> {
        return Day {
            problems,
            input_parser
        }
    }

    fn new_l(problems: Vec<fn(&Vec<T>)>, input_parser: fn(String) -> T) -> Day<T> {
        return Day::new(problems, Box::new(LineParser::new(input_parser)));
    }
}

impl <T> DayRunnable for Day<T> {
    fn run(&self, file_name: &str, problem_id: usize) -> Result<()> {
        let mut inputs: Vec<T> = Vec::new();
        let file = File::open(file_name)?;
        let lines = BufReader::new(file).lines();
        for line in lines {
            if let Ok(l) = line {
                self.input_parser.parse(l, inputs.as_mut());
            }
        }

        if problem_id < self.problems.len() {
            self.problems[problem_id](&inputs);
            return Ok(());
        }
        bail!("Invalid problem {}", problem_id);
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

    let problems: Vec<Box<dyn DayRunnable>> = vec![
        Box::new(Day::new_l(vec![d1::p1, d1::p2], |l| l)),
        Box::new(Day::new_l(vec![d2::p1, d2::p2], |l| l)),
        Box::new(Day::new_l(vec![d3::p1, d3::p2], |l| l)),
        Box::new(Day::new_l(vec![d4::p1, d4::p2], |l| l)),
    ];

    if day < problems.len() {
        return problems[day].run(input_file_name, problem);
    } else {
        bail!("Invalid day {}", day + 1);
    }
}
