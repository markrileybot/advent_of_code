use std::cell::{RefCell, RefMut};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::iter::FromIterator;
use std::rc::Rc;
use std::slice::Iter;

use anyhow::{bail, Result};
use clap::{App, Arg};
use colored::Colorize;

type Link = Rc<RefCell<Node>>;

struct Node {
    value: char,
    out_edges: Vec<Rc<RefCell<Node>>>,
    in_edges: Vec<Rc<RefCell<Node>>>,
}

impl Node {
    fn connect_out(&mut self, node: Link) {
        self.out_edges.push(node);
    }

    fn connect_in(&mut self, node: Link) {
        self.in_edges.push(node);
    }

    fn connect(node_from: Link, node_to: Link) {
        node_from.borrow_mut().connect_in(node_to.clone());
        node_to.borrow_mut().connect_out(node_from.clone());
    }
}

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
        Box::new(Day::new_l(vec![d1_1, d1_2], |l| l)),
    ];

    if day < problems.len() {
        return problems[day].run(input_file_name, problem);
    } else {
        bail!("Invalid day {}", day);
    }
}

fn d1_1(inputs: &Vec<String>) {
    let elf_totals = parse_d1(inputs);
    match elf_totals.iter().max() {
        Some(max) => println!( "Max value: {}", max ),
        None      => println!( "Vector is empty" ),
    }
}

fn parse_d1(inputs: &Vec<String>) -> Vec<u32> {
    let mut elf_totals = Vec::new();
    for x in inputs {
        if x.is_empty() || elf_totals.is_empty() {
            elf_totals.push(0);
        } else {
            let elf_idx = &elf_totals.len() - 1;
            elf_totals[elf_idx] += x.parse::<u32>().unwrap();
        }
    }
    elf_totals
}

fn d1_2(inputs: &Vec<String>) {
    let mut elf_totals = parse_d1(inputs);
    elf_totals.sort_by(|a,b|
        if b > a { Ordering::Greater } else if a > b { Ordering::Less } else { Ordering::Equal });
    println!("{}", elf_totals[0] + elf_totals[1] + elf_totals[2]);
    for x in elf_totals {
        println!("{}", x);
    }
}