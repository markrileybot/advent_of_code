#[macro_use]
extern crate anyhow;

use std::fs::File;
use std::io::{self, BufRead};
use std::iter::FromIterator;

use anyhow::Result;
use clap::{App, Arg};
use colored::Colorize;

trait DayRunnable {
    fn run(&self, file_name: &str, problem_id: usize) -> Result<()>;
}

trait InputParser<T> {
    fn parse(&self, line: &str, inputs: &mut Vec<T>);
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
    fn parse(&self, line: &str, inputs: & mut Vec<T>) {
        inputs.push((self.input_parser)(line));
    }
}

struct Day<T> {
    input_parser: Box<dyn InputParser<T>>,
    problems: Vec<fn(&Vec<T>)>,
}

impl <T> Day<T> {
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
        let lines = io::BufReader::new(file).lines();
        for line in lines {
            if let Ok(l) = line {
                self.input_parser.parse(&l, inputs.as_mut());
            }
        }

        if problem_id > 0 && problem_id < self.problems.len() {
            self.problems[problem_id](&inputs);
            return Ok(());
        }
        bail!("Invalid problem {}", problem_id);
    }
}

fn main() -> Result<()> {

    let args = App::new("Advent of Code 2020 :: Day 1")
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
        Box::new(Day::new_l(vec![d1_1, d1_2], |l| l.parse::<i32>().unwrap())),
        Box::new(Day::new_l(vec![d2_1, d2_2], |l| l)),
        Box::new(Day::new_l(vec![d3_1, d3_2], |l| l.chars().collect()))
    ];

    if day < problems.len() {
        return problems[day].run(input_file_name, problem);
    } else {
        bail!("Invalid day {}", day);
    }
}

fn d2_1(inputs: &Vec<String>) {
    let mut bad_count = 0;
    let mut good_count = 0;
    for input in inputs {
        let parts: Vec<_> = input.split(|c| c == ' ' || c == '-' || c == ':').collect();
        let min = parts[0].parse::<i32>().unwrap();
        let max = parts[1].parse::<i32>().unwrap();
        let chr = parts[2].chars().nth(0).unwrap();
        let pwd = parts[4];
        let mut char_cnt = 0;
        for c in pwd.chars() {
            if c == chr {
                char_cnt += 1;
            }
        }
        if char_cnt < min || char_cnt > max {
            bad_count += 1;
            println!("{}\t{}\t{}\t{}\t{}", min, max, chr, char_cnt, pwd);
        } else {
            good_count += 1;
        }
    }
    println!("{} {} {}", bad_count, good_count, inputs.len());
}


fn d2_2(inputs: &Vec<String>) {
    let mut bad_count = 0;
    let mut good_count = 0;
    for input in inputs {
        let parts: Vec<_> = input.split(|c| c == ' ' || c == '-' || c == ':').collect();
        let pos0 = parts[0].parse::<usize>().unwrap() - 1;
        let pos1 = parts[1].parse::<usize>().unwrap() - 1;
        let chr = parts[2].chars().nth(0).unwrap();
        let pwd = parts[4];
        let mut char_cnt = 0;
        if pos0 < pwd.len() && pwd.chars().nth(pos0).unwrap() == chr {
            char_cnt += 1;
        }
        if pos1 < pwd.len() && pwd.chars().nth(pos1).unwrap() == chr {
            char_cnt += 1;
        }
        if char_cnt != 1 {
            bad_count += 1;
            println!("{}\t{}\t{}\t{}\t{}", pos0 + 1, pos1 + 1, chr, char_cnt, pwd);
        } else {
            good_count += 1;
        }
    }
    println!("{} {} {}", bad_count, good_count, inputs.len());
}

fn d3_2(inputs: &Vec<Vec<char>>) {
    let tc0: i64 = d3(inputs, 1, 1) as i64;
    let tc1: i64 = d3(inputs, 3, 1) as i64;
    let tc2: i64 = d3(inputs, 5, 1) as i64;
    let tc3: i64 = d3(inputs, 7, 1) as i64;
    let tc4: i64 = d3(inputs, 1, 2) as i64;
    let tc5: i64 = tc0 * tc1 * tc2 * tc3 * tc4;
    println!("{} x {} x {} x {} x {} = {}", tc0, tc1, tc2, tc3, tc4, tc5);
}

fn d3_1(inputs: &Vec<Vec<char>>) {
    d3(inputs, 3, 1);
}

fn d3(inputs: &Vec<Vec<char>>, x_mov: usize, y_mov: usize) -> i32 {
    println!("\t======================================================= {} x {}\t", x_mov, y_mov);
    let mut x_off: usize = 0;
    let mut y_off: usize = 0;
    let mut tree_count = 0;
    let width = (inputs.len() * x_mov) as f32;

    for (y, row) in inputs.iter().enumerate() {
        let row_segment_width = row.len();
        let row_segments = (width / row_segment_width as f32).ceil() as usize;
        let mut curr_row_segment: usize = 0;
        let x_segment = x_off / row_segment_width;

        let mut row_tree_count = 0;
        let mut x = 0;
        let mut hit = false;

        while curr_row_segment < row_segments {
            if curr_row_segment == x_segment && y == y_off {
                hit = true;
                x = x_off - x_segment * row_segment_width;
                if x > 0 {
                    print!("{}", String::from_iter(&row[0..x]));
                }
                if row[x] == '.' {
                    print!("{}", "O".red().bold());
                } else {
                    row_tree_count += 1;
                    print!("{}", "X".red().bold());
                }
                if x < row_segment_width {
                    print!("{}", String::from_iter(&row[x + 1..row_segment_width]));
                }
            } else {
                print!("{}", String::from_iter(row));
            }
            curr_row_segment += 1;
        }

        tree_count += row_tree_count;
        println!("\t{} {} {}", row_tree_count, tree_count, x);

        if hit {
            x_off += x_mov;
            y_off += y_mov;
        }
    }

    println!("{}", tree_count);
    return tree_count;
}

fn d1_1(inputs: &Vec<i32>) {
    let expect = inputs[0];
    let inputs_sl = &inputs[1..inputs.len()];
    for x in inputs_sl {
        for y in inputs_sl {
            if x + y == expect {
                println!("{} x {} = {}", x, y, x * y);
            }
        }
    }
}

fn d1_2(inputs: &Vec<i32>) {
    let expect = inputs[0];
    let inputs_sl = &inputs[1..inputs.len()];
    for x in inputs_sl {
        for y in inputs_sl {
            for z in inputs_sl {
                if x + y + z == expect {
                    println!("{} x {} x {} = {}", x, y, z, x * y * z);
                }
            }
        }
    }
}
