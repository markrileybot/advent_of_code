use std::cell::RefCell;
use std::rc::Rc;
use std::str::FromStr;

use crate::d10::Op::{ADDX, NOOP};

#[derive(Debug, Clone)]
enum Op {
    ADDX(u8, i32),
    NOOP(u8)
}

impl Op {
    fn new_addx(value: i32) -> Self {
        ADDX(2u8, value)
    }
    fn new_noop() -> Self {
        NOOP(1u8)
    }
    fn tick(&mut self, cpu: &mut CPU) -> bool {
        match self {
            ADDX(c, v) => {
                *c -= &1u8;
                if *c <= 0 {
                    cpu.register += *v;
                    true
                } else {
                    false
                }
            },
            NOOP(c) => {
                *c -= &1u8;
                *c <= 0
            }
        }
    }
}

impl FromStr for Op {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.split_once(' ') {
            None => Op::new_noop(),
            Some((_, v)) => Op::new_addx(v.parse().unwrap())
        })
    }
}

struct CPU {
    program: Vec<Rc<RefCell<Op>>>,
    counter: i32,
    op_counter: usize,
    register: i32,
}

impl CPU {
    fn new(program: Vec<Op>) -> Self {
        return Self {
            program: program.iter().map(|o| Rc::new(RefCell::new(o.clone()))).collect(),
            counter: 1,
            op_counter: 0,
            register: 1
        }
    }

    fn tick(&mut self) -> bool {
        let op = self.program.first();
        let next = match op {
            None => 0,
            Some(op) => {
                if op.clone().borrow_mut().tick(self) {
                    1
                } else {
                    2
                }
            }
        };
        if next == 0 {
            return false;
        } else if next == 1 {
            self.op_counter += &1;
            self.program.remove(0);
            if self.program.is_empty() {
                return false;
            }
        }
        self.counter += &1;
        return true;
    }

    fn signal_strength(&self) -> i32 {
        return self.counter * self.register;
    }
}



fn parse(inputs: &Vec<String>) -> CPU {
    let program = inputs.iter()
        .map(|s| s.parse::<Op>().unwrap())
        .collect::<Vec<Op>>();
    CPU::new(program)
}

pub fn p1(inputs: &Vec<String>) {
    let mut cpu = parse(inputs);
    let mut total_ss = 0;
    while cpu.tick() {
        if cpu.counter == 20 || (cpu.counter - 20) % 40 == 0 {
            let ss = cpu.signal_strength();
            total_ss += ss;
            println!("{: >8}{: >8}{: >8}{: >8}{: >8}", cpu.counter, cpu.register, cpu.op_counter, ss, total_ss);
        }
    }
}

pub fn p2(inputs: &Vec<String>) {
    let mut cpu = parse(inputs);
    let mut pixel_pos = 0;
    loop {
        print!("{}", if cpu.register >= (pixel_pos - 1) && cpu.register <= (pixel_pos + 1) {"#"} else {"."});
        pixel_pos += 1;
        if pixel_pos % 40 == 0 {
            pixel_pos = 0;
            println!();
        }
        if !cpu.tick() {
            break;
        }
    }
}