use std::cell::RefCell;
use std::collections::VecDeque;
use std::io::BufRead;
use std::rc::Rc;
use std::str::FromStr;

#[derive(Debug, Clone)]
enum Op {
    ADD(Option<u128>),
    SUB(Option<u128>),
    MLT(Option<u128>),
    DIV(Option<u128>),
}

impl Op {
    fn exec(&self, x: u128) -> u128 {
        match self {
            Op::ADD(o) => o.unwrap_or(x) + x,
            Op::SUB(o) => o.unwrap_or(x) - x,
            Op::MLT(o) => o.unwrap_or(x) * x,
            Op::DIV(o) => o.unwrap_or(x) / x
        }
    }
}

impl FromStr for Op {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.split_once(' ') {
            None => Err("Invalid op".to_string()),
            Some((o, v)) => {
                let i = match v.parse::<u128>() {
                    Ok(i) => Some(i),
                    Err(_) => None
                };
                match o {
                    "+" => Ok(Op::ADD(i)),
                    "-" => Ok(Op::SUB(i)),
                    "*" => Ok(Op::MLT(i)),
                    "/" => Ok(Op::DIV(i)),
                    _ => Err("Invalid op".to_string())
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Monkey {
    items: VecDeque<u128>,
    inspections: u128,
    operation: Op,
    divisor: u128,
    true_monkey: usize,
    false_monkey: usize,
}

impl Monkey {
    fn toss(&mut self, worry_dampener: u128, worry_max: u128, others: &Vec<Rc<RefCell<Monkey>>>) {
        while !self.items.is_empty() {
            if let Some(item) = self.items.pop_front() {
                self.inspections += 1;
                let mut new_worry_level = self.operation.exec(item);
                new_worry_level = new_worry_level / worry_dampener;
                if worry_max > 0 {
                    new_worry_level -= worry_max * (new_worry_level / worry_max);
                }
                let next = if new_worry_level % self.divisor == 0 { self.true_monkey } else { self.false_monkey };
                others.get(next).unwrap().borrow_mut().items.push_back(new_worry_level);
            }
        }
    }
}

impl FromStr for Monkey {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.split("\n");
        parts.next(); // skip monkey line
        let items = parts.next().unwrap().trim()["Starting items: ".len()..]
            .split(",").map(|s| s.trim()).map(|s| s.parse::<u128>().unwrap()).collect::<VecDeque<u128>>();
        let op = parts.next().unwrap().trim()["Operation: new = old ".len()..].parse::<Op>().unwrap();
        let div = parts.next().unwrap().trim()["Test: divisible by ".len()..].parse::<u128>().unwrap();
        let t = parts.next().unwrap().trim()["If true: throw to monkey ".len()..].parse::<usize>().unwrap();
        let f = parts.next().unwrap().trim()["If false: throw to monkey ".len()..].parse::<usize>().unwrap();
        return Ok(Self {
            items,
            inspections: 0,
            operation: op,
            divisor: div,
            true_monkey: t,
            false_monkey: f
        });
    }
}

fn parse<T:BufRead>(inputs: T) -> Vec<Rc<RefCell<Monkey>>> {
    inputs.lines().map(|f| f.unwrap()).collect::<Vec<String>>()
        .chunks(7)
        .map(|c| c.join("\n"))
        .map(|s| Rc::new(RefCell::new(s.parse::<Monkey>().unwrap())))
        .collect::<Vec<Rc<RefCell<Monkey>>>>()
}

pub fn p1<T:BufRead>(inputs: T) {
    let monkeys = parse(inputs);
    for _ in 0..20 {
        for monkey in &monkeys {
            monkey.borrow_mut().toss(3, 0, &monkeys);
        }
    }

    let mut monkeys = monkeys;
    monkeys.sort_by_key(|m| -(m.borrow().inspections as i32));
    for monkey in &monkeys {
        println!("{:?}", monkey.borrow());
    }
    let monkey_business = monkeys.get(0).unwrap().borrow().inspections *
        monkeys.get(1).unwrap().borrow().inspections;
    println!("{}", monkey_business);
}

pub fn p2<T:BufRead>(inputs: T) {
    let monkeys = parse(inputs);
    let worry_max = monkeys.iter()
        .map(|f| f.borrow().divisor)
        .reduce(|a, v| a * v)
        .unwrap();

    for _ in 0..10_000 {
        for monkey in &monkeys {
            monkey.borrow_mut().toss(1, worry_max, &monkeys);
        }
    }

    let mut monkeys = monkeys;
    monkeys.sort_by_key(|m| -(m.borrow().inspections as i32));
    for monkey in &monkeys {
        println!("{:?}", monkey.borrow());
    }
    let monkey_business = monkeys.get(0).unwrap().borrow().inspections *
        monkeys.get(1).unwrap().borrow().inspections;
    println!("{}", monkey_business);
}