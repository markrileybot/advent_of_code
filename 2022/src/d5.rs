use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Clone, Debug)]
struct Stack {
    id: u8,
    crates: Vec<char>
}

impl Stack {
    fn mv(&mut self, num: u8, to: &RefCell<Stack>) {
        println!("Move {} from {:?} to {:?}", num, self, to.borrow());
        for _ in 0..num {
            if let Some(c) = self.crates.pop() {
                to.borrow_mut().crates.push(c);
            } else {
                panic!("Can not move {} from {} to {}", num, self.id, to.borrow().id);
            }
        }
    }

    fn mv_stack(&mut self, num: u8, to: &RefCell<Stack>) {
        println!("Move {} from {:?} to {:?}", num, self, to.borrow());
        for x in self.crates.drain(self.crates.len() - num as usize..) {
            to.borrow_mut().crates.push(x);
        }
    }

    fn get_top(&self) -> char {
        self.crates.get(self.crates.len() - 1).unwrap().clone()
    }
}

fn parse(inputs: &Vec<String>) -> (HashMap<u8, RefCell<Stack>>, Vec<(u8, u8, u8)>) {
    let mut stacks = HashMap::new();
    let mut moves = Vec::new();
    let mut reading_stacks = true;
    for x in inputs {
        if !stacks.is_empty() && (x.trim().is_empty() || x.chars().nth(0).unwrap_or(' ') == ' ') {
            reading_stacks = false;
            continue;
        } else if reading_stacks {
            for (i, c) in x.chars().enumerate() {
                if c != ' ' && c != '[' && c != ']' {
                    let stack_id = (i as f32 / 4f32).ceil() as u8;
                    if !stacks.contains_key(&stack_id) {
                        stacks.insert(stack_id, RefCell::new(Stack {id: stack_id, crates: Vec::new()}));
                    }
                    stacks.get(&stack_id).unwrap().borrow_mut().crates.insert(0, c);
                }
            }
        } else {
            let move_parts = x.split(' ').collect::<Vec<&str>>();
            if move_parts.len() == 6 {
                moves.push((
                    move_parts.get(1).map(|s| s.parse::<u8>().unwrap()).unwrap(),
                    move_parts.get(3).map(|s| s.parse::<u8>().unwrap()).unwrap(),
                    move_parts.get(5).map(|s| s.parse::<u8>().unwrap()).unwrap()
                ));
            }
        }
    }
    return (stacks, moves);
}

pub fn p1(inputs: &Vec<String>) {
    let (stacks, moves) = parse(inputs);
    for mv in moves {
        if let Some(from) = stacks.get(&mv.1) {
            if let Some(to) = stacks.get(&mv.2) {
                from.borrow_mut().mv(mv.0, to);
            }
        }
    }
    let mut stacks = stacks.values().map(|s| s.borrow().clone())
        .collect::<Vec<Stack>>();
    stacks.sort_by_key(|s| s.id);
    stacks.iter().for_each(|s| print!("{}", s.get_top()));
    println!();
}

pub fn p2(inputs: &Vec<String>) {
    let (stacks, moves) = parse(inputs);
    for mv in moves {
        if let Some(from) = stacks.get(&mv.1) {
            if let Some(to) = stacks.get(&mv.2) {
                from.borrow_mut().mv_stack(mv.0, to);
            }
        }
    }
    let mut stacks = stacks.values().map(|s| s.borrow().clone())
        .collect::<Vec<Stack>>();
    stacks.sort_by_key(|s| s.id);
    stacks.iter().for_each(|s| print!("{}", s.get_top()));
    println!();
}

