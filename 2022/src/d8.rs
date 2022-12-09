use std::cell::RefCell;
use std::rc::Rc;

use colored::{Color, Colorize};

#[derive(Debug, Clone)]
enum Direction {
    Left = 0,
    Right = 1,
    Up = 2,
    Down = 3
}

struct Tree {
    links: Vec<Option<Rc<RefCell<Tree>>>>,
    height: u8
}

impl Tree {
    fn line_feed(&self) {
        match &self.links.get(Direction::Left as usize).unwrap() {
            None => {
                self.print();
            }
            Some(tb) => {
                tb.borrow().line_feed();
            }
        }
    }

    fn print(&self) {
        let color = if self.visible().is_empty() {
            Color::BrightCyan
        } else {
            Color::White
        };
        print!("{}", format!("{}", self.height).color(color));
        match &self.links.get(Direction::Right as usize).unwrap() {
            None => {
                match &self.links.get(Direction::Down as usize).unwrap() {
                    None => {}
                    Some(t) => {
                        println!();
                        t.borrow().line_feed()
                    }
                }
            }
            Some(t) => t.borrow().print()
        }
    }

    fn clear_below(&self, dir: Direction, height: u8) -> bool {
        if self.height < height {
            return match &self.links.get(dir.clone() as usize).unwrap() {
                None => true,
                Some(t) => {
                    let tb = t.borrow();
                    tb.clear_below(dir, height)
                }
            }
        }
        return false;
    }

    fn visible_from(&self, dir: Direction) -> bool {
        return match &self.links.get(dir.clone() as usize).unwrap() {
            None => true,
            Some(t) => {
                let tb = t.borrow();
                tb.clear_below(dir, self.height)
            }
        }
    }

    fn scenic_score_below(&self, dir: Direction, height: u8) -> u32 {
        if height > self.height {
            return 1 + match &self.links.get(dir.clone() as usize).unwrap() {
                None => 0,
                Some(t) => {
                    let tb = t.borrow();
                    tb.scenic_score_below(dir, height)
                }
            }
        }
        return 1;
    }

    fn scenic_score_from(&self, dir: Direction) -> u32 {
        return match &self.links.get(dir.clone() as usize).unwrap() {
            None => 0,
            Some(t) => {
                let tb = t.borrow();
                tb.scenic_score_below(dir, self.height)
            }
        }
    }

    fn scenic_score(&self) -> u32 {
        return self.scenic_score_from(Direction::Left)
            * self.scenic_score_from(Direction::Right)
            * self.scenic_score_from(Direction::Up)
            * self.scenic_score_from(Direction::Down)
    }

    fn visible(&self) -> Vec<Direction> {
        let mut v = Vec::new();
        if self.visible_from(Direction::Left) {
            v.push(Direction::Left)
        }
        if self.visible_from(Direction::Up) {
            v.push(Direction::Up)
        }
        if self.visible_from(Direction::Right) {
            v.push(Direction::Right)
        }
        if self.visible_from(Direction::Down) {
            v.push(Direction::Down)
        }
        return v;
    }
}



fn parse(inputs: &Vec<String>) -> Vec<Rc<RefCell<Tree>>> {
    let mut forest: Vec<Rc<RefCell<Tree>>> = Vec::new();
    for (row, s) in inputs.iter().enumerate() {
        let row_size = s.len();
        for (col, h) in s.chars().enumerate() {
            let new_tree = Rc::new(RefCell::new(Tree {
                links: vec![None, None, None, None],
                height: h.to_digit(10).unwrap() as u8
            }));
            if col > 0 {
                let left = forest.get(forest.len() - 1).unwrap();
                new_tree.borrow_mut().links[Direction::Left as usize] = Some(left.clone());
                left.borrow_mut().links[Direction::Right as usize] = Some(new_tree.clone());
            }
            if row > 0 {
                let above = forest.get(forest.len() - row_size).unwrap();
                new_tree.borrow_mut().links[Direction::Up as usize] = Some(above.clone());
                above.borrow_mut().links[Direction::Down as usize] = Some(new_tree.clone());
            }
            forest.push(new_tree.clone());
        }
    }
    forest
}

pub fn p1(inputs: &Vec<String>) {
    let forest = parse(inputs);
    let mut visible_count = 0;
    for x in &forest {
        let dirs = x.borrow().visible();
        if !dirs.is_empty() {
            visible_count += 1;
        }
    }
    forest.get(0).unwrap().borrow().print();
    println!();
    println!("{}", visible_count);
}

pub fn p2(inputs: &Vec<String>) {
    let forest = parse(inputs);
    let mut max_score = 0;
    for x in &forest {
        let score = x.borrow().scenic_score();
        if score > max_score {
            max_score = score;
            println!("{}", max_score)
        }
    }
}