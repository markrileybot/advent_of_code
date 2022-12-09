use std::{io, result};
use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use std::io::{Stdout, Write};
use std::str::FromStr;
use std::thread::sleep;
use std::time::Duration;

use colored::Colorize;
pub use crossterm::{
    Command,
    cursor,
    event::{self, Event, KeyCode, KeyEvent}, execute, queue,
    Result,
    style, terminal::{self, ClearType},
};
use crossterm::style::Print;

#[derive(Debug, Clone)]
enum Direction {
    Left,
    Right,
    Up,
    Down
}

impl Direction {
    fn mv(&self, pos: &mut (i32, i32), spaces: i32) {
        match self {
            Direction::Left => {
                pos.0 -= spaces;
            }
            Direction::Right => {
                pos.0 += spaces;
            }
            Direction::Up => {
                pos.1 += spaces;
            }
            Direction::Down => {
                pos.1 -= spaces;
            }
        }
    }
}

impl FromStr for Direction {
    type Err = String;
    fn from_str(s: &str) -> result::Result<Self, Self::Err> {
        return match s {
            "L" => Ok(Direction::Left),
            "R" => Ok(Direction::Right),
            "U" => Ok(Direction::Up),
            "D" => Ok(Direction::Down),
            &_ => Err(format!("Invalid direction {}", s).to_string())
        };
    }
}

#[derive(Clone)]
struct Map {
    rope: Vec<(i32, i32)>,
    visited: HashSet<(i32,i32)>,
    grid_size: (i32, i32)
}

impl Map {
    fn new(knot_count: u8) -> Self {
        let mut rope = Vec::new();
        for _ in 0..knot_count {
            rope.push((0, 0))
        }
        return Map {
            rope,
            visited: HashSet::new(),
            grid_size: (0, 0),
        }
    }

    fn mv(&mut self, direction: Direction, spaces: i32) {
        for _ in 0..spaces {
            direction.mv(self.rope.get_mut(0).unwrap(), 1);
            let tail_idx = self.rope.len() - 1;
            let mut previous: Option<(i32, i32)> = None;
            for (i, knot) in self.rope.iter_mut().enumerate() {
                match previous {
                    None => {}
                    Some(prev) => {
                        let dx = prev.0 - knot.0;
                        let dy = prev.1 - knot.1;
                        if (dx.abs() > 1 && dy.abs() > 0) || (dy.abs() > 1 && dx.abs() > 0) {
                            knot.0 += 1 * dx.signum();
                            knot.1 += 1 * dy.signum();
                        } else if dx.abs() > 1 {
                            knot.0 += 1 * dx.signum();
                        } else if dy.abs() > 1 {
                            knot.1 += 1 * dy.signum();
                        } else {
                            break;
                        }
                        if i == tail_idx {
                            self.visited.insert(knot.clone());
                        }
                    }
                }
                previous = Some(knot.clone());
            }
        }
    }

    fn init(&mut self, moves: &Vec<(Direction, i32)>) {
        let mut max_pos = (i32::MIN, i32::MIN);
        let mut min_pos = (i32::MAX, i32::MAX);
        let mut c_pos = (0, 0);
        for (m, i) in moves {
            m.mv(&mut c_pos, *i);
            max_pos.0 = max_pos.0.max(c_pos.0);
            max_pos.1 = max_pos.1.max(c_pos.1);
            min_pos.0 = min_pos.0.min(c_pos.0);
            min_pos.1 = min_pos.1.min(c_pos.1);
        }
        self.grid_size.0 = (max_pos.0 - min_pos.0) + 1;
        self.grid_size.1 = (max_pos.1 - min_pos.1) + 1;
        let start_pos = (0 - min_pos.0, 0 - min_pos.1);
        for x in self.rope.iter_mut() {
            x.0 = start_pos.0;
            x.1 = start_pos.1;
        }
        self.visited.insert(start_pos);
    }

    fn cross_term_render<W>(&self, w: &mut W, x_scale: f32, y_scale: f32)  -> Result<()>
        where W: Write {
        let tail_idx = self.rope.len() - 1;
        let grid_height = self.grid_size.1;
        queue!(w, style::ResetColor, style::SetForegroundColor(style::Color::Magenta))?;
        for (x, y) in &self.visited {
            queue!(w, cursor::MoveTo((*x as f32 * x_scale) as u16, ((grid_height - *y) as f32 * y_scale) as u16), style::Print("#"))?;
        }
        for (i, knot) in self.rope.iter().enumerate().rev() {
            queue!(w, cursor::MoveTo((knot.0 as f32 * x_scale) as u16, ((grid_height - knot.1) as f32 * y_scale) as u16))?;
            if i == 0 {
                queue!(w, style::SetForegroundColor(style::Color::Red), style::Print("H"))?;
            } else if i == tail_idx {
                queue!(w, style::SetForegroundColor(style::Color::Green), style::Print("T"))?;
            } else {
                queue!(w, style::SetForegroundColor(style::Color::Yellow), style::Print(format!("{}", i)))?;
            }
        }
        Ok(())
    }
}

impl Display for Map {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let tail_idx = self.rope.len() - 1;
        for y in (0..self.grid_size.1).rev() {
            for x in 0..self.grid_size.0 {
                let pos = (x, y);
                let mut printed = false;
                for (i, knot) in self.rope.iter().enumerate() {
                    if knot == &pos {
                        if i == 0 {
                            write!(f, "{}", "H".bright_red()).unwrap();
                        } else if i == tail_idx {
                            write!(f, "{}", "T".bright_green()).unwrap();
                        } else {
                            write!(f, "{}", format!("{}", i).bright_yellow()).unwrap();
                        }
                        printed = true;
                    }
                    if printed {
                        break;
                    }
                }
                if !printed {
                    if self.visited.contains(&pos) {
                        write!(f, "{}", "#".bright_magenta()).unwrap();
                    } else {
                        write!(f, " ").unwrap();
                    }
                }
            }
            write!(f, "\n").unwrap();
        }
        Ok(())
    }
}

fn p(inputs: &Vec<String>, num_knots: u8) -> Result<()> {
    let mut map = Map::new(num_knots);
    let moves = inputs.iter()
        .map(|s| s.split_once(' ').unwrap())
        .map(|(m, i) | (m.parse::<Direction>().unwrap(), i.parse::<i32>().unwrap()))
        .collect::<Vec<(Direction, i32)>>();
    map.init(&moves);
    let speed = Duration::from_millis(if moves.len() > 1000 {
        1
    } else if moves.len() > 100 {
        10
    } else {
        100
    });

    let mut w = io::stdout();
    execute!(w, terminal::EnterAlternateScreen)?;
    terminal::enable_raw_mode()?;
    let (width, height) = terminal::size()?;
    let x_scale = (width as f32 / map.grid_size.0 as f32).min(1.0);
    let y_scale = (height as f32 / map.grid_size.1 as f32).min(1.0);
    queue!(w, cursor::Hide)?;
    render(&mut map, &mut w, x_scale, y_scale)?;
    for (mv, i) in moves {
        for _ in 0..i {
            sleep(speed);
            map.mv(mv.clone(), 1);
            render(&map, &mut w, x_scale, y_scale)?;
        }
    }
    terminal::disable_raw_mode()?;
    execute!(w, terminal::LeaveAlternateScreen)?;

    terminal::enable_raw_mode()?;
    queue!(w, cursor::Hide)?;
    render(&map, &mut w, x_scale, y_scale)?;
    execute!(w, cursor::MoveTo(0, height.min((map.grid_size.1 + 1) as u16)))?;
    queue!(w, cursor::Show)?;
    terminal::disable_raw_mode()?;
    Ok(())
}

fn render(map: &Map, mut w: &mut Stdout, x_scale: f32, y_scale: f32) -> Result<()> {
    let visited_len = map.visited.len();
    queue!(w, style::ResetColor, terminal::Clear(ClearType::All))?;
    queue!(w, style::ResetColor, cursor::MoveTo(0, 0), Print(format!("Tail visits: {}", visited_len)))?;
    map.cross_term_render(&mut w, x_scale, y_scale)?;
    w.flush()?;
    Ok(())
}

pub fn p1(inputs: &Vec<String>) {
    p(inputs, 2).unwrap();
}

pub fn p2(inputs: &Vec<String>) {
    p(inputs, 10).unwrap();
}