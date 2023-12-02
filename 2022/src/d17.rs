use std::collections::HashSet;
use std::io::{BufRead, stdout, Stdout, Write};
use std::str::FromStr;
use std::string::ToString;
use std::thread::sleep;
use std::time::Duration;

pub use crossterm::{
    Command,
    cursor,
    event::{self, Event, KeyCode, KeyEvent, poll}, execute, queue,
    style, terminal::{self, ClearType},
};

#[derive(Debug, Clone, Default, Eq, PartialEq, Hash)]
struct Point {
    x: i32,
    y: i32
}


struct Viewport {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
}

impl Viewport {
    fn new(size: (u16, u16)) -> Self {
        return Self {
            x0: 0,
            y0: 0,
            x1: size.0 as i32,
            y1: size.1 as i32
        }
    }

    fn shift_by(&mut self, off: (i32, i32)) {
        self.x0 += off.0;
        self.x1 += off.0;

        self.y0 += off.1;
        self.y1 += off.1;
    }

    fn shift(&mut self, pos: &(i32, i32)) {
        if pos.0 < self.x0 || pos.0 > self.x1 {
            let off = pos.0 - self.x1;
            self.x0 += off;
            self.x1 += off;
        }

        if pos.1 < self.y0 || pos.1 > self.y1 {
            let off = pos.1 - self.y1;
            self.y0 += off;
            self.y1 += off;
        }
    }

    fn contains(&self, p: &Point) -> bool {
        p.x >= self.x0 && p.x <= self.x1 && p.y >= self.y0 && p.y <= self.y1
    }
}

#[derive(Debug, Clone, Default, Eq, PartialEq, Hash)]
struct Rock {
    shape: Vec<Point>
}
impl Rock {
    fn start(&self, map: &Map) -> Self {
        let mut next = self.clone();
        next.shift(2, map.height - self.height());
        return next;
    }
    fn shift(&mut self, x: i32, y: i32) {
        for p in self.shape.iter_mut() {
            p.x += x;
            p.y += y;
        }
    }
    fn mv(&mut self, mv: char, map: &Map) {
        let x_by = match mv {
            '>' => 1,
            '<' => -1,
            _ => 0
        };
        self.shift(x_by, 0);
        if self.collision(map) {
            self.shift(-x_by, 0);
        }
    }
    fn fall(&mut self, map: &Map) -> bool {
        self.shift(0, -1);
        if self.collision(map) {
            self.shift(0, 1);
            return false;
        }
        return true;
    }
    fn top(&self) -> i32 {
        self.shape.iter().map(|p| p.y).max().unwrap() + 1
    }
    fn bottom(&self) -> i32 {
        self.shape.iter().map(|p| p.y).min().unwrap()
    }
    fn height(&self) -> i32 {
        self.top() - self.bottom()
    }
    fn collision(&self, m: &Map) -> bool {
        self.shape.iter().any(|p|  {
            if p.x < 0 || p.x >= m.width || p.y < 0 {
                // println!("OOB {:?}", p);
                return true;
            }
            return false;
        })
            || self.shape.iter().any(|p| {
            if m.pile.contains(p) {
                // println!("COLLISION {:?}", p);
                return true;
            }
            return false;
        })
    }
}
impl FromStr for Rock {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut x = 0;
        let mut y = (s.lines().count() - 1) as i32;
        let mut shape = Vec::new();
        for l in s.lines() {
            x = 0;
            for c in l.chars() {
                if c == '#' {
                    shape.push(Point {x, y});
                }
                x += 1;
            }
            y -= 1;
        }

        Ok(Rock {shape})
    }
}
struct Map {
    rock_stream: Vec<Rock>,
    rock: Option<Rock>,
    pile: HashSet<Point>,
    height: i32,
    width: i32,
    rock_count: usize
}
impl Map {
    fn new() -> Self {
        let mut map = Self {
            rock_stream: ROCK_PATTERN.trim()
                .split("\n\n")
                .map(|s| s.parse::<Rock>().unwrap())
                .collect::<Vec<Rock>>(),
            pile: HashSet::new(),
            height: 0,
            width: 7,
            rock: None,
            rock_count: 0
        };
        map.rock = Some(map.next_rock());
        return map;
    }
    fn next_rock(&mut self) -> Rock {
        let next = self.rock_count % self.rock_stream.len();
        let next = self.rock_stream.get(next).unwrap();
        self.rock_count += 1;
        self.height = 3 + next.height() + self.pile.iter().map(|p| p.y).max().unwrap_or(0);
        next.start(self)
    }
    fn mv(&mut self, mv: char) {
        let mut rock = self.rock.take().unwrap();
        rock.mv(mv, &self);
        self.rock = Some(rock);
    }
    fn fall(&mut self) {
        let mut rock = self.rock.take().unwrap();
        if rock.fall(&self) {
            self.rock = Some(rock);
        } else {
            for x in rock.shape {
                self.pile.insert(x);
            }
            self.rock = Some(self.next_rock());
        }
    }
    fn render<W>(&self, w: &mut W, view_port: &Viewport)  -> crossterm::Result<()>
        where W: Write {
        for y in 0..self.height {
            for x in 0..self.width {
                queue!(w, cursor::MoveTo((x - view_port.x0) as u16, (self.height - y - view_port.y0) as u16))?;
                queue!(w, style::SetForegroundColor(style::Color::Blue), style::Print("·"))?;
            }
        }
        for x in 0..self.width {
            queue!(w, cursor::MoveTo((x - view_port.x0) as u16, (self.height - view_port.y0) as u16))?;
            queue!(w, style::SetForegroundColor(style::Color::Red), style::Print("_"))?;
        }
        for p in &self.pile {
            if view_port.contains(p) {
                queue!(w, cursor::MoveTo((p.x - view_port.x0) as u16, (self.height - p.y - view_port.y0) as u16))?;
                queue!(w, style::SetForegroundColor(style::Color::DarkCyan), style::Print("#"))?;
            }
        }
        if let Some(r) = &self.rock {
            for p in &r.shape {
                if view_port.contains(p) {
                    queue!(w, cursor::MoveTo((p.x - view_port.x0) as u16, (self.height - p.y - view_port.y0) as u16))?;
                    queue!(w, style::SetForegroundColor(style::Color::Yellow), style::Print("@"))?;
                }
            }
        }
        Ok(())
    }
}

const ROCK_PATTERN: &str = r###"
####

.#.
###
.#.

..#
..#
###

#
#
#
#

##
##
"###;

const MOVES: &str = ">>><<><>><<<>><>>><<<>>><<<><<<>><>><<>>";

pub fn p1<T:BufRead>(inputs: T) {
    (|| -> crossterm::Result<()> {
        let mut w = stdout();
        execute!(w, terminal::Clear(ClearType::All), terminal::EnterAlternateScreen, cursor::Hide)?;
        terminal::enable_raw_mode()?;
        let terminal_size = terminal::size()?;
        let mut view_port = Viewport::new(terminal_size);

        let mut map = Map::new();
        loop {
            if map.rock_count > 2 {
                break;
            }
            for x in MOVES.chars() {
                map.mv(x);
                render(&mut w, &view_port, &map)?;

                map.fall();
                render(&mut w, &view_port, &map)?;
                if map.rock_count > 2 {
                    break;
                }
            }
        }

        terminal::disable_raw_mode()?;
        execute!(w, terminal::LeaveAlternateScreen)?;
        terminal::enable_raw_mode()?;
        map.render(&mut w, &view_port)?;
        execute!(w, cursor::Show)?;
        terminal::disable_raw_mode()?;

        println!("\n\n\n{}", map.height);
        Ok(())
    })().unwrap();
}

fn render(mut w: &mut Stdout, view_port: &Viewport, map: &Map) -> crossterm::Result<()> {
    queue!(w, terminal::Clear(ClearType::All))?;
    map.render(&mut w, &view_port)?;
    w.flush()?;
    sleep(Duration::from_millis(1000));
    Ok(())
}

pub fn p2<T:BufRead>(inputs: T) {
}







