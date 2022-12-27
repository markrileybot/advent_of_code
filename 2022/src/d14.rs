use std::io::{BufRead, stdout, Stdout, Write};
use std::str::FromStr;
use std::thread::sleep;
use std::time::Duration;

pub use crossterm::{
    Command,
    cursor,
    event::{self, Event, KeyCode, KeyEvent, poll}, execute, queue,
    Result,
    style, terminal::{self, ClearType},
};
use crossterm::event::read;

use crate::d14::DropState::{Abyssed, Falling, Full, Resting};
use crate::d14::TileFill::{Abyss, Air, Rock};

#[derive(PartialEq,Debug,Clone)]
enum TileFill {
    Air,
    Rock,
    Sand,
    Abyss
}

#[derive(PartialEq,Debug,Clone)]
enum DropState {
    Falling, Resting, Abyssed, Full
}

struct Sand {
    pos: (i32, i32),
    start: (i32, i32),
    path: Vec<(i32,i32)>
}

impl Sand {
    fn new(start: (i32,i32)) -> Self {
        Self {
            pos: start.clone(),
            path: Vec::new(),
            start
        }
    }
    fn reset(&mut self) {
        self.path.clear();
        self.pos = self.start.clone()
    }
    fn drop(&mut self, map: &mut Map) -> DropState {
        for next_pos in vec![
            (self.pos.0, self.pos.1 + 1),
            (self.pos.0 - 1, self.pos.1 + 1),
            (self.pos.0 + 1, self.pos.1 + 1)] {
            if let Some(t) = map.get_at_mut(&next_pos) {
                if t.fill == Air {
                    self.pos = next_pos.clone();
                    self.path.push(next_pos);
                    return Falling;
                } else if t.fill == Abyss {
                    t.fill = TileFill::Sand;
                    return Abyssed;
                }
            }
        }
        if let Some(t) = map.get_at_mut(&self.pos) {
            if t.fill == TileFill::Sand {
                return Full;
            }
            t.fill = TileFill::Sand;
        }
        return Resting;
    }
    fn render(&self, w: &mut Stdout, view_port: &Viewport) -> Result<()> {
        for x in &self.path {
            if view_port.contains(x) {
                queue!(w, cursor::MoveTo((x.0 - view_port.x0) as u16, (x.1 - view_port.y0) as u16))?;
                queue!(w, style::SetForegroundColor(style::Color::DarkRed), style::Print("▒"))?;
            }
        }
        if view_port.contains(&self.pos) {
            queue!(w, cursor::MoveTo((self.pos.0 - view_port.x0) as u16, (self.pos.1 - view_port.y0) as u16))?;
            queue!(w, style::SetForegroundColor(style::Color::DarkRed), style::Print("▒"))?;
        }
        Ok(())
    }
}

struct Viewport {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
}

impl Viewport {
    fn new(size: (u16, u16)) -> Self {
        let x = 0.max(500i32 - (size.0 / 2) as i32);
        return Self {
            x0: x,
            y0: 0,
            x1: x + size.0 as i32,
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

    fn contains(&self, p: &(i32, i32)) -> bool {
        p.0 >= self.x0 && p.0 <= self.x1 && p.1 >= self.y0 && p.1 <= self.y1
    }
}

struct Tile {
    fill: TileFill,
}

struct Map {
    //    y   x
    grid: Vec<Vec<Tile>>,
}


impl Map {

    fn get_at_mut(&mut self, pos: &(i32, i32)) -> Option<&mut Tile> {
        if pos.0 < 0 || pos.1 < 0 {
            None
        } else if let Some(t) = self.grid.get_mut(pos.1 as usize) {
            t.get_mut(pos.0 as usize)
        } else {
            None
        }
    }

    fn new_p1<T:BufRead>(inputs: T) -> Self {
        let mut map = Self::parse(inputs);
        let second_to_last = map.grid.len() - 2;
        if let Some(l) = map.grid.get_mut(second_to_last) {
            for x in l {
                x.fill = Abyss;
            }
        }
        return map;
    }

    fn new_p2<T:BufRead>(inputs: T) -> Self {
        let mut map = Self::parse(inputs);
        if let Some(l) = map.grid.last_mut() {
            for x in l {
                x.fill = Rock;
            }
        }
        return map;
    }

    fn parse<T: BufRead>(inputs: T) -> Map {
        inputs.lines().map(|f| f.unwrap())
            .collect::<Vec<String>>()
            .join("\n")
            .parse::<Map>()
            .unwrap()
    }

    fn render<W>(&self, w: &mut W, view_port: &Viewport)  -> Result<()>
        where W: Write {

        for (y, row) in self.grid.iter().enumerate() {
            let y = y as i32;
            if y >= view_port.y0 && y <= view_port.y1 {
                for (x, i) in row.iter().enumerate() {
                    let x = x as i32;
                    if x >= view_port.x0 && x <= view_port.x1 && i.fill != Air {
                        let (fill, color) = match i.fill {
                            Air => ('·', style::Color::White),
                            Rock => ('█', style::Color::DarkCyan),
                            TileFill::Sand => ('▒', style::Color::DarkYellow),
                            Abyss => ('~', style::Color::DarkRed)
                        };
                        queue!(w, cursor::MoveTo((x - view_port.x0) as u16, (y - view_port.y0) as u16))?;
                        queue!(w, style::SetForegroundColor(color), style::Print(format!("{}", fill)))?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl FromStr for Map {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let mut walls= Vec::new();
        for s in s.split("\n") {
            let l = s.split(" ")
                .filter(|f| !f.contains("->"))
                .map(|f| f.split_once(",").unwrap())
                .map(|f| (f.0.parse::<i32>().unwrap(), f.1.parse::<i32>().unwrap()))
                .collect::<Vec<(i32,i32)>>();
            walls.push(l);
        }

        let min = (0, 0);
        let mut max = (0 as i32, 0 as i32);
        for v in &walls {
            for x in v {
                max = (max.0.max(x.0), max.1.max(x.1));
            }
        }
        max.0 = 1000;
        max.1 += 2;

        let mut grid: Vec<Vec<Tile>> = Vec::new();
        for _ in min.1..=max.1 {
            let mut row = Vec::new();
            for _ in min.0..=max.0 {
                row.push(Tile {fill: Air});
            }
            grid.push(row);
        }

        let mut map = Map {grid};

        for v in walls {
            for i in 0..v.len() - 1 {
                let p0 = v[i];
                let p1 = v[i + 1];
                let min = p0.min(p1);
                let max = p0.max(p1);
                for x in min.0..=max.0 {
                    for y in min.1..=max.1 {
                        map.get_at_mut(&(x, y)).unwrap().fill = Rock;
                    }
                }
            }
        }

        Ok(map)
    }
}

fn render(map: &Map, sand: &Sand, mut w: &mut Stdout, sand_count: &mut i32, view_port: &mut Viewport, done: bool) -> Result<()> {
    queue!(w, terminal::Clear(ClearType::All))?;
    queue!(w, style::ResetColor, cursor::MoveTo(0, 0), style::Print(format!("Sand: {}", sand_count)))?;
    if done {
        queue!(w, style::SetForegroundColor(style::Color::Cyan), cursor::MoveTo(0, 1), style::Print(format!("Scroll using ARROWs or ESC to quit!")))?;
    }
    map.render(&mut w, &view_port)?;
    sand.render(&mut w, &view_port)?;
    w.flush()?;
    Ok(())
}

pub fn p<T:BufRead>(inputs: T, p1: bool, render_every: i32) {
    (|| -> Result<()> {
        let mut map = if p1 {Map::new_p1(inputs)} else {Map::new_p2(inputs)};
        let mut w = stdout();
        let mut sand_count = 0;
        execute!(w, terminal::Clear(ClearType::All), terminal::EnterAlternateScreen, cursor::Hide)?;
        terminal::enable_raw_mode()?;
        let terminal_size = terminal::size()?;
        let mut view_port = Viewport::new(terminal_size);
        let mut sand = Sand::new((500, 0));

        loop {
            sand.reset();
            let mut drop_result;

            loop {
                drop_result = sand.drop(&mut map);
                if drop_result != Falling {
                    break;
                }
            }

            if drop_result == Resting {
                sand_count += 1;
            }

            if sand_count % render_every == 0 {
                view_port.shift(&sand.pos);
                render(&map, &sand, &mut w, &mut sand_count, &mut view_port, false)?;
                sleep(Duration::from_millis(5));
            }

            if drop_result == Abyssed || drop_result == Full {
                break;
            }
        }

        render(&map, &sand, &mut w, &mut sand_count, &mut view_port, true)?;
        loop {
            let event = read()?;
            view_port.shift_by(
                if event == Event::Key(KeyCode::Left.into()) {
                    (-5, 0)
                } else if event == Event::Key(KeyCode::Right.into()) {
                    (5, 0)
                } else if event == Event::Key(KeyCode::Up.into()) {
                    (0, -5)
                } else if event == Event::Key(KeyCode::Down.into()) {
                    (0, 5)
                } else {
                    (0, 0)
                }
            );
            if event == Event::Key(KeyCode::Esc.into()) {
                break;
            }
            render(&map, &sand, &mut w, &mut sand_count, &mut view_port, true)?;
        }

        terminal::disable_raw_mode()?;
        execute!(w, terminal::LeaveAlternateScreen)?;

        terminal::enable_raw_mode()?;
        render(&map, &sand, &mut w, &mut sand_count, &mut view_port, false)?;
        execute!(w, cursor::MoveTo(0u16, (map.grid.len() + 2) as u16), cursor::Show)?;
        terminal::disable_raw_mode()?;
        Ok(())
    })().unwrap();
}

pub fn p1<T:BufRead>(inputs: T) {
    p(inputs, true, 1);
}

pub fn p2<T:BufRead>(inputs: T) {
    p(inputs, false, 40);
}