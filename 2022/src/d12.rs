use std::io::BufRead;
use std::io::stdout;
use std::io::Write;
use std::str::FromStr;
use std::thread::sleep;
use std::time::Duration;

use bytebuffer::ByteBuffer;
pub use crossterm::{
    Command,
    cursor,
    event::{self, Event, KeyCode, KeyEvent}, execute, queue,
    Result,
    style, terminal::{self, ClearType},
};
use crossterm::style::Print;

#[derive(Debug, Clone)]
struct Path {
    id: u8,
    points: Vec<(i32, i32)>,
    done: Option<bool>
}
impl Path {
    fn start(start_pos: &(i32, i32)) -> Self {
        Path {
            id: 0,
            points: vec![start_pos.clone()],
            done: None
        }
    }
    fn head(&self) -> (i32, i32) {
        self.points.last().unwrap().clone()
    }
    fn contains(&self, point: &(i32, i32)) -> bool {
        self.points.contains(point)
    }
    fn is_done(&self) -> bool {
        self.done.is_some()
    }
    fn hit_target(&self) -> bool {
        self.done.unwrap_or(false)
    }
    fn set_done(&mut self, hit_target: bool) {
        self.done = Some(hit_target);
    }
}

#[derive(Debug, Clone)]
struct PathGraph {
    paths: Vec<Path>
}
impl PathGraph {
    fn new(map: &Map) -> Self {
        PathGraph {
            paths: vec![Path::start(&map.start_pos)]
        }
    }

    fn leader(&self) -> Option<usize> {
        self.paths.iter()
            .filter(|p| p.hit_target())
            .map(|p| p.points.len() - 1)
            .min()
    }

    fn advance(&mut self, map: &mut Map) -> bool {
        let mut new_paths = Vec::new();
        self.paths.iter_mut()
            .filter(|path| !path.is_done())
            .for_each(|path| {
                let mut count = 0;
                let head = path.head();
                for neighbor in vec![(head.0, head.1 - 1),
                                     (head.0 + 1, head.1),
                                     (head.0 - 1, head.1),
                                     (head.0, head.1 + 1)] {
                    if let Some(next) = map.get_neighbor(&head, neighbor) {
                        if !path.contains(&next) {
                            map.set_visited(&next);
                            if count == 0 {
                                path.points.push(next);
                                if next == map.end_pos {
                                    path.set_done(true);
                                }
                            } else {
                                let mut p = path.clone();
                                p.id = path.id + 1;
                                p.points.pop();
                                p.points.push(next);
                                if next == map.end_pos {
                                    path.set_done(true);
                                }
                                new_paths.push(p);
                            }
                            count += 1;
                        }
                    }
                }
                if count == 0 {
                    path.set_done(false);
                }
            });

        self.paths.extend(new_paths);
        self.paths.iter().all(|p| p.is_done())
    }

    fn render<W>(&self, w: &mut W)  -> Result<()>
        where W: Write {

        let mut min_len = u32::MAX as usize;
        let mut leader = None;
        for p in self.paths.iter()
            .filter(|p| !p.is_done() || p.hit_target()) {
            if p.points.len() < min_len {
                min_len = p.points.len();
                leader = Some(p);
            }
        }
        if let Some(leader) = leader {
            for x in &leader.points {
                queue!(w, cursor::MoveTo(x.0 as u16, (x.1 + 1) as u16))?;
                queue!(w, style::SetForegroundColor(style::Color::Red), style::Print(format!("X")))?;
            }
        }

        Ok(())
    }
}

struct Tile {
    height: i32,
    visited: bool
}

impl Tile {
    fn new(height: char) -> Self {
        return Self {
            height: height as i32,
            visited: false
        }
    }
}

struct Map {
    //    y   x
    grid: Vec<Vec<Tile>>,
    start_pos: (i32, i32),
    end_pos: (i32, i32)
}

impl Map {
    fn start_at(&mut self, start: &(i32, i32)) {
        for row in self.grid.iter_mut() {
            for t in row.iter_mut() {
                t.visited = false;
            }
        }
        self.start_pos = start.clone();
    }
    fn get_starts(&self) -> Vec<(i32,i32)> {
        let lowest = 'a' as i32;
        let mut starts = Vec::new();
        for (y, row) in self.grid.iter().enumerate() {
            for (x, t) in row.iter().enumerate() {
                if t.height == lowest {
                    starts.push((x as i32, y as i32))
                }
            }
        }
        return starts;
    }
    fn set_visited(&mut self, xy: &(i32, i32)) {
        if let Some(tiles) = self.grid.get_mut(xy.1 as usize) {
            if let Some(tile) = tiles.get_mut(xy.0 as usize)  {
                tile.visited = true;
            }
        }
    }
    fn get_neighbor(&self, current: &(i32, i32), next: (i32, i32)) -> Option<(i32, i32)> {
        match self.get_at(current) {
            None => None,
            Some(c) => match self.get_at(&next) {
                None => None,
                Some(v) => if !v.visited && v.height - c.height <= 1 {Some(next)} else {None}
            }
        }
    }
    fn get_at(&self, xy: &(i32, i32)) -> Option<&Tile> {
        match self.grid.get(xy.1 as usize) {
            None => None,
            Some(v) => v.get(xy.0 as usize)
        }
    }

    fn render<W>(&self, w: &mut W)  -> Result<()>
        where W: Write {

        for (y, row) in self.grid.iter().enumerate() {
            for (x, i) in row.iter().enumerate() {
                let color = if i.visited {style::Color::Magenta} else {style::Color::Grey};
                let height = char::from_u32(i.height as u32).unwrap_or('?');
                queue!(w, cursor::MoveTo(x as u16, (y + 1) as u16))?;
                queue!(w, style::SetForegroundColor(color), style::Print(format!("{}", height)))?;
            }
        }
        Ok(())
    }
}

impl FromStr for Map {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let mut rows = Vec::new();
        let mut start_pos = (0i32, 0i32);
        let mut end_pos = (0i32, 0i32);
        rows.push(Vec::new());
        for c in s.chars() {
            match c {
                'S' => {
                    start_pos.0 = rows.last().unwrap().len() as i32;
                    start_pos.1 = (rows.len() as i32) - 1;
                    rows.last_mut().unwrap().push(Tile::new('a'));
                },
                'E' => {
                    end_pos.0 = rows.last().unwrap().len() as i32;
                    end_pos.1 = (rows.len() as i32) - 1;
                    rows.last_mut().unwrap().push(Tile::new('z'));
                },
                '\n' => {
                    rows.push(Vec::new());
                },
                c => {
                    rows.last_mut().unwrap().push(Tile::new(c));
                }
            }
        }

        Ok(Map {
            grid: rows,
            start_pos,
            end_pos
        })
    }
}

fn render<W>(map: &Map, graph: &PathGraph, w: &mut W) -> Result<()>
    where W: Write {
    let num_paths = graph.paths.len();
    let leader = graph.leader().map(|v| format!("{}", v)).unwrap_or("?".to_string());
    queue!(w, style::ResetColor, terminal::Clear(ClearType::All))?;
    map.render(w)?;
    graph.render(w)?;
    queue!(w, style::ResetColor, cursor::MoveTo(0, 0), Print(format!("Tail paths: {}.  Leader: {}", num_paths, leader)))?;
    w.flush()?;
    Ok(())
}

fn parse<T: BufRead>(inputs: T) -> Map {
    inputs.lines()
        .map(|f| f.unwrap())
        .collect::<Vec<String>>().join("\n")
        .parse::<Map>().unwrap()
}

pub fn p1<T:BufRead>(inputs: T) {
    (|| -> Result<()> {
        let mut map = parse(inputs);
        let mut graph = PathGraph::new(&map);

        let mut w = stdout();
        execute!(w, terminal::EnterAlternateScreen, cursor::Hide)?;
        terminal::enable_raw_mode()?;

        while !graph.advance(&mut map) {
            render(&map, &graph, &mut w)?;
            w.flush()?;
            sleep(Duration::from_millis(5));
        }

        terminal::disable_raw_mode()?;
        execute!(w, terminal::LeaveAlternateScreen)?;

        terminal::enable_raw_mode()?;
        render(&map, &graph, &mut w)?;
        w.flush()?;
        execute!(w, cursor::Show, cursor::MoveTo(0, (map.grid.len() + 1) as u16))?;
        terminal::disable_raw_mode()?;
        println!();
        Ok(())
    })().unwrap();
}

pub fn p2<T:BufRead>(inputs: T) {
    (|| -> Result<()> {
        let mut map = parse(inputs);
        let height = (map.grid.len() + 1) as u16;

        let mut w = stdout();
        let mut b = ByteBuffer::new();
        let mut a = ByteBuffer::new();
        execute!(w, terminal::EnterAlternateScreen, cursor::Hide)?;
        terminal::enable_raw_mode()?;

        let mut current_leader = u32::MAX as usize;
        for xy in map.get_starts() {
            map.start_at(&xy);
            let mut graph = PathGraph::new(&map);
            while !graph.advance(&mut map) {}
            if let Some(leader) = graph.leader() {
                if leader < current_leader {
                    b.clear();
                    render(&map, &graph, &mut b)?;
                    current_leader = leader;
                    w.write_all(b.as_bytes())?;
                }
            }
            if let Some(t) = map.get_at(&xy) {
                let height = char::from_u32(t.height as u32).unwrap_or('?');
                queue!(a, cursor::MoveTo(xy.0 as u16, (xy.1 + 1) as u16))?;
                queue!(a, style::SetForegroundColor(style::Color::Black), style::Print(format!("{}", height)))?;
                w.write_all(a.as_bytes())?;
            }
        }

        terminal::disable_raw_mode()?;
        execute!(w, terminal::LeaveAlternateScreen)?;

        terminal::enable_raw_mode()?;
        w.write_all(b.as_bytes())?;
        w.write_all(a.as_bytes())?;
        execute!(w, cursor::MoveTo(0, height), cursor::Show)?;
        terminal::disable_raw_mode()?;

        println!("{}", current_leader);
        Ok(())
    })().unwrap();
}

