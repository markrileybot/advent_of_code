use std::io::BufRead;
use std::str::FromStr;
use std::time::SystemTime;

#[derive(Debug, Clone, Default)]
struct Point
{
    x: i32,
    y: i32
}

impl Point {
    fn distance(&self, p: &Point) -> i32 {
        (self.x - p.x).abs() + (self.y - p.y).abs()
    }
}

#[derive(Debug, Clone, Default)]
struct Rect
{
    p0: Point,
    p1: Point
}

impl Rect
{
    fn new(p0: Point, p1: Point) -> Self {
        Self {
            p0,
            p1
        }
    }

    fn intersects(&self, other: &Rect) -> bool {
        self.p0.x < other.p1.x
            && self.p1.x > other.p0.x
            && self.p0.y < other.p1.y
            && self.p1.y > other.p0.y
    }

    fn subdivide(&self) -> [Rect; 4] {
        let x0 = self.p0.x;
        let y0 = self.p0.y;
        let x1 = self.p1.x;
        let y1 = self.p1.y;
        let mid_x = x0 + (x1 - x0) / 2;
        let mid_y = y0 + (y1 - y0) / 2;
        [
            Rect::new(Point {x: x0, y: y0}, Point {x: mid_x, y: mid_y}),
            Rect::new(Point {x: mid_x, y: y0}, Point {x: x1, y: mid_y}),
            Rect::new(Point {x: x0, y: mid_y}, Point {x: mid_x, y: y1}),
            Rect::new(Point {x: mid_x, y: mid_y}, Point {x: x1, y: y1})
        ]
    }
}


#[derive(Debug)]
struct QuadTree
{
    quads: Option<Box<[QuadTree; 4]>>,
    bounds: Rect,
    item_mask: u32,
    level: u8,
    max_depth: u8,
    occluded: bool
}

impl QuadTree {
    fn new(bounds: Rect, max_depth: u8) -> Self {
        Self::new_quad(bounds, max_depth, 0)
    }
    fn new_quad(bounds: Rect, max_depth: u8, level: u8) -> Self {
        Self {
            bounds,
            max_depth,
            level,
            quads: None,
            item_mask: 0,
            occluded: false
        }
    }

    fn insert(&mut self, id: usize, sensor: &Sensor) {
        self.insert_internal(id, sensor, &sensor.rect());
    }

    fn insert_internal(&mut self, id: usize, sensor: &Sensor, v: &Rect) {
        if !self.occluded {
            if self.bounds.intersects(v) {
                if sensor.occludes(&self.bounds) {
                    self.occluded = true;
                } else {
                    if self.max_depth == self.level {
                        self.item_mask |= 1 << id;

                    } else {
                        if self.quads.is_none() {
                            self.quads = Some(Box::new(self.subdivide()));
                        }
                        let qs = self.quads.as_mut().unwrap();
                        qs[0].insert_internal(id, sensor, v);
                        qs[1].insert_internal(id, sensor, v);
                        qs[2].insert_internal(id, sensor, v);
                        qs[3].insert_internal(id, sensor, v);

                        self.occluded = qs[0].occluded
                            && qs[1].occluded
                            && qs[2].occluded
                            && qs[3].occluded;
                    }
                }

                if self.occluded {
                    self.quads = None;
                    self.item_mask = 0;
                }
            }
        }
    }

    fn visit<F,S>(&self, state: &mut S, visitor: &F) -> bool
        where F: Fn(&mut S, &QuadTree) -> bool
    {
        if !self.occluded {
            if let Some(qs) = &self.quads {
                if qs[0].visit(state, visitor) ||
                    qs[1].visit(state, visitor) ||
                    qs[2].visit(state, visitor) ||
                    qs[3].visit(state, visitor) {
                    return true;
                }
            } else if visitor(state, &self) {
                return true;
            }
        }
        return false;
    }

    fn subdivide(&self) -> [QuadTree; 4] {
        let div = self.bounds.subdivide();
        let next_level = self.level + 1;
        [
            QuadTree::new_quad(div[0].clone(), self.max_depth, next_level),
            QuadTree::new_quad(div[1].clone(), self.max_depth, next_level),
            QuadTree::new_quad(div[2].clone(), self.max_depth, next_level),
            QuadTree::new_quad(div[3].clone(), self.max_depth, next_level)
        ]
    }
}

#[derive(Debug, Clone)]
struct Sensor {
    pos: Point,
    beacon_pos: Point,
    distance: i32
}

impl Sensor {
    fn in_range(&self, p: &Point) -> bool {
        self.distance >= self.pos.distance(&p)
    }

    fn is_beacon(&self, p: &Point) -> bool {
        self.beacon_pos.x == p.x && self.beacon_pos.y == p.y
    }

    fn rect(&self) -> Rect {
        Rect {
            p0: Point {
                x: self.pos.x - self.distance,
                y: self.pos.y - self.distance
            },
            p1: Point {
                x: self.pos.x + self.distance,
                y: self.pos.y + self.distance
            }
        }
    }

    fn occludes(&self, v: &Rect) -> bool {
        self.pos.distance(&v.p0) <= self.distance
            && self.pos.distance(&v.p1) <= self.distance
            && self.pos.distance(&Point { x: v.p1.x, y: v.p0.y }) <= self.distance
            && self.pos.distance(&Point { x: v.p0.x, y: v.p1.y }) <= self.distance
    }
}

impl FromStr for Sensor {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts = s.split(' ').map(|s| s.to_string()).collect::<Vec<String>>();
        let x = parts[2].split_once('=').unwrap().1;
        let sx = x[0..x.len()-1].parse::<i32>().unwrap();
        let x = parts[3].split_once('=').unwrap().1;
        let sy = x[0..x.len()-1].parse::<i32>().unwrap();
        let x = parts[8].split_once('=').unwrap().1;
        let bx = x[0..x.len()-1].parse::<i32>().unwrap();
        let by = parts[9].split_once('=').unwrap().1.parse::<i32>().unwrap();
        let distance = Point {x: sx, y: sy}.distance(&Point {x: bx, y: by});

        Ok(Self {
            pos: Point {x: sx, y: sy},
            beacon_pos: Point {x: bx, y: by},
            distance
        })
    }
}


pub fn p1<T:BufRead>(inputs: T) {
    let sensors = parse(inputs);
    let mut max_x = 0;
    let mut min_x = i32::MAX;
    for x in &sensors {
        min_x = min_x.min(x.pos.x - x.distance);
        max_x = max_x.max(x.pos.x + x.distance);
    }

    println!("{}, {}, {}", min_x, max_x, max_x - min_x);
    let mut slot_count = 0;
    let mut in_range_count = 0;
    for x in min_x..=max_x {
        slot_count += 1;
        let t = Point {x, y: 2_000_000};
        let mut beacon = false;
        for s in &sensors {
            if s.is_beacon(&t) {
                beacon = true;
                break;
            }
        }
        if !beacon {
            for s in &sensors {
                if s.in_range(&t) {
                    in_range_count += 1;
                    break;
                }
            }
        } else {
            println!("BEACON: {:?}", t)
        }
    }
    println!("{} {}", in_range_count, slot_count);
}

fn parse<T: BufRead>(inputs: T) -> Vec<Sensor> {
    let mut sensors = inputs.lines().into_iter()
        .map(|s| s.unwrap().parse::<Sensor>().unwrap())
        .collect::<Vec<Sensor>>();
    sensors.sort_by_key(|s| -s.distance);
    return sensors;
}

pub fn p2<T:BufRead>(inputs: T) {
    let start = SystemTime::now();
    let sensors = parse(inputs);
    let depth = 13;
    let max_width = 4_000_000;
    let mut qt = QuadTree::new(
        Rect::new( Point {x: 0, y: 0}, Point {x: max_width, y: max_width }), depth);
    for (i, s) in sensors.iter().enumerate() {
        qt.insert(i, s);
    }

    let mut counter = 0;
    qt.visit(&mut counter, &|s, q| {
        *s += 1;
        if !q.occluded && q.item_mask > 0 {
            let mut p = Point {x:0, y:0};
            for y in q.bounds.p0.y..=q.bounds.p1.y {
                p.y = y;
                for x in q.bounds.p0.x..=q.bounds.p1.x {
                    p.x = x;
                    let mut in_range = false;
                    for i in 0..32 {
                        if (1 << i) & q.item_mask != 0 {
                            if let Some(s) = sensors.get(i as usize) {
                                if s.in_range(&p) {
                                    in_range = true;
                                    break;
                                }
                            }
                        }
                    }
                    if !in_range {
                        print!("{} ", 4_000_000u64 * (x as u64) + (y as u64));
                        return true;
                    }
                }
            }
        }
        return false;
    });
    let num_quads_width = 2u32.pow(depth as u32);
    let num_quads = num_quads_width * num_quads_width;
    println!("(hit {} of {} {}% depth={} qsize={} in {:?})", counter, num_quads, ((counter as f32 / num_quads as f32) * 100f32),
             depth, max_width as u32 / num_quads_width, start.elapsed().unwrap());
}
