use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;
use std::io::{BufRead};

#[derive(Debug, Clone, Default)]
struct Path {
    valves_opened: HashSet<String>,
    pressure_released: u32
}

impl Path {
    fn new() -> Self {
        Self {
            valves_opened: Default::default(),
            pressure_released: 0
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Valve {
    name: String,
    flow_rate: u32,
    links: Vec<String>,
}

impl Valve {
    fn new(name: String, flow_rate: u32) -> Self {
        Self {
            name,
            flow_rate,
            links: vec![],
        }
    }
}

#[derive(Debug, Clone, Default, Eq, PartialEq, Hash)]
struct PathRate {
    name: String,
    rate: u32,
}

impl PathRate {
    fn new(name: String, flow_rate: u32) -> Self {
        return Self {
            name,
            rate: flow_rate
        }
    }
}


#[derive(Debug, Clone, Default, Eq, PartialEq, Hash)]
struct TeamPathRate {
    name0: String,
    name1: String,
    rate: u32,
}

impl TeamPathRate {
    fn new(name0: String, name1: String, flow_rate: u32) -> Self {
        return Self {
            name0,
            name1,
            rate: flow_rate
        }
    }
}

fn parse<T: BufRead>(inputs: T) -> HashMap<String, Valve> {
    let mut map = HashMap::new();

    for x in inputs.lines() {
        let s = x.unwrap().split(' ').map(|s| s.to_string()).collect::<Vec<String>>();
        let name = s.get(1).unwrap();
        let a = s.get(4).unwrap().split_once('=').unwrap().1;
        let flow_rate = a[0..a.len()-1].parse::<u32>().unwrap();
        map.insert(name.clone(), Valve::new(name.clone(), flow_rate));

        for x in s[9..].iter() {
            map.get_mut(name).unwrap().links.push(x[0..2].to_string());
        }
        map.get_mut(name).unwrap().links.push(name.clone());
    }

    return map;
}

pub fn p1<T:BufRead>(inputs: T) {
    let valves = parse(inputs);
    let mut paths = HashMap::new();
    paths.insert(PathRate::new("AA".to_string(), 0), Path::new());
    for _i in 0..30 {
        let mut new_paths: HashMap<PathRate, Path> = HashMap::new();
        let mut update_new_paths = (|path_rate: PathRate, path: Path| {
            match new_paths.entry(path_rate) {
                Entry::Occupied(mut e) => {
                    if e.get().pressure_released < path.pressure_released {
                        e.insert(path);
                    }
                }
                Entry::Vacant(e) => {
                    e.insert(path);
                }
            }
        });

        for (path_rate, path) in paths {
            let pressure_released = path.pressure_released + path_rate.rate;

            // links
            for x in &valves.get(&path_rate.name).unwrap().links {
                let mut path = path.clone();
                path.pressure_released = pressure_released;
                update_new_paths(PathRate::new(x.to_string(), path_rate.rate), path);
            }

            // open
            if !path.valves_opened.contains(&path_rate.name) {
                let mut path = path.clone();
                path.valves_opened.insert(path_rate.name.to_string());
                path.pressure_released = pressure_released;
                let new_path_rate = path_rate.rate + valves.get(&path_rate.name).unwrap().flow_rate;
                update_new_paths(PathRate::new(path_rate.name.to_string(), new_path_rate), path);
            }
        }

        paths = new_paths;
    }

    println!("{:?}", paths.values().map(|p| p.pressure_released).max());
}

pub fn p2<T:BufRead>(inputs: T) {
    let valves = parse(inputs);
    let mut paths = HashMap::new();
    paths.insert(TeamPathRate::new("AA".to_string(), "AA".to_string(), 0), Path::new());
    for i in 0..26 {
        let mut new_paths: HashMap<TeamPathRate, Path> = HashMap::new();
        let mut update_new_paths = (|path_rate: TeamPathRate, path: Path| {
            match new_paths.entry(path_rate) {
                Entry::Occupied(mut e) => {
                    if e.get().pressure_released < path.pressure_released {
                        e.insert(path);
                    }
                }
                Entry::Vacant(e) => {
                    e.insert(path);
                }
            }
        });

        for (path_rate, path) in paths {
            let pressure_released = path.pressure_released + path_rate.rate;

            // links
            for x0 in &valves.get(&path_rate.name0).unwrap().links {
                for x1 in &valves.get(&path_rate.name1).unwrap().links {
                    let mut path = path.clone();
                    let mut new_path_rate = path_rate.rate;
                    path.pressure_released = pressure_released;

                    if x0 == &path_rate.name0 {
                        if path.valves_opened.contains(&path_rate.name0) {
                            continue;
                        }
                        path.valves_opened.insert(path_rate.name0.to_string());
                        new_path_rate += valves.get(&path_rate.name0).unwrap().flow_rate;
                    }

                    if x1 == &path_rate.name1 {
                        if path.valves_opened.contains(&path_rate.name1) {
                            continue;
                        }
                        path.valves_opened.insert(path_rate.name1.to_string());
                        new_path_rate += valves.get(&path_rate.name1).unwrap().flow_rate;
                    }

                    update_new_paths(TeamPathRate::new(x0.to_string(), x1.to_string(), new_path_rate), path);
                }
            }
        }

        paths = new_paths;
        println!("{} {:?}", i, paths.values().map(|p| p.pressure_released).max());
    }

    println!("{:?}", paths.values().map(|p| p.pressure_released).max());
}