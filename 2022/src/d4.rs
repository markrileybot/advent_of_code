use std::io::BufRead;

struct Assignment {
    min: u32,
    max: u32,
}

impl Assignment {
    fn contains_section(&self, section: u32) -> bool {
        self.min <= section && self.max >= section
    }
    fn intersects(&self, other: &Assignment) -> bool {
        self.contains_section(other.min) || self.contains_section(other.max)
    }
    fn contains(&self, other: &Assignment) -> bool {
        self.contains_section(other.min) && self.contains_section(other.max)
    }
}

impl From<&str> for Assignment {
    fn from(s: &str) -> Self {
        if let Some((ss, es)) = s.split_once('-') {
            if let Ok(s) = ss.parse::<u32>() {
                if let Ok(e) = es.parse::<u32>() {
                    return Self {min: s, max: e}
                }
            }
        }
        return Self {min: 0, max: 0};
    }
}

fn parse<T:BufRead>(inputs: T) -> Vec<(Assignment, Assignment)> {
    inputs.lines()
        .map(|f| f.unwrap())
        .collect::<Vec<String>>().iter()
        .map(|s| s.split_once(',').unwrap())
        .map(|s| (Assignment::from(s.0), Assignment::from(s.1)))
        .collect()
}

pub fn p1<T:BufRead>(inputs: T) {
    let total = parse(inputs).iter()
        .filter(|a| a.0.contains(&a.1) || a.1.contains(&a.0))
        .count();
    println!("{}", total);
}

pub fn p2<T:BufRead>(inputs: T) {
    let total = parse(inputs).iter()
        .filter(|a| a.0.intersects(&a.1) || a.1.intersects(&a.0))
        .count();
    println!("{}", total);
}