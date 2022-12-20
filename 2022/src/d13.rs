use std::cell::RefCell;
use std::cmp::Ordering;
use std::cmp::Ordering::Equal;
use std::fmt::{Display, Formatter};
use std::io::BufRead;
use std::rc::Rc;
use std::str::FromStr;

#[derive(Debug, Clone)]
struct Value {
    integer: i32,
    list: Vec<Rc<RefCell<Value>>>,
    parent: Option<Rc<RefCell<Value>>>
}

impl Value {
    fn ni(integer: i32) -> Self {
        return Self {
            integer,
            list: Vec::new(),
            parent: None
        }
    }
    fn nl(list: Vec<Rc<RefCell<Value>>>) -> Self {
        return Self {
            integer: -1,
            list,
            parent: None
        }
    }
    fn is_scalar(&self) -> bool {
        self.integer > -1 && self.list.is_empty()
    }
    fn restructure(&self, other: &Rc<RefCell<Self>>) {
        if !self.is_scalar() {
            if other.borrow().is_scalar() {
                let i = other.borrow().integer;
                other.borrow_mut().list = vec![Rc::new(RefCell::new(Value::ni(i)))];
                other.borrow_mut().integer = -1;
            }

            for (i, v) in self.list.iter().enumerate() {
                if let Some(other2) = other.borrow().list.get(i) {
                    v.borrow().restructure(other2);
                } else {
                    break;
                }
            }
        }
    }
    fn compare(&self, other: &Rc<RefCell<Value>>) -> Ordering {
        if self.is_scalar() {
            if other.borrow().is_scalar() {
                self.integer.cmp(&other.borrow().integer)
            } else {
                Equal
            }
        } else {
            let l = &self.list;
            let l2 = &other.borrow().list;
            for i in 0..l2.len().min(l.len()) {
                let v = l.get(i).unwrap();
                let v2 = l2.get(i).unwrap();
                let ordering = v.borrow().compare(v2);
                if ordering != Equal {
                    return ordering;
                }
            }
            return l.len().cmp(&l2.len());
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.integer > -1 {
            write!(f, "{}", self.integer)?;
        } else {
            write!(f, "[")?;
            for (i, x) in self.list.iter().enumerate() {
                if i > 0 {
                    write!(f, ",")?;
                }
                write!(f, "{}", x.borrow())?;
            }
            write!(f, "]")?;
        }
        Ok(())
    }
}

struct Packet {
    values: Rc<RefCell<Value>>,
    original: String
}

impl Packet {
    fn restructure(&self, other: &Self) {
        self.values.borrow().restructure(&other.values);
    }
}

impl PartialEq<Self> for Packet {
    fn eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl PartialOrd for Packet {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.values.borrow().compare(&other.values))
    }
}

impl Eq for Packet {}

impl Ord for Packet {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl FromStr for Packet {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut values = Rc::new(RefCell::new(Value::nl(Vec::new())));
        let mut start = true;
        let mut tmp = String::new();

        for x in s.chars() {
            match x {
                '[' => {
                    if !start {
                        let parent = values.clone();
                        values = Rc::new(RefCell::new(Value::nl(Vec::new())));
                        values.borrow_mut().parent = Some(parent.clone());
                        parent.borrow_mut().list.push(values.clone());
                    }
                    start = false;
                },
                ']' => {
                    if !tmp.is_empty() {
                        values.borrow_mut().list.push(Rc::new(RefCell::new(Value::ni(tmp.parse().unwrap()))))
                    }
                    tmp.clear();
                    let p = values.borrow().parent.clone();
                    if let Some(p) = p {
                        values = p;
                    }
                },
                ',' => {
                    if !tmp.is_empty() {
                        values.borrow_mut().list.push(Rc::new(RefCell::new(Value::ni(tmp.parse().unwrap()))))
                    }
                    tmp.clear();
                },
                v => {
                    tmp.push(v);
                }
            }
        }
        let packet = Packet {values, original: s.to_string()};
        return Ok(packet)
    }
}

impl Display for Packet {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.values.borrow())?;
        Ok(())
    }
}

pub fn p1<T:BufRead>(inputs: T) {
    let mut total = 0;
    for (i, packets) in inputs.lines().map(|f| f.unwrap()).collect::<Vec<String>>().chunks(3).enumerate() {
        let index = i + 1;
        let packet0 = packets[0].parse::<Packet>().unwrap();
        let packet1 = packets[1].parse::<Packet>().unwrap();
        packet0.restructure(&packet1);
        packet1.restructure(&packet0);

        if packet0 < packet1 {
            println!("{} RIGHT \n{} \n{} \n{} \n{}", index, packets[0], packets[1], packet0, packet1);
            total += index;
        } else if packet0 > packet1 {
            println!("{} WRONG \n{} \n{} \n{}  \n{}", index, packets[0], packets[1], packet0, packet1);
        } else {
            println!("{} BAD!! {} = {} ({} = {})", index, packets[0], packets[1], packet0, packet1);
        }
    }
    println!("{}", total);
}

pub fn p2<T:BufRead>(inputs: T) {
    let mut packets = inputs.lines()
        .map(|f| f.unwrap())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<Packet>().unwrap())
        .collect::<Vec<Packet>>();

    for x in &packets {
        for y in &packets {
            x.restructure(y)
        }
    }

    packets.sort();
    let mut total = 1;
    for (i, x) in packets.iter().enumerate() {
        println!("{}", x.original);
        if x.original == "[[2]]" || x.original == "[[6]]" {
            total *= i + 1;
        }
    }
    println!("{}", total);
}