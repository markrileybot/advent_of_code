use std::cell::RefCell;
use std::io::BufRead;
use std::rc::Rc;

#[derive(Debug)]
struct File {
    parent: Option<Rc<RefCell<File>>>,
    children: Vec<Rc<RefCell<File>>>,
    size: u32,
    name: String
}

impl File {
    fn size(&self) -> u32 {
        let mut total = self.size;
        for x in &self.children {
            total += x.borrow().size();
        }
        return total;
    }

    fn visit(&self, visitor: &mut dyn FnMut(&Rc<RefCell<File>>)) {
        for x in &self.children {
            visitor(x);
        }
        for x in &self.children {
            x.borrow().visit(visitor);
        }
    }

    fn get_child(&self, name: &str) -> Option<Rc<RefCell<File>>> {
        for x in &self.children {
            if x.borrow().name == name {
                return Some(x.clone());
            }
        }
        return None;
    }

    fn parent(&self) -> Option<Rc<RefCell<File>>> {
        match &self.parent {
            None => None,
            Some(p) => Some(p.clone())
        }
    }
}

const CMD_CD: &'static str = "$ cd";
const CMD_LS: &'static str = "$ ls";
const DIR_ROOT: &'static str = "/";
const DIR_UP: &'static str = "..";

fn parse<T:BufRead>(inputs: T) -> Rc<RefCell<File>> {
    let root = Rc::new(RefCell::new(File {parent: None, children: Vec::new(), size: 0, name: "/".to_string()}));
    let mut listing = false;
    let mut dir = root.clone();
    for x in inputs.lines().map(|f| f.unwrap()) {
        if x.starts_with(CMD_CD) {
            listing = false;
            let cd_dir = x[CMD_CD.len()..].trim();
            if cd_dir == DIR_ROOT {
                dir = root.clone();
            } else if cd_dir == DIR_UP {
                let parent = dir.borrow().parent().unwrap_or(root.clone());
                dir = parent;
            } else {
                let child = dir.borrow().get_child(cd_dir);
                match child {
                    None => panic!("Failed to find '{}' in '{}'", cd_dir, dir.borrow().name),
                    Some(c) => dir = c,
                }
            }
        } else if x == CMD_LS {
            listing = true;
        } else if listing {
            if let Some((size, file_name)) = x.split_once(" ") {
                if dir.borrow().get_child(file_name).is_none() {
                    let size = size.parse::<u32>().unwrap_or(0);
                    dir.borrow_mut().children.push(Rc::new(RefCell::new(
                        File {
                            parent: Some(dir.clone()),
                            children: Vec::new(),
                            size,
                            name: file_name.to_string()
                        })));
                }
            }
        }
    }
    return root;
}

pub fn p1<T:BufRead>(inputs: T) {
    let root = parse(inputs);
    let mut accum = Vec::new();
    root.borrow().visit(&mut |f| {
        let fb = f.borrow();
        if fb.size() < 100000 && fb.children.len() != 0 {
            println!("{} {}", fb.name, fb.size());
            accum.push(f.clone());
        }
    });

    let mut total = 0;
    for x in accum {
        total += x.borrow_mut().size();
    }
    println!("{}", total);
}

pub fn p2<T:BufRead>(inputs: T) {
    let root = parse(inputs);
    let space_used = root.borrow().size();
    let space_needed = 30000000 - (70000000 - space_used);
    let mut closest = u32::MAX;
    println!("{}", space_needed);
    root.borrow().visit(&mut |f: &Rc<RefCell<File>>| {
        let fb = f.borrow();
        let size = fb.size();
        if size > space_needed && fb.children.len() != 0 {
            if size - space_needed < closest {
                closest = size - space_needed;
                println!("{} {} {}", fb.name, size, closest);
            } else {
                println!("{} {}", fb.name, size);
            }
        }
    });
}