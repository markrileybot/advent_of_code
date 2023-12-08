use cuda_std::{kernel, thread, println};
use cuda_std::thread::sync_threads;
use cust_core::DeviceCopy;


pub type Id = u32;

#[repr(C)]
#[derive(Clone, Copy, DeviceCopy, PartialOrd, PartialEq)]
pub enum Type {
    Seed,
    Soil,
    Fertilizer,
    Water,
    Light,
    Temp,
    Humidity,
    Location
}

impl Type {
    pub fn dest(&self) -> Option<Self> {
        match self {
            Type::Seed => Some(Type::Soil),
            Type::Soil => Some(Type::Fertilizer),
            Type::Fertilizer => Some(Type::Water),
            Type::Water => Some(Type::Light),
            Type::Light => Some(Type::Temp),
            Type::Temp => Some(Type::Humidity),
            Type::Humidity => Some(Type::Location),
            Type::Location => None
        }
    }

    pub fn ord(&self) -> usize {
        match self {
            Type::Seed => 0,
            Type::Soil => 1,
            Type::Fertilizer => 2,
            Type::Water => 3,
            Type::Light => 4,
            Type::Temp => 5,
            Type::Humidity => 6,
            Type::Location => 7
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, DeviceCopy)]
pub struct Mapping {
    pub source_type: Type,
    pub dest_type: Type,
    pub source: (Id,  Id),
    pub dest: Id,
}

impl Mapping {
    pub fn map(&self, num: Id) -> Option<Id> {
        if num >= self.source.0 && num < self.source.1  {
            Some(self.dest + (num - self.source.0))
        } else {
            None
        }
    }
}

impl FromIterator<Id> for Mapping {
    fn from_iter<T: IntoIterator<Item=Id>>(iter: T) -> Self {
        let mut x = iter.into_iter();
        let d = x.next().unwrap() as Id;
        let s = x.next().unwrap() as Id;
        let l = x.next().unwrap() as Id;
        Mapping {
            source_type: Type::Seed,
            dest_type: Type::Seed,
            source: (s, s + l),
            dest: d,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, DeviceCopy)]
pub struct Mappings {
    pub mappings: [[Mapping; 50];8],
}

impl Mappings {
    pub fn new() -> Self {
        Self {
            mappings: [[Mapping {
                source_type: Type::Seed,
                dest_type: Type::Seed,
                source: (0, 0),
                dest: 0,
            }; 50];8],
        }
    }

    pub fn map(&self, id: Id, t: Type) -> Id {
        // have we reached the end?
        if t == Type::Location {
            return id;
        }

        // get our next dest
        let dest_type = t.dest().unwrap();
        let mut dest = id;
        for m in &self.mappings[t.ord()] {
            if let Some(d) = m.map(id) {
                dest = d;
                break;
            }
        }
        return self.map(dest, dest_type);
    }
}

#[kernel]
pub unsafe fn day5(mappings: &Mappings, seeds: *mut Id, seed_count: usize, results: *mut Id) {
    let idx = thread::index_1d() as usize;
    if idx < seed_count {
        let seed = seeds.add(idx).as_mut().unwrap();
        *seed = mappings.map(*seed, Type::Seed);

        sync_threads();
        if idx == 0 {
            let min = results.as_mut().unwrap();
            *min = Id::max_value();
            for i in 0..seed_count {
                *min = *min.min(seeds.add(i).as_mut().unwrap());
            }
        }
    }
}