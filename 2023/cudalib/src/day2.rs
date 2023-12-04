use core::str::FromStr;

use cuda_std::{kernel, thread};
use cuda_std::thread::sync_threads;
use cust_core::DeviceCopy;

/// [`from_str`]: super::FromStr::from_str
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError;

#[repr(C)]
#[derive(Clone, Copy, DeviceCopy)]
pub struct Rgb {
    pub r: u32,
    pub g: u32,
    pub b: u32
}

impl Rgb {
    pub fn max(&mut self, set: &Self) {
        self.r = set.r.max(self.r);
        self.g = set.g.max(self.g);
        self.b = set.b.max(self.b);
    }

    pub fn power(&self) -> u32 {
        return if self.r == 0 && self.g == 0 && self.b == 0 {
            0u32
        } else {
            self.r.max(1) * self.g.max(1) * self.b.max(1)
        };
    }

    pub fn possible(&self, max: &Self) -> bool {
        self.r <= max.r && self.g <= max.g && self.b <= max.b
    }
}

impl From<(u32, u32, u32)> for Rgb {
    fn from(rgb: (u32, u32, u32)) -> Self {
        Rgb {
            r: rgb.0,
            g: rgb.1,
            b: rgb.2
        }
    }
}

impl FromStr for Rgb {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut rgb = Self {r: 0, g: 0, b: 0};
        for c in s.split(",") {
            let split = c.trim().split_once(" ").unwrap();
            let count = split.0.parse::<u32>().unwrap();
            match split.1 {
                "red" => rgb.r = count,
                "green" => rgb.g = count,
                "blue" => rgb.b = count,
                _ => return Err(ParseError)
            }
        }
        Ok(rgb)
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Game<'a> {
    pub sets: &'a [Rgb],
}

unsafe impl DeviceCopy for Game<'_> {}

#[kernel]
pub unsafe fn day2(games: &[Game], max: Rgb, possible_ids: *mut u32) {
    let idx = thread::index_1d() as usize;
    let len = games.len();

    if idx < len {
        let count_elem = possible_ids.add(idx).as_mut().unwrap();
        let power_elem = possible_ids.add(idx + len).as_mut().unwrap();
        let mut possible = true;
        let mut game_max = Rgb {r: 0, g: 0, b: 0};
        let game = &games[idx];
        for set in game.sets {
            possible &= set.possible(&max);
            game_max.max(set);
        }
        *count_elem = if possible {
            (idx + 1) as u32
        } else {
            0u32
        };
        *power_elem = game_max.power();
        sync_threads();
        if idx == 0 {
            let mut count_result = 0u32;
            let mut power_result = 0u32;
            for i in 0..len {
                count_result += possible_ids.add(i).as_ref().unwrap_or(&0u32);
                power_result += possible_ids.add(i + len).as_ref().unwrap_or(&0u32);
            }
            *count_elem = count_result;
            *power_elem = power_result;
        }
        sync_threads();
    }
}