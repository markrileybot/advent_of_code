use core::str::FromStr;

use cuda_std::{kernel, shared_array, thread};
use cuda_std::thread::sync_threads;
use cust_core::DeviceCopy;

use crate::day2::ParseError;

pub const WIN_LEN: usize = 10;
pub const MY_LEN: usize = 25;

#[repr(C)]
#[derive(Clone, Copy, DeviceCopy)]
pub struct Number {
    pub value: u32,
    pub winner: bool,
}

#[repr(C)]
#[derive(Clone, Copy, DeviceCopy)]
pub struct Card {
    pub winning_set: [u32; WIN_LEN],
    pub my_set: [Number; MY_LEN],
    pub points: u32,
    pub winners: u32,
    pub count: u32
}

impl Card {
    pub fn mark_winner(&mut self, pos: usize) {
        let my_num = &mut self.my_set[pos];
        for x in &self.winning_set {
            if x == &my_num.value {
                my_num.winner = true;
                break;
            }
        }
    }

    pub fn count_winners(&mut self) {
        for n in &self.my_set {
            if n.winner {
                self.winners += 1;
            }
        }
        self.points = 1u32.checked_shl(self.winners - 1).unwrap_or(0);
    }
}

impl FromStr for Card {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut card = Card { winning_set: [0u32; WIN_LEN], my_set: [Number { value: 0, winner: false }; MY_LEN],
            points: 0, winners: 0, count: 1 };
        let (w, m) = s.split_once(":").unwrap().1.trim()
            .split_once("|").unwrap();
        let mut i = 0;
        for s in w.split(" ") {
            let s = s.trim();
            if !s.is_empty() {
                card.winning_set[i] = s.parse::<u32>().unwrap();
                i += 1;
            }
        }
        let mut i = 0;
        for s in m.split(" ") {
            let s = s.trim();
            if !s.is_empty() {
                card.my_set[i] = Number { value: s.parse::<u32>().unwrap(), winner: false };
                i += 1;
            }
        }
        Ok(card)
    }
}

#[kernel]
pub unsafe fn day4(cards: *mut Card, num_cards: usize, results: *mut u32) {
    let idx = thread::index_1d() as usize;
    let len = num_cards * MY_LEN;

    if idx < len {
        let y = idx / MY_LEN;
        let x = idx - (y * MY_LEN);
        let card = cards.add(y).as_mut().unwrap();
        card.mark_winner(x);
        sync_threads();

        if idx % MY_LEN == 0 {
            cards.add(idx / MY_LEN).as_mut().unwrap().count_winners();
        }
        sync_threads();

        if idx == 0 {
            let result1 = results.add(idx).as_mut().unwrap();
            let result2 = results.add(idx+1).as_mut().unwrap();
            *result1 = (0..num_cards)
                .map(|i| cards.add(i).as_ref().unwrap().points)
                .sum();

            // I can figure out how to multithread this in CUDA yet
            for i in 0..num_cards {
                let card = cards.add(i).as_ref().unwrap();
                for _ in 0..card.count {
                    for ii in 0..card.winners as usize {
                        let card2 = cards.add(i+ii+1).as_mut().unwrap();
                        card2.count += 1;
                    }
                }
            }
            *result2 = (0..num_cards)
                .map(|i| cards.add(i).as_ref().unwrap().count)
                .sum();
        }
    }
}