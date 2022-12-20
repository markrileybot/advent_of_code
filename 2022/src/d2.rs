use std::io::BufRead;

#[derive(Clone, Debug)]
enum RPS {
    Rock,
    Paper,
    Scissors
}

impl RPS {
    fn score(&self, other: &RPS) -> u32 {
        match self {
            Self::Rock => 1 + match other {
                Self::Rock => 3,
                Self::Paper => 0,
                Self::Scissors => 6
            },
            Self::Paper => 2 + match other {
                Self::Rock => 6,
                Self::Paper => 3,
                Self::Scissors => 0
            },
            Self::Scissors => 3 + match other {
                Self::Rock => 0,
                Self::Paper => 6,
                Self::Scissors => 3
            },
        }
    }

    fn beats(&self) -> RPS {
        match self {
            Self::Rock => Self::Scissors,
            Self::Paper => Self::Rock,
            Self::Scissors => Self::Paper
        }
    }

    fn loses(&self) -> RPS {
        match self {
            Self::Rock => Self::Paper,
            Self::Paper => Self::Scissors,
            Self::Scissors => Self::Rock
        }
    }
}

impl From<&str> for RPS {
    fn from(s: &str) -> Self {
        return match s {
            "B" => Self::Paper,
            "Y" => Self::Paper,
            "C" => Self::Scissors,
            "Z" => Self::Scissors,
            &_ => Self::Rock
        }
    }
}

#[derive(Clone, Debug)]
enum WLD {
    Win,
    Lose,
    Draw
}

impl WLD {
    fn chose_my_move(&self, other: &RPS) -> RPS {
        match self {
            Self::Win => other.loses(),
            Self::Lose => other.beats(),
            Self::Draw => other.clone(),
        }
    }
}

impl From<&str> for WLD {
    fn from(s: &str) -> Self {
        return match s {
            "Y" => Self::Draw,
            "Z" => Self::Win,
            &_ => Self::Lose
        }
    }
}

pub fn p1<T:BufRead>(inputs: T) {
    let mut score = 0;
    for x in inputs.lines().map(|f| f.unwrap()) {
        let moves = x.split(" ").into_iter()
            .map(|s| RPS::from(s))
            .collect::<Vec<RPS>>();
        if moves.len() == 2 {
            score += moves.get(1).unwrap().score(moves.get(0).unwrap());
        }
    }
    println!("{}", score);
}

pub fn p2<T:BufRead>(inputs: T) {
    let mut score = 0;
    for x in inputs.lines().map(|f| f.unwrap()) {
        if let Some((op, oc)) = x.split_once(" ") {
            let op_move: RPS = op.into();
            let outcome: WLD = oc.into();
            score += outcome.chose_my_move(&op_move).score(&op_move);
        }
    }
    println!("{}", score);
}