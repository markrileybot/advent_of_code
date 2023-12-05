use cuda_std::{kernel, thread, println};
use cuda_std::thread::sync_threads;
use cust_core::DeviceCopy;

#[repr(C)]
#[derive(Clone, Copy, DeviceCopy)]
pub struct Element {
    pub x: usize,
    pub y: usize,
    pub value: char
}

impl Element {
    pub fn is_digit(&self) -> bool {
        self.value.is_digit(10)
    }

    pub fn to_digit(&self) -> Option<u32> {
        self.value.to_digit(10)
    }

    pub fn is_symbol(&self) -> bool {
        !self.is_digit() && self.value != '.'
    }

    pub fn is_gear(&self) -> bool {
        self.value == '*'
    }

    pub fn get_position(&self, grid_width: usize) -> usize {
        self.get_position_in_row(grid_width, self.x)
    }

    pub fn get_position_in_row(&self, grid_width: usize, x: usize) -> usize {
        self.y * grid_width + x
    }

    pub fn get_position_of_el(grid_width: usize, x: usize, y: usize) -> usize {
        y * grid_width + x
    }

    pub fn to_gear_ratio(&self, grid: &[Element], grid_size: (usize, usize)) -> Option<u32> {
        if self.is_gear() {
            let mut adjacent_part_number_0 = None;
            let mut adjacent_part_number_1 = None;

            self.walk_adjacent(grid, grid_size, |e| {
                if let Some((pn, start)) = e.to_part_number(grid, grid_size) {
                    if let Some((_, start0)) = adjacent_part_number_0 {
                        if start != start0 {
                            if let Some((_, start1)) = adjacent_part_number_1 {
                                if start != start1 {
                                    adjacent_part_number_0 = None;
                                    adjacent_part_number_1 = None;
                                    return true;
                                }
                            } else {
                                adjacent_part_number_1 = Some((pn, start));
                            }
                        }
                    } else {
                        adjacent_part_number_0 = Some((pn, start));
                    }
                }
                return false;
            });

            if let (Some((pn0, _)), Some((pn1, _))) = (adjacent_part_number_0, adjacent_part_number_1) {
                return Some(pn0 * pn1);
            }
        }
        return None;
    }

    pub fn to_part_number(&self, grid: &[Element], grid_size: (usize, usize)) -> Option<(u32, usize)> {
        if self.is_part_number(grid, grid_size) {
            //find the part number start
            let mut start = self.get_position(grid_size.0);
            let mut x = self.x;
            while x > 0 {
                x -= 1;
                let maybe_start = self.get_position_in_row(grid_size.0, x);
                if (&grid[maybe_start]).is_digit() {
                    start = maybe_start;
                } else {
                    break;
                }
            }

            // now add'em urp
            let mut res = 0;
            for i in start..(start + grid_size.0) {
                if let Some(d) = (&grid[i]).to_digit() {
                    res = res * 10 + d;
                } else {
                    break;
                }
            }

            Some((res, start))
        } else {
            None
        }
    }

    pub fn is_part_number(&self, grid: &[Element], grid_size: (usize, usize)) -> bool {
        self.is_digit() && self.walk_adjacent(grid, grid_size, |e| e.is_symbol())
    }

    pub fn walk_adjacent<F>(&self, grid: &[Element], grid_size: (usize, usize), mut call: F) -> bool
        where F: FnMut(&Element) -> bool {

        let grid_width = grid_size.0;
        let grid_height = grid_size.1;
        let position = self.get_position(grid_width);
        let last_row = grid_height - 1;
        let last_col = grid_width - 1;
        let above = if self.y > 0 {
            position - grid_width
        } else {
            0
        };
        let below = if self.y < last_row {
            position + grid_width
        } else {
            0
        };

        // left col check
        if self.x > 0 {
            if call(&grid[position - 1]) {
                return true;
            }
            if self.y > 0 && call(&grid[above - 1]) {
                return true;
            }
            if self.y < last_row && call(&grid[below - 1]) {
                return true;
            }
        }

        // my col check
        if self.y > 0 && call(&grid[above]) {
            return true;
        }
        if self.y < last_row && call(&grid[below]) {
            return true;
        }

        // right col check
        if self.x < last_col {
            if call(&grid[position + 1]) {
                return true;
            }
            if self.y > 0 && call(&grid[above + 1]) {
                return true;
            }
            if self.y < last_row && call(&grid[below + 1]) {
                return true;
            }
        }

        return false;
    }
}

#[kernel]
pub unsafe fn day3(grid: &[Element], grid_width: usize, grid_height: usize, digits: *mut i32) {
    let mut position = thread::index() as usize;
    let len = grid.len();
    let grid_size = (grid_width, grid_height);

    if position < len {
        let el = grid[position].clone();
        let d = digits.add(position).as_mut().unwrap();
        let d1 = digits.add(position + len).as_mut().unwrap();

        // make part numbers
        *d = el.to_part_number(grid, grid_size)
            .map(|(pn, _) | pn as i32)
            .unwrap_or(-1)
        ;

        // make gear ratio
        *d1 = el.to_gear_ratio(grid, grid_size)
            .map(|gr | gr as i32)
            .unwrap_or(-1)
        ;

        sync_threads();

        // remove dups
        if position % grid_size.0 == 0 {
            let mut last_result = -1;
            for i in position..(position + grid_size.0) {
                let v = digits.add(i).as_mut().unwrap();
                if v != &-1 && last_result != -1 {
                    last_result = *v;
                    *v = -1;
                } else {
                    last_result = *v;
                }
            }
        }
        sync_threads();

        // sum
        if position == 0 {
            let mut sum0 = 0;
            for i in 0..len {
                let v = digits.add(i).as_ref().unwrap();
                if v != &-1 {
                    sum0 += v;
                }
            }
            *d = sum0;

            let mut sum1 = 0;
            let d1 = digits.add(1).as_mut().unwrap();
            for i in len..len  * 2 {
                let v = digits.add(i).as_ref().unwrap();
                if v != &-1 {
                    sum1 += v;
                }
            }
            *d1 = sum1;
        }
    }
}