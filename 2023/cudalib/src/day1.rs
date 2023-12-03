use cuda_std::{kernel, thread, println};
use cuda_std::thread::sync_threads;

#[kernel]
pub unsafe fn day1(chunk_size: usize, len: usize, numbers: *mut u32) {
    let idx = thread::index_1d() as usize;
    let mut len = len;
    let input_offset = idx * chunk_size;

    while input_offset < len && len >= chunk_size {
        let mut result = 0u32;
        for i in 0..chunk_size {
            result += numbers.add(input_offset + i).as_ref().unwrap();
        }
        sync_threads();
        *numbers.add(idx) = result;
        len /= chunk_size;
    }
}