#![cfg_attr(
target_os = "cuda",
no_std,
feature(register_attr),
register_attr(nvvm_internal)
)]

#![allow(improper_ctypes_definitions)]

extern crate alloc;

pub mod add;
pub mod day1;