use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("cudalib")
        .copy_to("target/aoc-2023-cudalib.ptx")
        .build()
        .unwrap();
}