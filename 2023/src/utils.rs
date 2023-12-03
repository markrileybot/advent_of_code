use cust::context::Context;
use cust::error::CudaResult;
use cust::function::Function;
use cust::module::Module;
use cust::prelude::{Stream, StreamFlags};

static PTX: &str = include_str!("../target/aoc-2023-cudalib.ptx");

pub struct Ctx {
    context: Context,
    module: Module,
    pub stream: Stream,
}

impl Ctx {
    pub fn new() -> CudaResult<Self> {
        Ok(Self {
            context: cust::quick_init()?,
            stream: Stream::new(StreamFlags::NON_BLOCKING, None)?,
            module: Module::from_ptx(PTX, &[])?
        })
    }

    pub fn load_kernel(&self, name: &str) -> CudaResult<(Function, u32)> {
        // retrieve the add kernel from the module so we can calculate the right launch config.
        let func = self.module.get_function(name)?;
        let (_, block_size) = func.suggested_launch_configuration(0, 0.into())?;
        Ok((func, block_size))
    }
}
//
// pub(crate) fn load_kernel(name: &str) -> CudaResult<(Module, Function<'_>, Stream, u32)> {
//
//     // initialize CUDA, this will pick the first available device and will
//     // make a CUDA context from it.
//     // We don't need the context for anything but it must be kept alive.
//     let _ctx = cust::quick_init()?;
//
//     // Make the CUDA module, modules just house the GPU code for the kernels we created.
//     // they can be made from PTX code, cubins, or fatbins.
//     let module = Module::from_ptx(PTX, &[])?;
//
//     // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
//     // GPU calls.
//     let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
//
//     // retrieve the add kernel from the module so we can calculate the right launch config.
//     let func = module.get_function(name)?;
//
//     // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
//     // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
//     // current CUDA device/architecture.
//     let (_, block_size) = func.suggested_launch_configuration(0, 0.into())?;
//     Ok((module, func, stream, block_size))
// }
