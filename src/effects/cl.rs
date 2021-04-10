use opencl3::context::Context;
use opencl3::platform::get_platforms;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device};
use std::sync::Once;
use std::ffi::CString;
use opencl3::kernel::{Kernel, ExecuteKernel};
use opencl3::svm::SvmVec;
use opencl3::command_queue::CommandQueue;
use std::time::Duration;

pub fn create_context() -> Context {
    let platform_ids = get_platforms().unwrap();
    assert!(0 < platform_ids.len(), "Could not find any OpenCL platforms.");
    let platform_id = platform_ids.first().unwrap();
    let device_ids = opencl3::device::get_device_ids(platform_id.id(), CL_DEVICE_TYPE_GPU).unwrap();
    assert!(0 < device_ids.len(), "Could not find any OpenCL devices.");
    let devices: Vec<Device> = device_ids.into_iter().map(|id| Device::new(id)).collect();
    let device = devices.into_iter().next().unwrap();

    let mut context = Context::from_device(device).unwrap();
    context.create_command_queues_with_properties(0, 0).unwrap();
    context
}

pub fn compile_kernels(context: &mut Context) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let mut timer = yatl::Timer::new();
        debug!("# Compiling kernels... ");
        timer.start().unwrap();
        let options = CString::new("").unwrap();
        context.build_program_from_source(&CString::new(include_str!("simple.cl")).unwrap(), &options).unwrap();
        context.build_program_from_source(&CString::new(include_str!("channel_swap.cl")).unwrap(), &options).unwrap();
        context.build_program_from_source(&CString::new(include_str!("color_intensity.cl")).unwrap(), &options).unwrap();
        let kernel_count = context.kernels().len();
        debugln!("[ {} OK ] {}", kernel_count, yatl::duration_to_human_string(&timer.lap().unwrap()));
    });
}

pub fn run_pixel_based_kernel(kernel: &Kernel, pixel_count: usize, input: &SvmVec<u8>, output: &mut SvmVec<u8>, queue: &CommandQueue) -> Duration {
    let mut timer = yatl::Timer::new();
    timer.start().unwrap();
    let kernel_event = ExecuteKernel::new(kernel)
        .set_arg_svm(input.as_ptr())
        .set_arg_svm(output.as_mut_ptr())
        .set_global_work_size(pixel_count)
        .enqueue_nd_range(&queue)
        .unwrap();

    kernel_event.wait().unwrap();
    timer.lap().unwrap()
}

pub fn run_pixel_based_kernel_1<T>(context: &Context, kernel: &Kernel, pixel_count: usize, input: &SvmVec<u8>, output: &mut SvmVec<u8>, queue: &CommandQueue, value: T) -> Duration where T: Clone {
    let svm_capability = context.get_svm_mem_capability();
    let mut value_svm = SvmVec::<T>::with_capacity(&context, svm_capability, 1);
    value_svm.push(value);
    let mut timer = yatl::Timer::new();
    timer.start().unwrap();
    let kernel_event = ExecuteKernel::new(kernel)
        .set_arg_svm(input.as_ptr())
        .set_arg_svm(output.as_mut_ptr())
        .set_arg_svm(value_svm.as_ptr())
        .set_global_work_size(pixel_count)
        .enqueue_nd_range(&queue)
        .unwrap();
    kernel_event.wait().unwrap();
    timer.lap().unwrap()
}

pub fn run_pixel_based_kernel_1v<T>(context: &Context, kernel: &Kernel, pixel_count: usize, input: &SvmVec<u8>, output: &mut SvmVec<u8>, queue: &CommandQueue, value: &Vec<T>) -> Duration where T: Clone {
    let svm_capability = context.get_svm_mem_capability();
    let mut value_svm = SvmVec::<T>::with_capacity(&context, svm_capability, value.len());
    value.iter().cloned().for_each(|v| value_svm.push(v));
    let mut timer = yatl::Timer::new();
    timer.start().unwrap();
    let kernel_event = ExecuteKernel::new(kernel)
        .set_arg_svm(input.as_ptr())
        .set_arg_svm(output.as_mut_ptr())
        .set_arg_svm(value_svm.as_ptr())
        .set_global_work_size(pixel_count)
        .enqueue_nd_range(&queue)
        .unwrap();
    kernel_event.wait().unwrap();
    timer.lap().unwrap()
}