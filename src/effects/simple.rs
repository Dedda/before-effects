use opencl3::svm::SvmVec;
use opencl3::context::Context;
use std::ffi::CString;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::command_queue::CommandQueue;
use std::time::Duration;

fn run_in_out_pixel_based_kernel(kernel: &Kernel, pixel_count: usize, input: &SvmVec<u8>, output: &mut SvmVec<u8>, queue: &CommandQueue) -> Duration {
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

pub fn invert(context: &Context, byte_count: usize, input: &SvmVec<u8>, mut output: &mut SvmVec<u8>) {
    let kernel_name: CString = CString::new("invert").unwrap();
    let queue = context.default_queue();
    if let Some(kernel) = context.get_kernel(&kernel_name) {
        let lap = run_in_out_pixel_based_kernel(&kernel, byte_count / 4, &input, &mut output, &queue);
        println!("Inverted {} pixels in {}", byte_count / 4, yatl::duration_to_human_string(&lap));
    } else {
        panic!("Cannot find kernel for INVERT");
    }
}

pub fn greyscale(context: &Context, byte_count: usize, input: &SvmVec<u8>, mut output: &mut SvmVec<u8>) {
    let kernel_name: CString = CString::new("greyscale").unwrap();
    let queue = context.default_queue();
    if let Some(kernel) = context.get_kernel(&kernel_name) {
        let lap = run_in_out_pixel_based_kernel(&kernel, byte_count / 4, &input, &mut output, &queue);
        println!("Greyscaled {} pixels in {}", byte_count / 4, yatl::duration_to_human_string(&lap));
    } else {
        panic!("Cannot find kernel for GREYSCALE");
    }
}

