use opencl3::svm::SvmVec;
use opencl3::context::Context;
use std::ffi::CString;
use opencl3::kernel::ExecuteKernel;

pub fn invert(context: &Context, byte_count: usize, input: &SvmVec<u8>, output: &mut SvmVec<u8>) {
    let kernel_name: CString = CString::new("invert").unwrap();
    let queue = context.default_queue();
    if let Some(kernel) = context.get_kernel(&kernel_name) {
        let mut timer = yatl::Timer::new();
        timer.start().unwrap();
        let kernel_event = ExecuteKernel::new(kernel)
            .set_arg_svm(input.as_ptr())
            .set_arg_svm(output.as_mut_ptr())
            .set_global_work_size(byte_count / 4)
            .enqueue_nd_range(&queue)
            .unwrap();

        kernel_event.wait().unwrap();
        let lap = timer.lap().unwrap();
        println!("Inverted {} pixels in {}", byte_count / 4, yatl::duration_to_human_string(&lap));
    } else {
        panic!("Cannot find kernel for INVERT");
    }
}

pub fn greyscale(context: &Context, byte_count: usize, input: &SvmVec<u8>, output: &mut SvmVec<u8>) {
    let kernel_name: CString = CString::new("greyscale").unwrap();
    let queue = context.default_queue();
    if let Some(kernel) = context.get_kernel(&kernel_name) {
        let mut timer = yatl::Timer::new();
        timer.start().unwrap();
        let kernel_event = ExecuteKernel::new(kernel)
            .set_arg_svm(input.as_ptr())
            .set_arg_svm(output.as_mut_ptr())
            .set_global_work_size(byte_count / 4)
            .enqueue_nd_range(&queue)
            .unwrap();

        kernel_event.wait().unwrap();
        let lap = timer.lap().unwrap();
        println!("Greyscaled {} pixels in {}", byte_count / 4, yatl::duration_to_human_string(&lap));
    } else {
        panic!("Cannot find kernel for GREYSCALE");
    }
}