use opencl3::svm::SvmVec;
use opencl3::context::Context;
use std::ffi::CString;
use crate::exit_codes::UNKNOWN_KERNEL;
use crate::effects::cl::{run_in_out_pixel_based_kernel, run_in_out_pixel_based_kernel_1f};

pub fn invert(context: &Context, byte_count: usize, input: &SvmVec<u8>, mut output: &mut SvmVec<u8>) {
    let kernel_name: CString = CString::new("invert").unwrap();
    let queue = context.default_queue();
    if let Some(kernel) = context.get_kernel(&kernel_name) {
        let lap = run_in_out_pixel_based_kernel(&kernel, byte_count / 4, &input, &mut output, &queue);
        println!("Inverted {} pixels in {}", byte_count / 4, yatl::duration_to_human_string(&lap));
    } else {
        exit!(UNKNOWN_KERNEL, "Cannot find kernel for INVERT");
    }
}

pub fn greyscale(context: &Context, byte_count: usize, input: &SvmVec<u8>, mut output: &mut SvmVec<u8>) {
    let kernel_name: CString = CString::new("greyscale").unwrap();
    let queue = context.default_queue();
    if let Some(kernel) = context.get_kernel(&kernel_name) {
        let lap = run_in_out_pixel_based_kernel(&kernel, byte_count / 4, &input, &mut output, &queue);
        println!("Greyscaled {} pixels in {}", byte_count / 4, yatl::duration_to_human_string(&lap));
    } else {
        exit!(UNKNOWN_KERNEL, "Cannot find kernel for GREYSCALE");
    }
}

pub fn contrast(context: &Context, byte_count: usize, input: &SvmVec<u8>, mut output: &mut SvmVec<u8>, intensity: f32) {
    let kernel_name: CString = CString::new("contrast").unwrap();
    let queue = context.default_queue();
    if let Some(kernel) = context.get_kernel(&kernel_name) {
        let lap = run_in_out_pixel_based_kernel_1f(&context, &kernel, byte_count / 4, &input, &mut output, &queue, intensity);
        println!("Adjusted contrast of {} pixels in {}", byte_count / 4, yatl::duration_to_human_string(&lap));
    } else {
        exit!(UNKNOWN_KERNEL, "Cannot find kernel for CONTRAST");
    }
}

pub fn brightness(context: &Context, byte_count: usize, input: &SvmVec<u8>, mut output: &mut SvmVec<u8>, intensity: f32) {
    let kernel_name: CString = CString::new("brightness").unwrap();
    let queue = context.default_queue();
    if let Some(kernel) = context.get_kernel(&kernel_name) {
        let lap = run_in_out_pixel_based_kernel_1f(&context, &kernel, byte_count / 4, &input, &mut output, &queue, intensity);
        println!("Adjusted brightness of {} pixels in {}", byte_count / 4, yatl::duration_to_human_string(&lap));
    } else {
        exit!(UNKNOWN_KERNEL, "Cannot find kernel for BRIGHTNESS");
    }
}

pub fn schwurbel(context: &Context, byte_count: usize, input: &SvmVec<u8>, mut output: &mut SvmVec<u8>, intensity: f32) {
    let kernel_name: CString = CString::new("schwurbel").unwrap();
    let queue = context.default_queue();
    if let Some(kernel) = context.get_kernel(&kernel_name) {
        let lap = run_in_out_pixel_based_kernel_1f(&context, &kernel, byte_count / 4, &input, &mut output, &queue, intensity);
        println!("Schwurbeled {} pixels in {}", byte_count / 4, yatl::duration_to_human_string(&lap));
    } else {
        exit!(UNKNOWN_KERNEL, "Cannot find kernel for SCHWURBEL");
    }
}
