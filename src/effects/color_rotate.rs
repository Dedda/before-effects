use opencl3::context::Context;
use std::ffi::CString;
use opencl3::svm::SvmVec;
use opencl3::types::cl_float;
use std::convert::TryFrom;
use crate::exit_codes::UNKNOWN_KERNEL;
use crate::effects::cl::run_pixel_based_kernel_1;

pub struct ColorRotate {
    absolute: bool,
    degrees: f32,
}

impl TryFrom<&str> for ColorRotate {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let (absolute, degrees) = if value.starts_with("+") || value.starts_with("-") {
            let degrees: f32 = value.parse().unwrap();
            (false, (degrees + 360.0) % 360.0)
        } else {
            let degrees: f32 = value.parse().unwrap();
            (true, degrees % 360.0)
        };
        Ok(ColorRotate {
            absolute,
            degrees,
        })
    }
}

impl ColorRotate {
    pub fn run(&self, context: &Context, byte_count: usize, input: &SvmVec<u8>, mut output: &mut SvmVec<u8>) {
        let kernel_name: CString = CString::new(if self.absolute {
            "color_rotate_absolute"
        } else {
            "color_rotate"
        }).unwrap();
        let queue = context.default_queue();
        if let Some(kernel) = context.get_kernel(&kernel_name) {
            let lap = run_pixel_based_kernel_1::<cl_float>(&context, &kernel, byte_count / 4, &input, &mut output, &queue, &self.degrees);
            println!("Rotated colors of {} pixels in {}", byte_count / 4, yatl::duration_to_human_string(&lap));
        } else {
            exit!(UNKNOWN_KERNEL, "Cannot find kernel for COLOR_ROTATE");
        }
    }
}