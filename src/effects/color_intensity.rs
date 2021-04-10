use opencl3::context::Context;
use std::ffi::CString;
use opencl3::svm::SvmVec;
use opencl3::types::cl_float;
use std::convert::TryFrom;
use crate::exit_codes::UNKNOWN_KERNEL;
use crate::effects::cl::run_pixel_based_kernel_1v;
use crate::rgba_char_to_channel_index;

pub struct ColorIntensity {
    intensities: Vec<f32>,
}

impl TryFrom<&str> for ColorIntensity {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let split = value.split(",");
        let mut intensities: [f32;4] = [0.5;4];
        assert!(5 > split.clone().count(), "Color intensity effect can only accept up to 4 channels");
        for s in split {
            let (channel, intensity) = channel_intensity(s)?;
            intensities[channel] = intensity;
        }
        Ok(ColorIntensity {
            intensities: intensities.to_vec(),
        })
    }
}

fn channel_intensity(s: &str) -> Result<(usize, f32), ()> {
    let mut parts = s.split(":");
    let channel = rgba_char_to_channel_index(&parts.next().unwrap().chars().next().unwrap()).unwrap();
    let intensity = parts.next().unwrap().parse().unwrap();
    Ok((channel, intensity))
}

impl ColorIntensity {
    pub fn run(&self, context: &Context, byte_count: usize, input: &SvmVec<u8>, mut output: &mut SvmVec<u8>) {
        let kernel_name: CString = CString::new("color_intensity").unwrap();
        let queue = context.default_queue();
        if let Some(kernel) = context.get_kernel(&kernel_name) {
            let lap = run_pixel_based_kernel_1v::<cl_float>(&context, &kernel, byte_count / 4, &input, &mut output, &queue, &self.intensities);
            println!("Intensified colors of {} pixels in {}", byte_count / 4, yatl::duration_to_human_string(&lap));
        } else {
            exit!(UNKNOWN_KERNEL, "Cannot find kernel for COLOR_INTENSITY");
        }
    }
}