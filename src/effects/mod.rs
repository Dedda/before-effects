use std::convert::TryFrom;
use crate::effects::Effect::{Invert, Greyscale, Contrast, Brightness, Schwurbel};
use opencl3::context::Context;
use opencl3::svm::SvmVec;
use opencl3::types::cl_uchar;
use crate::effects::cl::compile_kernels;
use yatl::{Timer, duration_to_human_string};
use crate::effects::channel_swap::ChannelSwap;
use crate::effects::color_intensity::ColorIntensity;
use crate::effects::color_rotate::ColorRotate;

mod channel_swap;
mod color_intensity;
mod color_rotate;
mod cl;
mod simple;

pub enum Effect {
    Invert,
    Greyscale,
    Contrast(f32),
    Brightness(f32),
    Schwurbel(f32),
    ChannelSwap(ChannelSwap),
    ColorIntensity(ColorIntensity),
    ColorRotate(ColorRotate),
}

impl TryFrom<&String> for Effect {
    type Error = String;

    fn try_from(value: &String) -> Result<Self, Self::Error> {
        let mut split = value.split("=");
        let name = split.next().unwrap();
        match name {
            "invert" => Ok(Invert),
            "greyscale" => Ok(Greyscale),
            "contrast" => {
                let intensity: f32 = split.next().unwrap().parse().unwrap();
                let intensity = intensity.clamp(0.0, 1.0);
                Ok(Contrast(intensity))
            },
            "brightness" => {
                let intensity: f32 = split.next().unwrap().parse().unwrap();
                let intensity = intensity.clamp(0.0, 1.0);
                Ok(Brightness(intensity))
            },
            "schwurbel" => {
                let intensity: f32 = split.next().unwrap().parse().unwrap();
                let intensity = intensity.clamp(0.0, 1.0);
                Ok(Schwurbel(intensity))
            },
            "chswap" => Ok(Effect::ChannelSwap(ChannelSwap::try_from(split.next().unwrap()).unwrap())),
            "intensity" => Ok(Effect::ColorIntensity(ColorIntensity::try_from(split.next().unwrap()).unwrap())),
            "crotate" => Ok(Effect::ColorRotate(ColorRotate::try_from(split.next().unwrap()).unwrap())),
            n => Err(format!("Unknown effect: {}", n)),
        }
    }
}

impl Effect {
    fn run(&self, context: &Context, byte_count: usize, input: &SvmVec<u8>, output: &mut SvmVec<u8>) {
        match self {
            Invert => simple::invert(&context, byte_count, input, output),
            Greyscale => simple::greyscale(&context, byte_count, input, output),
            Contrast(intensity) => simple::contrast(&context, byte_count, input, output, intensity.clone()),
            Brightness(intensity) => simple::brightness(&context, byte_count, input, output, intensity.clone()),
            Schwurbel(intensity) => simple::schwurbel(&context, byte_count, input, output, intensity.clone()),
            Effect::ChannelSwap(chswap) => chswap.run(&context, byte_count, input, output),
            Effect::ColorIntensity(color_intensity) => color_intensity.run(&context, byte_count, input, output),
            Effect::ColorRotate(color_rotate) => color_rotate.run(&context, byte_count, input, output),
        }
    }
}

pub fn run_effects(img: Vec<u8>, effects: Vec<Effect>) -> Vec<u8> {
    let mut timer = Timer::new();
    let byte_count = img.len();
    debug!("# Creating OpenCL context... ");
    timer.start().unwrap();
    let mut context = cl::create_context();
    let svm_capability = context.get_svm_mem_capability();
    debugln!("[ OK ] {}", duration_to_human_string(&timer.lap().unwrap()));
    compile_kernels(&mut context);
    timer.lap().unwrap();
    debug!("# Preparing buffers... ");
    let mut output_buffer = SvmVec::<cl_uchar>::with_capacity_zeroed(&context, svm_capability, byte_count);
    unsafe { output_buffer.set_len(byte_count) };
    let mut input_buffer = SvmVec::<cl_uchar>::with_capacity(&context, svm_capability, byte_count);
    for byte in img {
        input_buffer.push(byte);
    }
    debugln!("[ OK ] {}", duration_to_human_string(&timer.lap().unwrap()));
    let mut swap_buffers = false;
    debugln!("# Running {} effects", effects.len());
    for effect in effects {
        let (input, mut output) = match swap_buffers {
            true => (&output_buffer, &mut input_buffer),
            false => (&input_buffer, &mut output_buffer),
        };
        effect.run(&context, byte_count, &input, &mut output);
        swap_buffers = !swap_buffers;
    }

    if swap_buffers {
        output_buffer.to_vec()
    } else {
        input_buffer.to_vec()
    }
}