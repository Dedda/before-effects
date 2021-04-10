use std::convert::TryFrom;
use crate::effects::Effect::{Invert, Greyscale};
use opencl3::context::Context;
use opencl3::svm::SvmVec;
use opencl3::types::cl_uchar;
use crate::effects::cl::compile_kernels;

mod cl;
mod simple;

pub enum Effect {
    Invert,
    Greyscale,
}

impl TryFrom<&String> for Effect {
    type Error = String;

    fn try_from(value: &String) -> Result<Self, Self::Error> {
        let mut split = value.split("=");
        let name = split.next().unwrap();
        match name {
            "invert" => Ok(Invert),
            "greyscale" => Ok(Greyscale),
            n => Err(format!("Unknown effect: {}", n)),
        }
    }
}

impl Effect {
    fn run(&self, context: &Context, byte_count: usize, input: &SvmVec<u8>, output: &mut SvmVec<u8>) {
        match self {
            Invert => simple::invert(&context, byte_count, input, output),
            Greyscale => simple::greyscale(&context, byte_count, input, output),
        }
    }
}

pub fn run_effects(img: Vec<u8>, effects: Vec<Effect>) -> Vec<u8> {
    let byte_count = img.len();
    let mut context = cl::create_context();
    let svm_capability = context.get_svm_mem_capability();
    compile_kernels(&mut context);
    let mut output_buffer = SvmVec::<cl_uchar>::with_capacity_zeroed(&context, svm_capability, byte_count);
    unsafe { output_buffer.set_len(byte_count) };
    let mut input_buffer = SvmVec::<cl_uchar>::with_capacity(&context, svm_capability, byte_count);
    for byte in img {
        input_buffer.push(byte);
    }
    let mut swap_buffers = false;
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