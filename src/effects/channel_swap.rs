use std::convert::TryFrom;
use opencl3::context::Context;
use opencl3::svm::SvmVec;

use crate::exit_codes::{UNKNOWN_CHANNEL, UNKNOWN_KERNEL};
use std::ffi::CString;
use crate::effects::cl::run_in_out_pixel_based_kernel_1iv;

pub struct ChannelSwap {
    order: Vec<i32>,
}

impl TryFrom<&str> for ChannelSwap {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let split = value.split(",");
        let count = split.clone().count();
        assert!(3 == count || 4 == count, "Channel swapping requires 3 or 4 parameters");
        assert!(split.clone().find(|s| s.len() != 1).is_none(), "Channel swapping parameters must only be one char");
        let split: Vec<char> = split.map(|s| s.chars().next().unwrap()).collect();
        let mut order = vec![];
        split.iter().map(|c| match c {
            'r' => 0,
            'g' => 1,
            'b' => 2,
            'a' => 3,
            c => exit!(UNKNOWN_CHANNEL, format!("Unknown color channel `{}`", c)),
        }).for_each(|c| order.push(c));
        if order.len() == 3 {
            order.push(3);
        }
        Ok(ChannelSwap {
            order,
        })
    }
}

impl ChannelSwap {
    pub fn run(&self, context: &Context, byte_count: usize, input: &SvmVec<u8>, mut output: &mut SvmVec<u8>) {
        let kernel_name: CString = CString::new("channel_swap").unwrap();
        let queue = context.default_queue();
        if let Some(kernel) = context.get_kernel(&kernel_name) {
            let lap = run_in_out_pixel_based_kernel_1iv(&context, &kernel, byte_count / 4, &input, &mut output, &queue, &self.order);
            println!("Swapped color channels of {} pixels in {}", byte_count / 4, yatl::duration_to_human_string(&lap));
        } else {
            exit!(UNKNOWN_KERNEL, "Cannot find kernel for CHANNEL_SWAP");
        }
    }
}