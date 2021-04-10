use crate::effects::Effect;
use std::convert::TryFrom;
use image::{ColorType, ImageFormat};

macro_rules! exit {
    ($code:expr, $message:expr) => {
        {
            use std::process::exit;
            println!("{}", $message);
            exit($code);
        }
    };
}

pub mod effects;

pub mod exit_codes {
    pub const NO_INPUT_FILE: i32 = 1;
    pub const UNKNOWN_KERNEL: i32 = 2;
    pub const UNKNOWN_EFFECT: i32 = 3;
}

fn main() {
    use exit_codes::*;

    let mut args = std::env::args();
    args.next();
    let img = if let Some(path) = args.next() {
        image::open(path).unwrap()
    } else {
        exit!(NO_INPUT_FILE, "Please give path!");
    };
    let img = img.into_rgba8();
    let (w, h) = (img.width(), img.height());
    let img = img.into_raw();
    let effect_list: Vec<Effect> = args.into_iter()
        .map(|arg| Effect::try_from(&arg).unwrap_or_else(|msg| exit!(UNKNOWN_EFFECT, msg)))
        .collect();
    let img = effects::run_effects(img, effect_list);
    image::save_buffer_with_format("output.png", &img.as_slice(), w, h, ColorType::Rgba8, ImageFormat::Png).unwrap()
}
