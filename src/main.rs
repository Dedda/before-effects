use crate::effects::Effect;
use std::convert::TryFrom;
use image::{ColorType, ImageFormat};
use yatl::{duration_to_human_string, Timer};

macro_rules! exit {
    ($code:expr, $message:expr) => {
        {
            use std::process::exit;
            println!("{}", $message);
            exit($code);
        }
    };
}

pub const DEBUG_MODE: Option<&'static str> = option_env!("DEBUG");

macro_rules! debug {
    ($($arg:tt)*) => {
        if crate::DEBUG_MODE.is_some() {
            print!($($arg)*);
        }
    }
}

macro_rules! debugln {
    ($($arg:tt)*) => {
        if crate::DEBUG_MODE.is_some() {
            println!($($arg)*);
        }
    }
}

pub mod effects;

pub mod exit_codes {
    pub const NO_INPUT_FILE: i32 = 1;
    pub const UNKNOWN_KERNEL: i32 = 2;
    pub const UNKNOWN_EFFECT: i32 = 3;
    pub const UNKNOWN_CHANNEL: i32 = 4;
}

fn main() {
    use exit_codes::*;

    debugln!("# Debug output is enabled");

    let mut args = std::env::args();
    args.next();
    let img = if let Some(path) = args.next() {
        debugln!("# Using input file: `{}`", &path);
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
    let mut timer = Timer::new();
    debug!("# Exporting polished picture... ");
    timer.start().unwrap();
    image::save_buffer_with_format("output.png", &img.as_slice(), w, h, ColorType::Rgba8, ImageFormat::Png).unwrap();
    debugln!("[ OK ] {}", duration_to_human_string(&timer.lap().unwrap()));
}
