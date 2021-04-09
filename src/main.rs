use crate::effects::Effect;
use std::convert::TryFrom;
use image::{ColorType, ImageFormat};

pub mod effects;

fn main() {
    let mut args = std::env::args();
    args.next();
    let img = if let Some(path) = args.next() {
        image::open(path).unwrap()
    } else {
        panic!("Please give path!")
    };
    let img = img.into_rgba8();
    let (w, h) = (img.width(), img.height());
    let img = img.into_raw();
    let effect_list: Vec<Effect> = args.into_iter().map(|arg| Effect::try_from(&arg).unwrap()).collect();
    let img = effects::run_effects(img, effect_list);
    image::save_buffer_with_format("output.png", &img.as_slice(), w, h, ColorType::Rgba8, ImageFormat::Png).unwrap()
}
