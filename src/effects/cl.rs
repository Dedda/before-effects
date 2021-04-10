use opencl3::context::Context;
use opencl3::platform::get_platforms;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device};
use std::sync::Once;
use std::ffi::CString;

pub fn create_context() -> Context {
    let platform_ids = get_platforms().unwrap();
    assert!(0 < platform_ids.len(), "Could not find any OpenCL platforms.");
    let platform_id = platform_ids.first().unwrap();
    let device_ids = opencl3::device::get_device_ids(platform_id.id(), CL_DEVICE_TYPE_GPU).unwrap();
    assert!(0 < device_ids.len(), "Could not find any OpenCL devices.");
    let devices: Vec<Device> = device_ids.into_iter().map(|id| Device::new(id)).collect();
    let device = devices.into_iter().next().unwrap();

    let mut context = Context::from_device(device).unwrap();
    context.create_command_queues_with_properties(0, 0).unwrap();
    context
}

pub fn compile_kernels(context: &mut Context) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let mut timer = yatl::Timer::new();
        debug!("# Compiling kernels... ");
        timer.start().unwrap();
        let options = CString::new("").unwrap();
        context.build_program_from_source(&CString::new(include_str!("simple.cl")).unwrap(), &options).unwrap();
        debugln!("[ OK ] {}", yatl::duration_to_human_string(&timer.lap().unwrap()));
    });
}