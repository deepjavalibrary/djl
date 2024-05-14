use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute::{
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
};
use candle_core::cuda_backend::cudarc::driver::CudaDevice;
use std::sync::Once;

static INIT: Once = Once::new();
static mut RUNTIME_COMPUTE_CAP: usize = 0;

fn init_compute_caps() {
    unsafe {
        INIT.call_once(|| {
            let device = CudaDevice::new(0).expect("cuda is not available");
            let major = device
                .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
                .unwrap();
            let minor = device
                .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
                .unwrap();
            RUNTIME_COMPUTE_CAP = (major * 10 + minor) as usize;
        });
    }
}

pub fn get_runtime_compute_cap() -> usize {
    unsafe {
        init_compute_caps();
        RUNTIME_COMPUTE_CAP
    }
}
