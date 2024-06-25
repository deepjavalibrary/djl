use crate::cast_handle;
use crate::ndarray::{as_data_type, as_device, as_shape, return_handle};
use candle::{DType, Error, Tensor};
use half::{bf16, f16};
use jni::objects::{JByteBuffer, JLongArray, JObject, JString};
use jni::sys::{jfloat, jint, jlong};
use jni::JNIEnv;
use std::slice;

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_tensorOf<'local>(
    mut env: JNIEnv<'local>,
    _: JObject,
    buffer: JByteBuffer<'local>,
    shape: JLongArray<'local>,
    dtype: jint,
    device_type: JString,
    device_id: jint,
) -> jlong {
    let tensor = || {
        let shape = as_shape(&mut env, &shape);
        let device = as_device(&mut env, device_type, device_id as usize)?;
        let dtype = as_data_type(dtype)?;

        let len = env.get_direct_buffer_capacity(&buffer).unwrap();
        let data = env.get_direct_buffer_address(&buffer).unwrap();
        let data = unsafe { slice::from_raw_parts(data, len) };
        Tensor::from_raw_buffer(data, dtype, shape.dims(), &device)
    };
    let ret = tensor();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_zeros<'local>(
    mut env: JNIEnv,
    _: JObject,
    shape: JLongArray<'local>,
    dtype: jint,
    device_type: JString,
    device_id: jint,
) -> jlong {
    let tensor = || {
        let shape = as_shape(&mut env, &shape);
        let device = as_device(&mut env, device_type, device_id as usize)?;
        let dtype = as_data_type(dtype)?;
        Tensor::zeros(&shape, dtype, &device)
    };
    let ret = tensor();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_ones<'local>(
    mut env: JNIEnv,
    _: JObject,
    shape: JLongArray<'local>,
    dtype: jint,
    device_type: JString,
    device_id: jint,
) -> jlong {
    let tensor = || {
        let shape = as_shape(&mut env, &shape);
        let device = as_device(&mut env, device_type, device_id as usize)?;
        let dtype = as_data_type(dtype)?;
        Tensor::ones(&shape, dtype, &device)
    };
    let ret = tensor();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_full<'local>(
    mut env: JNIEnv,
    _: JObject,
    value: jfloat,
    shape: JLongArray<'local>,
    dtype: jint,
    device_type: JString,
    device_id: jint,
) -> jlong {
    let tensor = || {
        let shape = as_shape(&mut env, &shape);
        let device = as_device(&mut env, device_type, device_id as usize)?;
        let dtype = as_data_type(dtype)?;
        match dtype {
            DType::U8 => {
                let tmp = value as i64;
                Tensor::full(tmp as u8, &shape, &device)
            }
            DType::U32 => {
                let tmp = value as i64;
                Tensor::full(tmp as u32, &shape, &device)
            }
            DType::I64 => Tensor::full(value as i64, &shape, &device),
            DType::BF16 => Tensor::full(bf16::from_f32(value), &shape, &device),
            DType::F16 => Tensor::full(f16::from_f32(value), &shape, &device),
            DType::F32 => Tensor::full(value as f32, &shape, &device),
            DType::F64 => Tensor::full(value as f64, &shape, &device),
        }
    };
    let ret = tensor();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_arange<'local>(
    mut env: JNIEnv,
    _: JObject,
    start: jfloat,
    stop: jfloat,
    step: jfloat,
    dtype: jint,
    device_type: JString,
    device_id: jint,
) -> jlong {
    let tensor = || {
        let device = as_device(&mut env, device_type, device_id as usize)?;
        let dtype = as_data_type(dtype)?;
        match dtype {
            DType::U8 => Tensor::arange_step(start as u8, stop as u8, step as u8, &device),
            DType::U32 => Tensor::arange_step(start as u32, stop as u32, step as u32, &device),
            DType::I64 => Tensor::arange_step(start as i64, stop as i64, step as i64, &device),
            DType::BF16 => Tensor::arange_step(
                bf16::from_f32(start),
                bf16::from_f32(stop),
                bf16::from_f32(step),
                &device,
            ),
            DType::F16 => Tensor::arange_step(
                f16::from_f32(start),
                f16::from_f32(stop),
                f16::from_f32(step),
                &device,
            ),
            DType::F32 => Tensor::arange_step(start as f32, stop as f32, step as f32, &device),
            DType::F64 => Tensor::arange_step(start as f64, stop as f64, step as f64, &device),
        }
    };
    let ret = tensor();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_eye<'local>(
    mut env: JNIEnv,
    _: JObject,
    rows: jint,
    _columns: jint,
    dtype: jint,
    device_type: JString,
    device_id: jint,
) -> jlong {
    let tensor = || {
        let device = as_device(&mut env, device_type, device_id as usize)?;
        let dtype = as_data_type(dtype)?;
        Tensor::eye(rows as usize, dtype, &device)
    };
    let ret = tensor();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_uniform<'local>(
    mut env: JNIEnv,
    _: JObject,
    low: jfloat,
    high: jfloat,
    shape: JLongArray<'local>,
    dtype: jint,
    device_type: JString,
    device_id: jint,
) -> jlong {
    let tensor = || {
        let shape = as_shape(&mut env, &shape);
        let device = as_device(&mut env, device_type, device_id as usize)?;
        let dtype = as_data_type(dtype)?;
        match dtype {
            DType::F32 => Tensor::rand(low as f32, high as f32, &shape, &device),
            DType::F64 => Tensor::rand(low as f64, high as f64, &shape, &device),
            _ => Err(Error::UnsupportedDTypeForOp(dtype, "rand_uniform")),
        }
    };
    let ret = tensor();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_randomNormal<'local>(
    mut env: JNIEnv,
    _: JObject,
    mean: jfloat,
    std: jfloat,
    shape: JLongArray<'local>,
    dtype: jint,
    device_type: JString,
    device_id: jint,
) -> jlong {
    let tensor = || {
        let shape = as_shape(&mut env, &shape);
        let device = as_device(&mut env, device_type, device_id as usize)?;
        let dtype = as_data_type(dtype)?;
        match dtype {
            DType::F32 => Tensor::randn(mean as f32, std as f32, &shape, &device),
            DType::F64 => Tensor::randn(mean as f64, std as f64, &shape, &device),
            _ => Err(Error::UnsupportedDTypeForOp(dtype, "rand_norm")),
        }
    };
    let ret = tensor();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_duplicate<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    return_handle(&mut env, tensor.copy())
}
