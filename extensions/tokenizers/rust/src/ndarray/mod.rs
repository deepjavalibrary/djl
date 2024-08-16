use candle::{DType, Device, Error, Result, Shape, Tensor, WithDType};
use half::{bf16, f16};
use jni::objects::{JByteBuffer, JIntArray, JLongArray, JObject, JString, ReleaseMode};
use jni::sys::{jint, jlong};
use jni::JNIEnv;

use crate::{cast_handle, drop_handle, to_handle};

mod binary;
mod cmp;
mod creation;
mod nn;
mod other;
mod reduce;
mod unary;

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_getDataType(
    _: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jint {
    let tensor = cast_handle::<Tensor>(handle);
    to_data_type(tensor.dtype())
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_getDevice<'local>(
    env: JNIEnv<'local>,
    _: JObject,
    handle: jlong,
) -> JIntArray<'local> {
    let tensor = cast_handle::<Tensor>(handle);
    let device = tensor.device();
    let array = env.new_int_array(2).unwrap();
    let mut device_type = 0;
    let mut device_id = -1;
    if device.is_cpu() {
        device_type = 0;
    } else if device.is_cuda() {
        device_type = 1;
        device_id = 0;
    } else if device.is_metal() {
        device_type = 2;
    }
    let values = [device_type, device_id];
    env.set_int_array_region(&array, 0, &values).unwrap();
    array
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_getShape<'local>(
    env: JNIEnv<'local>,
    _: JObject,
    handle: jlong,
) -> JLongArray<'local> {
    let tensor = cast_handle::<Tensor>(handle);
    let shape = tensor.shape();
    let dims = shape
        .dims()
        .into_iter()
        .map(|i| *i as jlong)
        .collect::<Vec<jlong>>();
    let len = dims.len() as jint;

    let array = env.new_long_array(len).unwrap();
    env.set_long_array_region(&array, 0, &dims).unwrap();
    array
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_getByteBuffer<'local>(
    mut env: JNIEnv<'local>,
    _: JObject,
    handle: jlong,
) -> JByteBuffer<'local> {
    let tensor = cast_handle::<Tensor>(handle).flatten_all().unwrap();
    let (ptr, len) = match tensor.dtype() {
        DType::U8 => convert_back_::<u8>(tensor.to_vec1().unwrap()),
        DType::U32 => convert_back_::<u32>(tensor.to_vec1().unwrap()),
        DType::I64 => convert_back_::<i64>(tensor.to_vec1().unwrap()),
        DType::F16 => convert_back_::<f16>(tensor.to_vec1().unwrap()),
        DType::BF16 => convert_back_::<bf16>(tensor.to_vec1().unwrap()),
        DType::F32 => convert_back_::<f32>(tensor.to_vec1().unwrap()),
        DType::F64 => convert_back_::<f64>(tensor.to_vec1().unwrap()),
    };

    let buf = unsafe { env.new_direct_byte_buffer(ptr, len) }.unwrap();
    buf
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_toDevice<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    device_type: JString,
    device_id: jint,
) -> jlong {
    let to_device = || {
        let device = as_device(&mut env, device_type, device_id as usize)?;
        let tensor = cast_handle::<Tensor>(handle);
        tensor.to_device(&device)
    };
    let ret = to_device();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_toDataType<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    dtype: jint,
) -> jlong {
    let to_data_type = || {
        let dtype = as_data_type(dtype)?;
        let tensor = cast_handle::<Tensor>(handle);
        tensor.to_dtype(dtype)
    };
    let ret = to_data_type();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_toBoolean<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let to_boolean = || {
        let tensor = cast_handle::<Tensor>(handle);
        let tensor = tensor.to_dtype(DType::U8)?;
        let zeros = tensor.zeros_like()?;
        tensor.ne(&zeros)
    };
    let ret = to_boolean();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_fullSlice<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    min: JLongArray<'local>,
    max: JLongArray<'local>,
    _: JLongArray<'local>,
) -> jlong {
    let mut index = || {
        let tensor = cast_handle::<Tensor>(handle);
        let min = unsafe { env.get_array_elements(&min, ReleaseMode::NoCopyBack) }.unwrap();
        let min = min.into_iter().map(|i| *i as usize).collect::<Vec<usize>>();
        let max = unsafe { env.get_array_elements(&max, ReleaseMode::NoCopyBack) }.unwrap();
        let max = max.into_iter().map(|i| *i as usize).collect::<Vec<usize>>();
        if min.len() == 0 {
            tensor.copy()
        } else {
            let mut slice = tensor.narrow(0, min[0], max[0] - min[0])?;
            for i in 1..min.len() {
                slice = slice.narrow(i, min[i], max[i] - min[i])?;
            }
            Ok(slice)
        }
    };
    let ret = index();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_gather<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    index_handle: jlong,
    axis: jint,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let index_tensor = cast_handle::<Tensor>(index_handle);
    let ret = tensor.gather(&index_tensor, axis as usize);
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_scatter<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    index_handle: jlong,
    value_handle: jlong,
    axis: jint,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let index_tensor = cast_handle::<Tensor>(index_handle);
    let value_tensor = cast_handle::<Tensor>(value_handle);
    let ret = tensor.scatter_add(&index_tensor, &value_tensor, axis as usize);
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_countNonzero<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let count = || {
        let tensor = cast_handle::<Tensor>(handle).to_dtype(DType::F32)?;
        let zeros = tensor.zeros_like()?;
        tensor.ne(&zeros)?.sum_all()?.to_dtype(DType::I64)
    };
    let ret = count();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_countNonzeroWithAxis<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    axis: jint,
) -> jlong {
    let count = || {
        let tensor = cast_handle::<Tensor>(handle).to_dtype(DType::U32)?;
        let zeros = tensor.zeros_like()?;
        tensor.ne(&zeros)?.sum(axis as usize)?.to_dtype(DType::I64)
    };
    let ret = count();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_deleteTensor(
    _: JNIEnv,
    _: JObject,
    handle: jlong,
) {
    drop_handle::<Tensor>(handle);
}

fn convert_back_<T: WithDType>(mut vs: Vec<T>) -> (*mut u8, usize) {
    let size_in_bytes = T::DTYPE.size_in_bytes();
    let length = vs.len() * size_in_bytes;
    let ptr = vs.as_mut_ptr() as *mut u8;
    // Don't run the destructor for Vec<T>
    std::mem::forget(vs);
    // SAFETY:
    //
    // Every T is larger than u8, so there is no issue regarding alignment.
    // This re-interpret the Vec<T> as a Vec<u8>.
    (ptr, length)
}

fn as_shape<'local>(env: &mut JNIEnv, shape: &JLongArray<'local>) -> Shape {
    let shape = unsafe { env.get_array_elements(&shape, ReleaseMode::NoCopyBack) }.unwrap();
    let shape = shape
        .into_iter()
        .map(|i| *i as usize)
        .collect::<Vec<usize>>();
    Shape::from_dims(&shape)
}

pub fn as_data_type(data_type: i32) -> Result<DType> {
    match data_type {
        0 => Ok(DType::F32),
        1 => Ok(DType::F64),
        2 => Ok(DType::F16),
        3 => Ok(DType::U8),
        6 => Ok(DType::I64),
        11 => Ok(DType::BF16),
        13 => Ok(DType::U32),
        _ => Err(Error::Msg(format!("Unsupported data type: {}", data_type))),
    }
}

fn to_data_type(data_type: DType) -> i32 {
    match data_type {
        DType::F32 => 0,
        DType::F64 => 1,
        DType::F16 => 2,
        DType::U8 => 3,
        DType::I64 => 6,
        DType::BF16 => 11,
        DType::U32 => 13,
    }
}

pub fn as_device<'local>(
    env: &mut JNIEnv<'local>,
    device_type: JString,
    device_id: usize,
) -> Result<Device> {
    let device_type: String = env
        .get_string(&device_type)
        .expect("Couldn't get java string!")
        .into();

    match device_type.as_str() {
        "cpu" => Ok(Device::Cpu),
        "gpu" => Device::new_cuda(device_id),
        "mps" => Device::new_metal(device_id),
        _ => Err(Error::Msg(format!("Invalid device type: {}", device_type))),
    }
}

fn return_handle(env: &mut JNIEnv, tensor: Result<Tensor>) -> jlong {
    match tensor {
        Ok(output) => to_handle(output),
        Err(err) => {
            let msg = format!("{err:?}");
            match err {
                Error::UnexpectedDType { .. }
                | Error::DTypeMismatchBinaryOp { .. }
                | Error::UnsupportedDTypeForOp(_, _) => {
                    env.throw_new("java/lang/UnsupportedOperationException", msg)
                        .unwrap();
                }
                _ => {
                    env.throw_new("ai/djl/engine/EngineException", msg).unwrap();
                }
            }
            0
        }
    }
}
