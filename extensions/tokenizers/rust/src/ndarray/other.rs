use candle_core::{Tensor, D};
use jni::objects::{JIntArray, JLongArray, JObject, ReleaseMode};
use jni::sys::{jdouble, jint, jlong, jsize};
use jni::JNIEnv;

use crate::ndarray::{as_shape, return_handle};
use crate::{cast_handle, to_handle};

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_flatten<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.flatten_all();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_flattenWithDims<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    start_dim: jint,
    end_dim: jint,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.flatten(start_dim as usize, end_dim as usize);
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_reshape<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    shape: JLongArray<'local>,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let shape = unsafe { env.get_array_elements(&shape, ReleaseMode::NoCopyBack) }.unwrap();
    let dims = shape
        .into_iter()
        .map(|i| *i as usize)
        .collect::<Vec<usize>>();
    let ret = tensor.reshape(dims);
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_squeeze<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    dims: JIntArray<'local>,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let shape = tensor.shape();
    let mut squeeze = || {
        let mut shape = Vec::from(shape.dims());
        let dims = unsafe { env.get_array_elements(&dims, ReleaseMode::NoCopyBack) }.unwrap();
        for i in dims.iter().rev() {
            let mut pos = *i as i32;
            if pos < 0 {
                pos = shape.len() as i32 + pos;
            }
            if shape[pos as usize] == 1 {
                shape.remove(pos as usize);
            }
        }
        tensor.reshape(shape)
    };
    if shape.rank() == 0 {
        return_handle(&mut env, tensor.copy())
    } else {
        let ret = squeeze();
        return_handle(&mut env, ret)
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_expandDims<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    axis: jint,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = if axis == -1 {
        tensor.unsqueeze(D::Minus1)
    } else {
        tensor.unsqueeze(axis as usize)
    };
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_stack<'local>(
    mut env: JNIEnv,
    _: JObject,
    handles: JLongArray<'local>,
    axis: jint,
) -> jlong {
    let handles = unsafe { env.get_array_elements(&handles, ReleaseMode::NoCopyBack) }.unwrap();
    let tensors = handles
        .into_iter()
        .map(|h| cast_handle::<Tensor>(*h))
        .collect::<Vec<&mut Tensor>>();
    let ret = Tensor::stack(&tensors, axis as usize);
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_concat<'local>(
    mut env: JNIEnv,
    _: JObject,
    handles: JLongArray<'local>,
    axis: jint,
) -> jlong {
    let handles = unsafe { env.get_array_elements(&handles, ReleaseMode::NoCopyBack) }.unwrap();
    let tensors = handles
        .into_iter()
        .map(|h| cast_handle::<Tensor>(*h))
        .collect::<Vec<&mut Tensor>>();
    let ret = Tensor::cat(&tensors, axis as usize);
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_split<'local>(
    mut env: JNIEnv<'local>,
    _: JObject,
    handle: jlong,
    indices: JLongArray<'local>,
    axis: jint,
) -> JLongArray<'local> {
    let tensor = cast_handle::<Tensor>(handle);
    let indices = unsafe { env.get_array_elements(&indices, ReleaseMode::NoCopyBack) }.unwrap();
    let mut array: Vec<jlong> = Vec::new();
    let mut prev = 0;
    for i in indices.into_iter() {
        let len = *i as usize - prev;
        if len > 0 {
            let slice = tensor.narrow(axis as usize, prev, len).unwrap();
            array.push(to_handle(slice));
        }
        prev = *i as usize;
    }

    let ret = env.new_long_array(array.len() as jsize).unwrap();
    env.set_long_array_region(&ret, 0, &array).unwrap();
    ret
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_cumSum<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    axis: jint,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.cumsum(axis as usize);
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_clip<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    min: jdouble,
    max: jdouble,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.clamp(min as f64, max as f64);
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_transpose<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    dim1: jint,
    dim2: jint,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.transpose(dim1 as usize, dim2 as usize);
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_permute<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    axes: JIntArray<'local>,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let axes = unsafe { env.get_array_elements(&axes, ReleaseMode::NoCopyBack) }.unwrap();
    let dims = axes
        .into_iter()
        .map(|i| *i as usize)
        .collect::<Vec<usize>>();

    let ret = tensor.permute(dims);
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_broadcast<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    shape: JLongArray<'local>,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let shape = as_shape(&mut env, &shape);
    let ret = tensor.broadcast_as(shape);
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_avgPool2d<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    kernel_size: JLongArray<'local>,
    stride: JLongArray<'local>,
) -> jlong {
    let mut op = || {
        let tensor = cast_handle::<Tensor>(handle);
        let kernel_size = as_shape(&mut env, &kernel_size).dims2()?;
        let stride = as_shape(&mut env, &stride).dims2()?;
        tensor.avg_pool2d_with_stride(kernel_size, stride)
    };
    let ret = op();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_maxPool2d<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    kernel_size: JLongArray<'local>,
    stride: JLongArray<'local>,
) -> jlong {
    let mut op = || {
        let tensor = cast_handle::<Tensor>(handle);
        let kernel_size = as_shape(&mut env, &kernel_size).dims2()?;
        let stride = as_shape(&mut env, &stride).dims2()?;
        tensor.max_pool2d_with_stride(kernel_size, stride)
    };
    let ret = op();
    return_handle(&mut env, ret)
}
