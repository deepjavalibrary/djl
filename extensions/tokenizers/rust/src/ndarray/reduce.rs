use candle::{DType, Tensor};
use jni::objects::{JIntArray, JObject, ReleaseMode};
use jni::sys::{jboolean, jdouble, jint, jlong, JNI_TRUE};
use jni::JNIEnv;

use crate::cast_handle;
use crate::ndarray::return_handle;

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_sum<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let dtype = tensor.dtype();
    let ret = if dtype.is_int() {
        tensor.to_dtype(DType::I64).unwrap().sum_all()
    } else {
        tensor.sum_all()
    };
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_sumWithAxis<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    axes: JIntArray<'local>,
    keep_dims: jboolean,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let rank = tensor.shape().rank() as i32;
    let axes = unsafe { env.get_array_elements(&axes, ReleaseMode::NoCopyBack) }.unwrap();
    let dims = axes
        .into_iter()
        .map(|i| {
            let mut dim = *i as i32;
            if dim < 0 {
                dim = rank + dim;
            }
            return dim as usize;
        })
        .collect::<Vec<usize>>();

    let ret = if keep_dims == JNI_TRUE {
        if tensor.dtype().is_int() {
            tensor.to_dtype(DType::I64).unwrap().sum_keepdim(dims)
        } else {
            tensor.sum_keepdim(dims)
        }
    } else {
        if tensor.dtype().is_int() {
            tensor.to_dtype(DType::I64).unwrap().sum(dims)
        } else {
            tensor.sum(dims)
        }
    };
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_mean<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.mean_all();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_meanWithAxis<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    axes: JIntArray<'local>,
    keep_dims: jboolean,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let axes = unsafe { env.get_array_elements(&axes, ReleaseMode::NoCopyBack) }.unwrap();
    let dims = axes
        .into_iter()
        .map(|i| *i as usize)
        .collect::<Vec<usize>>();

    let ret = if keep_dims == JNI_TRUE {
        tensor.mean_keepdim(dims)
    } else {
        tensor.mean(dims)
    };
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_min<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let min = || {
        let tensor = cast_handle::<Tensor>(handle);
        tensor.flatten_all()?.min(0usize)
    };
    let ret = min();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_minWithAxis<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    axis: jint,
    keep_dims: jboolean,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = if keep_dims == JNI_TRUE {
        tensor.min_keepdim(axis as usize)
    } else {
        tensor.min(axis as usize)
    };
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_max<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let max = || {
        let tensor = cast_handle::<Tensor>(handle);
        tensor.flatten_all()?.max(0usize)
    };
    let ret = max();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_maxWithAxis<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    axis: jint,
    keep_dims: jboolean,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = if keep_dims == JNI_TRUE {
        tensor.max_keepdim(axis as usize)
    } else {
        tensor.max(axis as usize)
    };
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_argMin<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let argmin = || {
        let tensor = cast_handle::<Tensor>(handle);
        tensor.flatten_all()?.argmin(0usize)?.to_dtype(DType::I64)
    };
    let ret = argmin();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_argMinWithAxis<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    axis: jint,
    keep_dims: jboolean,
) -> jlong {
    let argmin = || {
        let tensor = cast_handle::<Tensor>(handle);
        let tensor = if keep_dims == JNI_TRUE {
            tensor.argmin_keepdim(axis as usize)
        } else {
            tensor.argmin(axis as usize)
        };
        tensor?.to_dtype(DType::I64)
    };
    let ret = argmin();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_argMax<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let argmax = || {
        let tensor = cast_handle::<Tensor>(handle);
        tensor.flatten_all()?.argmax(0usize)?.to_dtype(DType::I64)
    };
    let ret = argmax();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_argMaxWithAxis<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    axis: jint,
    keep_dims: jboolean,
) -> jlong {
    let argmax = || {
        let tensor = cast_handle::<Tensor>(handle);
        let tensor = if keep_dims == JNI_TRUE {
            tensor.argmax_keepdim(axis as usize)
        } else {
            tensor.argmax(axis as usize)
        };
        tensor?.to_dtype(DType::I64)
    };
    let ret = argmax();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_normalize<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    p: jdouble,
    dim: jint,
    eps: jdouble,
) -> jlong {
    let normalize = || {
        let tensor = cast_handle::<Tensor>(handle);
        let device = tensor.device();
        let rank = tensor.shape().rank() as i32;
        let dim: i32 = if dim < 0 { rank + dim } else { dim };
        let pow = Tensor::new(vec![p as f64], device)?.to_dtype(tensor.dtype())?;
        let root = Tensor::new(vec![1f64 / p as f64], device)?.to_dtype(tensor.dtype())?;
        let eps = Tensor::new(vec![eps as f64], device)?.to_dtype(tensor.dtype())?;
        let sum = if p as u32 % 2 == 0 {
            tensor.abs()?.broadcast_pow(&pow)?.sum_keepdim(dim as usize)
        } else {
            tensor.broadcast_pow(&pow)?.sum_keepdim(dim as usize)
        };
        let norm = sum?.broadcast_pow(&root)?.broadcast_maximum(&eps)?;
        tensor.broadcast_div(&norm)
    };
    let ret = normalize();
    return_handle(&mut env, ret)
}
