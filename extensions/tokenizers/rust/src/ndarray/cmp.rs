use candle_core::{DType, Tensor};
use jni::objects::JObject;
use jni::sys::{jboolean, jlong, JNI_FALSE, JNI_TRUE};
use jni::JNIEnv;

use crate::cast_handle;
use crate::ndarray::return_handle;

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_eq<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    other_handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let other = cast_handle::<Tensor>(other_handle);
    let ret = tensor.broadcast_eq(&*other);
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_neq<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    other_handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let other = cast_handle::<Tensor>(other_handle);
    let ret = tensor.broadcast_ne(&*other);
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_gt<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    other_handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let other = cast_handle::<Tensor>(other_handle);
    let ret = tensor.broadcast_gt(&*other);
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_gte<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    other_handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let other = cast_handle::<Tensor>(other_handle);
    let ret = tensor.broadcast_ge(&*other);
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_lt<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    other_handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let other = cast_handle::<Tensor>(other_handle);
    let ret = tensor.broadcast_lt(&*other);
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_lte<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    other_handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let other = cast_handle::<Tensor>(other_handle);
    let ret = tensor.broadcast_le(&*other);
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_contentEqual<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    other_handle: jlong,
) -> jboolean {
    let tensor = cast_handle::<Tensor>(handle);
    let size = tensor.shape().elem_count();
    let cmp = || {
        let other = cast_handle::<Tensor>(other_handle);
        let sum = tensor.eq(&*other)?.sum_all()?;
        sum.to_dtype(DType::U32)?.to_scalar::<u32>()
    };
    let value = cmp();
    match value {
        Ok(v) => {
            if v as usize == size {
                JNI_TRUE
            } else {
                JNI_FALSE
            }
        }
        Err(err) => {
            env.throw_new("ai/djl/engine/EngineException", err.to_string())
                .unwrap();
            JNI_FALSE
        }
    }
}
