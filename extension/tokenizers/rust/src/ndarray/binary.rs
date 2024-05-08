use candle_core::Tensor;
use jni::objects::JObject;
use jni::sys::jlong;
use jni::JNIEnv;

use crate::cast_handle;
use crate::ndarray::return_handle;

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_add<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    other_handle: jlong,
) -> jlong {
    let op = || {
        let lhs = cast_handle::<Tensor>(handle);
        let rhs = cast_handle::<Tensor>(other_handle).to_dtype(lhs.dtype())?;
        lhs.broadcast_add(&rhs)
    };
    let ret = op();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_sub<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    other_handle: jlong,
) -> jlong {
    let op = || {
        let lhs = cast_handle::<Tensor>(handle);
        let rhs = cast_handle::<Tensor>(other_handle).to_dtype(lhs.dtype())?;
        lhs.broadcast_sub(&rhs)
    };
    let ret = op();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_mul<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    other_handle: jlong,
) -> jlong {
    let op = || {
        let lhs = cast_handle::<Tensor>(handle);
        let rhs = cast_handle::<Tensor>(other_handle).to_dtype(lhs.dtype())?;
        lhs.broadcast_mul(&rhs)
    };
    let ret = op();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_div<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    other_handle: jlong,
) -> jlong {
    let op = || {
        let lhs = cast_handle::<Tensor>(handle);
        let rhs = cast_handle::<Tensor>(other_handle).to_dtype(lhs.dtype())?;
        lhs.broadcast_div(&rhs)
    };
    let ret = op();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_maximum<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    other_handle: jlong,
) -> jlong {
    let op = || {
        let lhs = cast_handle::<Tensor>(handle);
        let rhs = cast_handle::<Tensor>(other_handle).to_dtype(lhs.dtype())?;
        lhs.broadcast_maximum(&rhs)
    };
    let ret = op();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_minimum<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    other_handle: jlong,
) -> jlong {
    let op = || {
        let lhs = cast_handle::<Tensor>(handle);
        let rhs = cast_handle::<Tensor>(other_handle).to_dtype(lhs.dtype())?;
        lhs.broadcast_minimum(&rhs)
    };
    let ret = op();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_pow<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    other_handle: jlong,
) -> jlong {
    let op = || {
        let lhs = cast_handle::<Tensor>(handle);
        let rhs = cast_handle::<Tensor>(other_handle).to_dtype(lhs.dtype())?;
        lhs.broadcast_pow(&rhs)
    };
    let ret = op();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_matmul<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    other_handle: jlong,
) -> jlong {
    let op = || {
        let lhs = cast_handle::<Tensor>(handle);
        let rhs = cast_handle::<Tensor>(other_handle).to_dtype(lhs.dtype())?;
        lhs.broadcast_matmul(&rhs)
    };
    let ret = op();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_batchMatMul<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    other_handle: jlong,
) -> jlong {
    let op = || {
        let lhs = cast_handle::<Tensor>(handle);
        let rhs = cast_handle::<Tensor>(other_handle).to_dtype(lhs.dtype())?;
        lhs.matmul(&rhs)
    };
    let ret = op();
    return_handle(&mut env, ret)
}
