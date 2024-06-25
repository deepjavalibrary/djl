use candle::Tensor;
use jni::objects::JObject;
use jni::sys::jlong;
use jni::JNIEnv;

use crate::cast_handle;
use crate::ndarray::return_handle;

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_exp<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.exp();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_log<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.log();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_sin<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.sin();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_cos<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.cos();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_tanh<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.tanh();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_abs<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.abs();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_neg<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.neg();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_square<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.sqr();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_sqrt<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.sqrt();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_floor<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.floor();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_ceil<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.ceil();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_round<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.round();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_gelu<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.gelu();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_relu<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.relu();
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_erf<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = tensor.erf();
    return_handle(&mut env, ret)
}
