use crate::cast_handle;
use crate::ndarray::return_handle;
use candle::Tensor;
use jni::objects::JObject;
use jni::sys::{jfloat, jint, jlong};
use jni::JNIEnv;

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_sigmoid<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = candle_nn::ops::sigmoid(&tensor);
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_leakyRelu<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    alpha: jfloat,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = candle_nn::ops::leaky_relu(&tensor, alpha as f64);
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_softmax<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    axis: jint,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = candle_nn::ops::softmax(&tensor, axis as usize);
    return_handle(&mut env, ret)
}

#[no_mangle]
pub extern "system" fn Java_ai_djl_engine_rust_RustLibrary_logSoftmax<'local>(
    mut env: JNIEnv,
    _: JObject,
    handle: jlong,
    axis: jint,
) -> jlong {
    let tensor = cast_handle::<Tensor>(handle);
    let ret = candle_nn::ops::log_softmax(&tensor, axis as usize);
    return_handle(&mut env, ret)
}
