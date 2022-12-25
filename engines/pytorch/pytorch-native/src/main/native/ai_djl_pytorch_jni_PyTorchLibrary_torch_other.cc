/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_exception.h"
#include "djl_pytorch_utils.h"

// The file is the implementation for PyTorch tensor other ops

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchFlatten(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jstart_dim, jlong jend_dim) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->flatten(jstart_dim, jend_dim));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchFft(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jn, jlong jaxis) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(torch::fft_fft(*tensor_ptr, jn, jaxis));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchStft(JNIEnv* env, jobject jthis, jlong jhandle,
    jlong jn_fft, jlong jhop_length, jlong jwindow, jboolean jcenter, jboolean jnormalize, jboolean jreturn_complex) {
#ifdef V1_11_X
  return -1;
#else
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  if (jwindow == -1L) {
    const auto* result_ptr = new torch::Tensor(tensor_ptr->stft(jn_fft, jhop_length, c10::nullopt, c10::nullopt,
        jcenter, "reflect", jnormalize, c10::nullopt, jreturn_complex));
    return reinterpret_cast<uintptr_t>(result_ptr);
  } else {
    const auto* window_ptr = reinterpret_cast<torch::Tensor*>(jwindow);
    const auto* result_ptr = new torch::Tensor(tensor_ptr->stft(
        jn_fft, jhop_length, c10::nullopt, *window_ptr, jcenter, "reflect", jnormalize, c10::nullopt, jreturn_complex));
    return reinterpret_cast<uintptr_t>(result_ptr);
  }
  API_END_RETURN()
#endif
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchViewAsReal(
    JNIEnv* env, jobject jthis, jlong jhandle) {
#ifdef V1_11_X
  return -1;
#else
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(torch::view_as_real_copy(*tensor_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
#endif
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchViewAsComplex(
    JNIEnv* env, jobject jthis, jlong jhandle) {
#ifdef V1_11_X
  return -1;
#else
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(torch::view_as_complex_copy(*tensor_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
#endif
}
