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
#include <djl/utils.h>

#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_exception.h"
#include "djl_pytorch_utils.h"

// The file is the implementation for PyTorch tensor creation ops

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchFromBlob(JNIEnv* env, jobject jthis,
    jobject jbuffer, jlongArray jshape, jint jdtype, jint jlayout, jintArray jdevice, jboolean jrequired_grad) {
  API_BEGIN()
  const auto shape_vec = djl::utils::jni::GetVecFromJLongArray(env, jshape);
  const auto device = utils::GetDeviceFromJDevice(env, jdevice);
  auto options = torch::TensorOptions().requires_grad(JNI_TRUE == jrequired_grad);
  // DJL's UNKNOWN type
  if (jdtype != 8) {
    options = options.dtype(utils::GetScalarTypeFromDType(jdtype));
  }
  // the java side hold the reference for the directByteBuffer
  torch::Tensor result = torch::from_blob(env->GetDirectBufferAddress(jbuffer), shape_vec, options);
  // from_blob doesn't support torch::kSparse and torch::kMkldnn, so explicit cast the type here
  if (jlayout == 1) {
    result = result.to_sparse();
  } else if (jlayout == 2) {
    result = result.to_mkldnn();
  }
  // Don't change device unless data on CPU
  if (!device.is_cpu()) {
    result = result.to(device);
  }
  const torch::Tensor* tensor_ptr = new torch::Tensor(result);
  return reinterpret_cast<uintptr_t>(tensor_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchEmpty(JNIEnv* env, jobject jthis, jlongArray jshape,
    jint jdtype, jint jlayout, jintArray jdevice, jboolean jrequired_grad) {
  API_BEGIN()
  const auto shape_vec = djl::utils::jni::GetVecFromJLongArray(env, jshape);
  const auto options = utils::CreateTensorOptions(env, jdtype, jlayout, jdevice, jrequired_grad);
  const torch::Tensor* tensor_ptr = new torch::Tensor(
      (jlayout == 2) ? torch::empty(shape_vec, options).to_mkldnn() : torch::empty(shape_vec, options));
  return reinterpret_cast<uintptr_t>(tensor_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchZeros(JNIEnv* env, jobject jthis, jlongArray jshape,
    jint jdtype, jint jlayout, jintArray jdevice, jboolean jrequired_grad) {
  API_BEGIN()
  const auto shape_vec = djl::utils::jni::GetVecFromJLongArray(env, jshape);
  const auto options = utils::CreateTensorOptions(env, jdtype, jlayout, jdevice, jrequired_grad);
  const torch::Tensor* tensor_ptr = new torch::Tensor(
      (jlayout == 2) ? torch::zeros(shape_vec, options).to_mkldnn() : torch::zeros(shape_vec, options));
  return reinterpret_cast<uintptr_t>(tensor_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchOnes(JNIEnv* env, jobject jthis, jlongArray jshape,
    jint jdtype, jint jlayout, jintArray jdevice, jboolean jrequired_grad) {
  API_BEGIN()
  const auto shape_vec = djl::utils::jni::GetVecFromJLongArray(env, jshape);
  const auto options = utils::CreateTensorOptions(env, jdtype, jlayout, jdevice, jrequired_grad);
  const torch::Tensor* tensor_ptr =
      new torch::Tensor((jlayout == 2) ? torch::ones(shape_vec, options).to_mkldnn() : torch::ones(shape_vec, options));
  return reinterpret_cast<uintptr_t>(tensor_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchFull(JNIEnv* env, jobject jthis, jlongArray jshape,
    jdouble jfill_value, jint jdtype, jint jlayout, jintArray jdevice, jboolean jrequired_grad) {
  API_BEGIN()
  const auto shape_vec = djl::utils::jni::GetVecFromJLongArray(env, jshape);
  const auto options = utils::CreateTensorOptions(env, jdtype, jlayout, jdevice, jrequired_grad);
  const torch::Tensor* tensor_ptr =
      new torch::Tensor((jlayout == 2) ? torch::full(shape_vec, jfill_value, options).to_mkldnn()
                                       : torch::full(shape_vec, jfill_value, options));
  return reinterpret_cast<uintptr_t>(tensor_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchZerosLike(
    JNIEnv* env, jobject jthis, jlong jhandle, jint jdtype, jint jlayout, jintArray jdevice, jboolean jrequired_grad) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto options = utils::CreateTensorOptions(env, jdtype, jlayout, jdevice, jrequired_grad);
  const torch::Tensor* result_ptr = new torch::Tensor(
      (jlayout == 2) ? torch::zeros_like(*tensor_ptr, options).to_mkldnn() : torch::zeros_like(*tensor_ptr, options));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchOnesLike(
    JNIEnv* env, jobject jthis, jlong jhandle, jint jdtype, jint jlayout, jintArray jdevice, jboolean jrequired_grad) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto options = utils::CreateTensorOptions(env, jdtype, jlayout, jdevice, jrequired_grad);
  const torch::Tensor* result_ptr = new torch::Tensor(
      (jlayout == 2) ? torch::ones_like(*tensor_ptr, options).to_mkldnn() : torch::ones_like(*tensor_ptr, options));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchArange(JNIEnv* env, jobject jthis, jfloat jstart,
    jfloat jend, jfloat jstep, jint jdtype, jint jlayout, jintArray jdevice, jboolean jrequired_grad) {
  API_BEGIN()
  const auto options = utils::CreateTensorOptions(env, jdtype, jlayout, jdevice, jrequired_grad);
  const torch::Tensor* tensor_ptr =
      new torch::Tensor((jlayout == 2) ? torch::arange(jstart, jend, jstep, options).to_mkldnn()
                                       : torch::arange(jstart, jend, jstep, options));
  return reinterpret_cast<uintptr_t>(tensor_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLinspace(JNIEnv* env, jobject jthis, jfloat jstart,
    jfloat jend, jint jstep, jint jdtype, jint jlayout, jintArray jdevice, jboolean jrequired_grad) {
  API_BEGIN()
  const auto options = utils::CreateTensorOptions(env, jdtype, jlayout, jdevice, jrequired_grad);
  const torch::Tensor* tensor_ptr =
      new torch::Tensor((jlayout == 2) ? torch::linspace(jstart, jend, jstep, options).to_mkldnn()
                                       : torch::linspace(jstart, jend, jstep, options));
  return reinterpret_cast<uintptr_t>(tensor_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchEye(JNIEnv* env, jobject jthis, jint jn, jint jm,
    jint jdtype, jint jlayout, jintArray jdevice, jboolean jrequired_grad) {
  API_BEGIN()
  const auto options = utils::CreateTensorOptions(env, jdtype, jlayout, jdevice, jrequired_grad);
  const torch::Tensor* tensor_ptr =
      new torch::Tensor((jlayout == 2) ? torch::eye(jn, jm, options).to_mkldnn() : torch::eye(jn, jm, options));
  return reinterpret_cast<uintptr_t>(tensor_ptr);
  API_END_RETURN();
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSparseCoo(
    JNIEnv* env, jobject jthis, jlongArray jshape, jlong jindices, jlong jvalue, jboolean jrequired_grad) {
  API_BEGIN()
  const auto* indices_ptr = reinterpret_cast<torch::Tensor*>(jindices);
  const auto* val_ptr = reinterpret_cast<torch::Tensor*>(jvalue);
  const auto shape_vec = djl::utils::jni::GetVecFromJLongArray(env, jshape);
  const auto options = torch::TensorOptions()
                           .device(val_ptr->device())
                           .layout(torch::kSparse)
                           .requires_grad(JNI_TRUE == jrequired_grad)
                           .dtype(val_ptr->dtype());
  const auto* result_ptr = new torch::Tensor(torch::sparse_coo_tensor(*indices_ptr, *val_ptr, shape_vec, options));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN();
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchHannWindow(
    JNIEnv* env, jobject jthis, jlong jn_fft, jboolean jperoidic, jintArray jdevice) {
  API_BEGIN()
  const auto device = utils::GetDeviceFromJDevice(env, jdevice);
  auto dtype = c10::optional<c10::ScalarType>(torch::kFloat32);
  auto options = torch::TensorOptions().device(device).dtype(dtype);
  const auto* result_ptr = new torch::Tensor(torch::hann_window(jn_fft, jperoidic, options));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}
