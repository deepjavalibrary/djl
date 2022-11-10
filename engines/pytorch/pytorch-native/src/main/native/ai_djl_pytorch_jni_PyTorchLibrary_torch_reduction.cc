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

// The file is the implementation for PyTorch tensor reduction ops

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchArgMax__J(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->argmax());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchArgMax__JJZ(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jdim, jboolean jkeep_dim) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->argmax(jdim, jkeep_dim == JNI_TRUE));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchArgMin__J(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->argmin());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchArgMin__JJZ(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jdim, jboolean jkeep_dim) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->argmin(jdim, jkeep_dim == JNI_TRUE));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchArgSort(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jdim, jboolean jkeep_dim) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->argsort(jdim, jkeep_dim == JNI_TRUE));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMax__J(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->max());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMax__JJZ(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jdim, jboolean jkeep_dim) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(std::get<0>(tensor_ptr->max(jdim, jkeep_dim == JNI_TRUE)));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMean__J(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->mean());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMean__JJZ(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jdim, jboolean jkeep_dim) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->mean(jdim, jkeep_dim == JNI_TRUE));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMin__J(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->min());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMin__JJZ(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jdim, jboolean jkeep_dim) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(std::get<0>(tensor_ptr->min(jdim, jkeep_dim == JNI_TRUE)));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSum__J(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->sum());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSum__J_3JZ(
    JNIEnv* env, jobject jthis, jlong jhandle, jlongArray jdims, jboolean jkeep_dim) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const std::vector<int64_t> dims = djl::utils::jni::GetVecFromJLongArray(env, jdims);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->sum(dims, jkeep_dim == JNI_TRUE));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCumProd
  (JNIEnv * env, jobject jthis, jlong jhandle, jlong jdim, jint jdtype) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  if (jdtype == -1) {
      const auto* result_ptr = new torch::Tensor(tensor_ptr->cumprod(jdim));
      return reinterpret_cast<uintptr_t>(result_ptr);
  } else {
      const auto* result_ptr = new torch::Tensor(tensor_ptr->cumprod(jdim, utils::GetScalarTypeFromDType(jdtype)));
      return reinterpret_cast<uintptr_t>(result_ptr);
  }
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchProd__J(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->prod());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchProd__JJZ(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jdim, jboolean jkeep_dim) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->prod(jdim, jkeep_dim == JNI_TRUE));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCumSum(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong dim) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->cumsum(dim));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNorm(
    JNIEnv* env, jobject jthis, jlong jhandle, jint jord, jlongArray jaxes, jboolean jkeep_dims) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const std::vector<int64_t> axes = djl::utils::jni::GetVecFromJLongArray(env, jaxes);
  const auto* result_ptr =
      new torch::Tensor(tensor_ptr->norm(c10::make_optional(c10::Scalar(jord)), axes, jkeep_dims == JNI_TRUE));

  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}