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

// The file is the implementation for PyTorch tensor comparison ops

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_contentEqual(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  auto tensor_ptr1 = reinterpret_cast<torch::Tensor*>(jself);
  auto tensor_ptr2 = reinterpret_cast<torch::Tensor*>(jother);
  return tensor_ptr1->equal(*tensor_ptr2);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchEq(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  auto tensor_ptr1 = reinterpret_cast<torch::Tensor*>(jself);
  auto tensor_ptr2 = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(tensor_ptr1->eq(*tensor_ptr2));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNeq(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  auto tensor_ptr1 = reinterpret_cast<torch::Tensor*>(jself);
  auto tensor_ptr2 = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(tensor_ptr1->eq(*tensor_ptr2).logical_not_());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchGt(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  auto tensor_ptr1 = reinterpret_cast<torch::Tensor*>(jself);
  auto tensor_ptr2 = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(tensor_ptr1->gt(*tensor_ptr2));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchGte(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  auto tensor_ptr1 = reinterpret_cast<torch::Tensor*>(jself);
  auto tensor_ptr2 = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(tensor_ptr1->ge(*tensor_ptr2));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLt(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  auto tensor_ptr1 = reinterpret_cast<torch::Tensor*>(jself);
  auto tensor_ptr2 = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(tensor_ptr1->lt(*tensor_ptr2));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLte(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  auto tensor_ptr1 = reinterpret_cast<torch::Tensor*>(jself);
  auto tensor_ptr2 = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(tensor_ptr1->le(*tensor_ptr2));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSort(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jdim, jboolean jdescending) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(std::get<0>(tensor_ptr->sort(jdim, jdescending == JNI_TRUE)));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchIsNaN(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(torch::isnan(*tensor_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchIsInf(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(torch::isinf(*tensor_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}
