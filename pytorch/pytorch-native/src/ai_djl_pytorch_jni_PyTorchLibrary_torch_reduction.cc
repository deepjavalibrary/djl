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
#include "djl_pytorch_jni_error.h"
#include "djl_pytorch_jni_utils.h"

// The file is the implementation for PyTorch tensor reduction ops

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchArgMax__Lai_djl_pytorch_jni_Pointer_2(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->argmax());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchArgMax__Lai_djl_pytorch_jni_Pointer_2JZ(
    JNIEnv* env, jobject jthis, jobject jhandle, jlong jdim, jboolean jkeep_dim) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->argmax(jdim, jkeep_dim == JNI_TRUE));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchArgMin__Lai_djl_pytorch_jni_Pointer_2(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->argmin());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchArgMin__Lai_djl_pytorch_jni_Pointer_2JZ(
    JNIEnv* env, jobject jthis, jobject jhandle, jlong jdim, jboolean jkeep_dim) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->argmin(jdim, jkeep_dim == JNI_TRUE));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchArgSort__Lai_djl_pytorch_jni_Pointer_2(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->argsort());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchArgSort__Lai_djl_pytorch_jni_Pointer_2JZ(
    JNIEnv* env, jobject jthis, jobject jhandle, jlong jdim, jboolean jkeep_dim) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->argsort(jdim, jkeep_dim == JNI_TRUE));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMax__Lai_djl_pytorch_jni_Pointer_2(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->max());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMax__Lai_djl_pytorch_jni_Pointer_2JZ(
    JNIEnv* env, jobject jthis, jobject jhandle, jlong jdim, jboolean jkeep_dim) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(std::get<0>(tensor_ptr->max(jdim, jkeep_dim == JNI_TRUE)));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMean__Lai_djl_pytorch_jni_Pointer_2(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->mean());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMean__Lai_djl_pytorch_jni_Pointer_2JZ(
    JNIEnv* env, jobject jthis, jobject jhandle, jlong jdim, jboolean jkeep_dim) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->mean(jdim, jkeep_dim == JNI_TRUE));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMin__Lai_djl_pytorch_jni_Pointer_2(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->min());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMin__Lai_djl_pytorch_jni_Pointer_2JZ(
    JNIEnv* env, jobject jthis, jobject jhandle, jlong jdim, jboolean jkeep_dim) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(std::get<0>(tensor_ptr->min(jdim, jkeep_dim == JNI_TRUE)));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSum__Lai_djl_pytorch_jni_Pointer_2(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->sum());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSum__Lai_djl_pytorch_jni_Pointer_2_3JZ(
    JNIEnv* env, jobject jthis, jobject jhandle, jlongArray jdims, jboolean jkeep_dim) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const std::vector<int64_t> dims = utils::GetVecFromJLongArray(env, jdims);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->sum(dims, jkeep_dim == JNI_TRUE));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchProd__Lai_djl_pytorch_jni_Pointer_2(
  JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
    const auto* result_ptr = new torch::Tensor(tensor_ptr->prod());
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchProd__Lai_djl_pytorch_jni_Pointer_2JZ
  (JNIEnv* env, jobject jthis, jobject jhandle, jlong jdim, jboolean jkeep_dim) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
    const auto* result_ptr = new torch::Tensor(tensor_ptr->prod(jdim, jkeep_dim == JNI_TRUE));
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCumSum(
  JNIEnv* env, jobject jthis, jobject jhandle, jlong dim) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
    const auto* result_ptr = new torch::Tensor(tensor_ptr->cumsum(dim));
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}