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
#include <torch/script.h>
#include <torch/torch.h>

#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_exception.h"
#include "djl_pytorch_utils.h"

// The file is the implementation for PyTorch random sampling operations

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchRandint(JNIEnv* env, jobject jthis, jlong jlow,
    jlong jhigh, jlongArray jsizes, jint jdtype, jint jlayout, jintArray jdevice, jboolean jrequire_grad) {
  API_BEGIN()
  const std::vector<int64_t> size_vec = djl::utils::jni::GetVecFromJLongArray(env, jsizes);
  const auto options = utils::CreateTensorOptions(env, jdtype, jlayout, jdevice, jrequire_grad);
  torch::Tensor tensor = torch::randint(jlow, jhigh, size_vec, torch::nullopt, options);
  // Tensor Option for mkldnn is not working
  // explicitly convert to mkldnn
  if (jlayout == 2) {
    tensor = tensor.to_mkldnn();
  }
  const auto* result_ptr = new torch::Tensor(tensor);
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchRandPerm(
    JNIEnv* env, jobject jthis, jlong jn, jint jdtype, jint jlayout, jintArray jdevice, jboolean jrequire_grad) {
  API_BEGIN()
  const auto options = utils::CreateTensorOptions(env, jdtype, jlayout, jdevice, jrequire_grad);
  torch::Tensor tensor = torch::randperm(jn, options);
  // Tensor Option for mkldnn is not working
  // explicitly convert to mkldnn
  if (jlayout == 2) {
    tensor = tensor.to_mkldnn();
  }
  const auto* result_ptr = new torch::Tensor(tensor);
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNormal(JNIEnv* env, jobject jthis, jdouble jmean,
    jdouble jstd, jlongArray jsizes, jint jdtype, jint jlayout, jintArray jdevice, jboolean jrequire_grad) {
  API_BEGIN()
  const std::vector<int64_t> size_vec = djl::utils::jni::GetVecFromJLongArray(env, jsizes);
  const auto options = utils::CreateTensorOptions(env, jdtype, jlayout, jdevice, jrequire_grad);
  torch::Tensor tensor = torch::normal(jmean, jstd, size_vec, torch::nullopt, options);
  // Tensor Option for mkldnn is not working
  // explicitly convert to mkldnn
  if (jlayout == 2) {
    tensor = tensor.to_mkldnn();
  }
  const auto* result_ptr = new torch::Tensor(tensor);
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_tensorUniform(JNIEnv* env, jobject jthis, jdouble jfrom,
    jdouble jto, jlongArray jsizes, jint jdtype, jint jlayout, jintArray jdevice, jboolean jrequire_grad) {
  API_BEGIN()
  const std::vector<int64_t> size_vec = djl::utils::jni::GetVecFromJLongArray(env, jsizes);
  const auto options = utils::CreateTensorOptions(env, jdtype, jlayout, jdevice, jrequire_grad);
  torch::Tensor tensor = torch::empty(size_vec, options).uniform_(jfrom, jto);
  // Tensor Option for mkldnn is not working
  // explicitly convert to mkldnn
  if (jlayout == 2) {
    tensor = tensor.to_mkldnn();
  }
  const auto* result_ptr = new torch::Tensor(tensor);
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}
