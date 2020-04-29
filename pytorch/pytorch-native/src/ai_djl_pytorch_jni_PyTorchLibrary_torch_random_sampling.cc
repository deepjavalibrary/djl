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
#include <torch/script.h>
#include <torch/torch.h>

#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_error.h"
#include "djl_pytorch_jni_utils.h"

// The file is the implementation for PyTorch random sampling operations

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_atNormal(JNIEnv* env, jobject jthis, jdouble jmean,
    jdouble jstd, jlongArray jsizes, jint jdtype, jint jlayout, jintArray jdevice, jboolean jrequire_grad) {
  API_BEGIN();
  const std::vector<int64_t> size_vec = utils::GetVecFromJLongArray(env, jsizes);
  const auto options = utils::CreateTensorOptions(env, jdtype, jlayout, jdevice, jrequire_grad);
  const auto* result_ptr = new torch::Tensor(torch::normal(jmean, jstd, size_vec, nullptr, options));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_tensorUniform(JNIEnv* env, jobject jthis,
    jdouble jfrom, jdouble jto, jlongArray jsizes, jint jdtype, jint jlayout, jintArray jdevice,
    jboolean jrequire_grad) {
  API_BEGIN();
  const std::vector<int64_t> size_vec = utils::GetVecFromJLongArray(env, jsizes);
  const auto options = utils::CreateTensorOptions(env, jdtype, jlayout, jdevice, jrequire_grad);
  const auto* result_ptr = new torch::Tensor((torch::empty(size_vec, options).uniform_(jfrom, jto)));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}
