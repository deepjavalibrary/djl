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
#include <torch/torch.h>

#include "../build/include/ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_error.h"
#include "djl_pytorch_jni_utils.h"


JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNLinear(
  JNIEnv* env, jobject jthis, jobject jhandle, jobject jweight, jobject jbias, jboolean bias) {
  API_BEGIN();
  auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  auto* weight_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jweight);
  if (bias) {
    auto* bias_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jbias);
    const auto* result_ptr = new torch::Tensor(torch::nn::functional::linear(*tensor_ptr, *weight_ptr, *bias_ptr));
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  }
  const auto* result_ptr = new torch::Tensor(torch::nn::functional::linear(*tensor_ptr, *weight_ptr));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNRelu(
  JNIEnv *env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->relu());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}
