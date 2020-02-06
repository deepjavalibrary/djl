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
#include "../build/include/ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_utils.h"

// The file is the implementation for PyTorch neural network functional ops

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSoftmax
  (JNIEnv* env, jobject jthis, jobject jhandle, jlong jdim, jint jdtype) {
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->softmax(jdim, utils::GetScalarTypeFromDType(jdtype)));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchUpsampleBilinear2d
  (JNIEnv* env, jobject jthis, jobject jhandle, jlongArray jsize, jboolean jalign_corners) {
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto size_vec = utils::GetVecFromJLongArray(env, jsize);
  const auto* result_ptr = new torch::Tensor(
    torch::upsample_bilinear2d(*tensor_ptr, size_vec, jalign_corners == JNI_TRUE));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_normalize
  (JNIEnv* env, jobject jthis, jobject jhandle, jfloatArray jmean, jfloatArray jstd) {
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  std::vector<float> mean = utils::GetVecFromJFloatArray(env, jmean);
  std::vector<float> std = utils::GetVecFromJFloatArray(env, jstd);
  torch::Tensor mean_tensor = torch::from_blob(mean.data(), {3, 1, 1});
  torch::Tensor std_tensor = torch::from_blob(std.data(), {3, 1, 1});
  if (tensor_ptr->dim() == 4) {
    mean_tensor = mean_tensor.reshape({1, 3, 1, 1});
    std_tensor = std_tensor.reshape({1, 3, 1, 1});
  }
  const auto* result_ptr = new torch::Tensor(tensor_ptr->sub(mean_tensor).div(std_tensor));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
}