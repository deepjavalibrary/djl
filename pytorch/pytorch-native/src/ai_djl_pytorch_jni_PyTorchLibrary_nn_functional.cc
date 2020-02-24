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

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_resize
  (JNIEnv* env, jobject jthis, jobject jhandle, jlongArray jsize, jboolean jalign_corners) {
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto size_vec = utils::GetVecFromJLongArray(env, jsize);
  size_t dim = tensor_ptr->dim();
  torch::Tensor temp_tensor = *tensor_ptr;
  if (dim == 3) {
    temp_tensor = temp_tensor.unsqueeze(0);
  }
  temp_tensor = temp_tensor.permute({0, 3, 1, 2});
  torch::Tensor result = torch::upsample_bilinear2d(temp_tensor, size_vec, jalign_corners == JNI_TRUE).permute(
    {0, 2, 3, 1});
  if (dim == 3) {
    result = result.squeeze(0);
  }
  const auto* result_ptr = new torch::Tensor(result);
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_normalize
  (JNIEnv* env, jobject jthis, jobject jhandle, jobject jmean, jobject jstd) {
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* mean_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jmean);
  const auto* std_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jstd);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->sub(*mean_ptr).div(*std_ptr));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
}
