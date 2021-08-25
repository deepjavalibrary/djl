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

#ifndef DJL_PADDLE_DJL_PADDLE_JNI_UTILS_H
#define DJL_PADDLE_DJL_PADDLE_JNI_UTILS_H

#include <jni.h>
#include <paddle_api.h>

#include <iostream>
#include <numeric>
#include <vector>

namespace utils {

inline void GetZTensorFromTensor(paddle::ZeroCopyTensor *z_tensor, paddle::PaddleTensor *tensor) {
  z_tensor->Reshape(tensor->shape);
  z_tensor->SetLoD(tensor->lod);
  switch (tensor->dtype) {
    case paddle::PaddleDType::FLOAT32:
      z_tensor->copy_from_cpu(static_cast<float *>(tensor->data.data()));
      break;
    case paddle::PaddleDType::INT32:
      z_tensor->copy_from_cpu(static_cast<int32_t *>(tensor->data.data()));
      break;
    case paddle::PaddleDType::INT64:
      z_tensor->copy_from_cpu(static_cast<int64_t *>(tensor->data.data()));
      break;
    case paddle::PaddleDType::UINT8:
      z_tensor->copy_from_cpu(static_cast<uint8_t *>(tensor->data.data()));
      break;
    default:
      // TODO improve the error handling
      throw;
  }
}

inline void GetTensorFromZTensor(paddle::ZeroCopyTensor *z_tensor, paddle::PaddleTensor *tensor) {
  tensor->name = z_tensor->name();
  tensor->dtype = z_tensor->type();
  tensor->shape = z_tensor->shape();
  tensor->lod = z_tensor->lod();
  std::vector<int> output_shape = z_tensor->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  auto dtype = z_tensor->type();
  if (dtype == paddle::PaddleDType::FLOAT32) {
    int size = sizeof(float);
    tensor->data.Resize(out_num * size);
    z_tensor->copy_to_cpu(static_cast<float *>(tensor->data.data()));
  } else if (dtype == paddle::PaddleDType::INT32) {
    int size = sizeof(int32_t);
    tensor->data.Resize(out_num * size);
    z_tensor->copy_to_cpu(static_cast<int32_t *>(tensor->data.data()));
  } else if (dtype == paddle::PaddleDType::INT64) {
    int size = sizeof(int64_t);
    tensor->data.Resize(out_num * size);
    z_tensor->copy_to_cpu(static_cast<int64_t *>(tensor->data.data()));
  } else if (dtype == paddle::PaddleDType::UINT8) {
    int size = sizeof(uint8_t);
    tensor->data.Resize(out_num * size);
    z_tensor->copy_to_cpu(static_cast<uint8_t *>(tensor->data.data()));
  } else {
    // throw error
    throw;
  }
}

}  // namespace utils

#endif  // DJL_PADDLE_DJL_PADDLE_JNI_UTILS_H
