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
#ifndef DJL_TORCH_DJL_PYTORCH_JNI_UTILS_H
#define DJL_TORCH_DJL_PYTORCH_JNI_UTILS_H

#include <c10/util/typeid.h>
#include <djl/utils.h>
#include <jni.h>
#include <torch/csrc/api/include/torch/enum.h>
#include <torch/script.h>

#include <iostream>
#include <limits>
#ifdef V1_13_X
#include <c10/util/variant.h>
#else
#include <variant>
#endif

#include "djl_pytorch_jni_log.h"

// The file is utilities that are used for JNI

namespace utils {

#if !defined(__ANDROID__)
// for image interpolation
#ifdef V1_13_X
typedef torch::variant<torch::enumtype::kNearest, torch::enumtype::kLinear, torch::enumtype::kBilinear,
    torch::enumtype::kBicubic, torch::enumtype::kTrilinear, torch::enumtype::kArea, torch::enumtype::kNearestExact>
    mode_t;
#else
typedef std::variant<torch::enumtype::kNearest, torch::enumtype::kLinear, torch::enumtype::kBilinear,
    torch::enumtype::kBicubic, torch::enumtype::kTrilinear, torch::enumtype::kArea, torch::enumtype::kNearestExact>
    mode_t;
#endif
#endif

inline jint GetDTypeFromScalarType(const torch::ScalarType& type) {
  if (torch::kFloat32 == type) {
    return 0;
  } else if (torch::kFloat64 == type) {
    return 1;
  } else if (torch::kFloat16 == type) {
    return 2;
  } else if (torch::kUInt8 == type) {
    return 3;
  } else if (torch::kInt32 == type) {
    return 4;
  } else if (torch::kInt8 == type) {
    return 5;
  } else if (torch::kInt64 == type) {
    return 6;
  } else if (torch::kBool == type) {
    return 7;
  } else if (torch::kComplexFloat == type) {
    return 8;
  } else {
    return 9;
  }
}

inline torch::ScalarType GetScalarTypeFromDType(jint dtype) {
  switch (dtype) {
    case 0:
      return torch::kFloat32;
    case 1:
      return torch::kFloat64;
    case 2:
      return torch::kFloat16;
    case 3:
      return torch::kUInt8;
    case 4:
      return torch::kInt32;
    case 5:
      return torch::kInt8;
    case 6:
      return torch::kInt64;
    case 7:
      return torch::kBool;
    case 8:
      return torch::kComplexFloat;
    default:
      // TODO improve the error handling
      throw;
  }
}

inline torch::Device GetDeviceFromJDevice(JNIEnv* env, jintArray jdevice) {
  jint* device = env->GetIntArrayElements(jdevice, JNI_FALSE);
  auto device_type = static_cast<torch::DeviceType>(*device);
  int device_idx = *(device + 1);
  if (device_type == torch::DeviceType::CPU) {
    device_idx = -1;
  }
  torch::Device torch_device(device_type, device_idx);
  env->ReleaseIntArrayElements(jdevice, device, djl::utils::jni::RELEASE_MODE);
  return torch_device;
}

#if !defined(__ANDROID__)
inline mode_t GetInterpolationMode(jint jmode) {
  switch (jmode) {
    case 0:
      return torch::kNearest;
    case 1:
      return torch::kLinear;
    case 2:
      return torch::kBilinear;
    case 3:
      return torch::kBicubic;
    case 4:
      return torch::kTrilinear;
    case 5:
      return torch::kArea;
    default:
      throw;
  }
}
#endif

inline std::vector<torch::indexing::TensorIndex> CreateTensorIndex(
    JNIEnv* env, jlongArray jmin_indices, jlongArray jmax_indices, jlongArray jstep_indices) {
  const auto min_indices = djl::utils::jni::GetVecFromJLongArray(env, jmin_indices);
  const auto max_indices = djl::utils::jni::GetVecFromJLongArray(env, jmax_indices);
  const auto step_indices = djl::utils::jni::GetVecFromJLongArray(env, jstep_indices);
  std::vector<torch::indexing::TensorIndex> indices;
  indices.reserve(min_indices.size());
  for (size_t i = 0; i < min_indices.size(); ++i) {
    indices.emplace_back(
        torch::indexing::TensorIndex(torch::indexing::Slice(min_indices[i], max_indices[i], step_indices[i])));
  }
  return indices;
}

inline torch::TensorOptions CreateTensorOptions(
    JNIEnv* env, jint jdtype, jint jlayout, jintArray jdevice, jboolean jrequired_grad) {
  // it gets the device and collect jdevice memory
  const auto device = utils::GetDeviceFromJDevice(env, jdevice);
  auto options = torch::TensorOptions()
                     // for tensor creation API, MKLDNN layout is not supported
                     // the workaround is to create with Strided then call to_mkldnn()
                     .layout((jlayout != 1) ? torch::kStrided : torch::kSparse)
                     .memory_format(torch::MemoryFormat::Contiguous)
                     .device(device)
                     .requires_grad(JNI_TRUE == jrequired_grad);
  // DJL's UNKNOWN type
  if (jdtype != 8) {
    options = options.dtype(GetScalarTypeFromDType(jdtype));
  }
  return options;
}

inline int64_t GetTensorByteSize(torch::IntArrayRef shape, int64_t element_size) {
  TORCH_CHECK(element_size > 0, "Invalid element size: ", element_size);
  constexpr int64_t limit = std::numeric_limits<int64_t>::max();
  int64_t numel = 1;
  for (const int64_t dim : shape) {
    TORCH_CHECK(dim >= 0, "Invalid shape dimension: ", dim);
    TORCH_CHECK(dim == 0 || numel <= limit / dim, "Tensor element count overflows int64");
    numel *= dim;
  }
  TORCH_CHECK(numel <= limit / element_size, "Tensor size in bytes overflows int64");
  return numel * element_size;
}

inline void* GetValidatedDirectBufferAddress(
    JNIEnv* env, jobject jbuffer, torch::IntArrayRef shape, int64_t element_size) {
  void* address = env->GetDirectBufferAddress(jbuffer);
  const int64_t capacity = env->GetDirectBufferCapacity(jbuffer);
  TORCH_CHECK(address != nullptr && capacity >= 0, "Expected a direct ByteBuffer");
  const int64_t required = GetTensorByteSize(shape, element_size);
  TORCH_CHECK(capacity >= required, "Buffer is too small for the requested shape: ", capacity,
      " bytes available, ", required, " bytes required");
  return address;
}

}  // namespace utils

#endif  // DJL_TORCH_DJL_PYTORCH_JNI_UTILS_H
