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

#include <jni.h>
#include <iostream>
#include <c10/util/typeid.h>
#include <torch/script.h>

// The file is utilities that ared used for JNI

namespace utils {

static constexpr const char* const POINTER_CLASS = "ai/djl/pytorch/jni/Pointer";

inline jint GetDTypeFromScalarType(const c10::ScalarType& type) {
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
  } else {
    return 8;
  }
}

inline c10::ScalarType GetScalarTypeFromDType(jint dtype) {
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
    default:
      // TODO improve the error handling
      throw;
  }
}

template<typename T>
inline T* GetPointerFromJHandle(JNIEnv* env, jobject jhandle) {
  jclass cls = env->FindClass(POINTER_CLASS);
  jmethodID get_value = env->GetMethodID(cls, "getValue", "()J");
  if (nullptr == get_value) {
    std::cout << "getValue method not found!" << std::endl;
  }
  jlong ptr = env->CallLongMethod(jhandle, get_value);
  return reinterpret_cast<T*>(ptr);
}

template<typename T>
inline std::vector<T> GetObjectVecFromJHandles(JNIEnv* env, jobjectArray jhandles) {
  jclass cls = env->FindClass(POINTER_CLASS);
  jmethodID get_value = env->GetMethodID(cls, "getValue", "()J");
  jsize length = env->GetArrayLength(jhandles);
  std::vector<T> vec;
  vec.reserve(length);
  for (auto i = 0; i < length; ++i) {
    jobject jhandle = env->GetObjectArrayElement(jhandles, i);
    if (nullptr == jhandle) {
      std::cout << "Pointer not found!" << std::endl;
    }
    jlong ptr = env->CallLongMethod(jhandle, get_value);
    vec.emplace_back(*(reinterpret_cast<T*>(ptr)));
  }
  return std::move(vec);
}

template<typename T>
inline jobject CreatePointer(JNIEnv* env, const T* ptr) {
  jclass cls = env->FindClass(POINTER_CLASS);
  if (nullptr == cls) {
    std::cout << "Pointer class not found!" << std::endl;
    return nullptr;
  }
  jmethodID init = env->GetMethodID(cls, "<init>", "(J)V");
  jobject new_obj = env->NewObject(cls, init, ptr);
  if (nullptr == new_obj) {
    std::cout << "object created failed" << std::endl;
    return nullptr;
  }
  return new_obj;
}

inline std::vector<int64_t> GetVecFromJLongArray(JNIEnv* env, jlongArray jarray) {
  jlong* jarr = env->GetLongArrayElements(jarray, JNI_FALSE);
  jsize length = env->GetArrayLength(jarray);
  const std::vector<int64_t> vec(jarr, jarr + length);
  return vec;
}

inline std::vector<int32_t> GetVecFromJIntArray(JNIEnv* env, jintArray jarray) {
  jint* jarr = env->GetIntArrayElements(jarray, JNI_FALSE);
  jsize length = env->GetArrayLength(jarray);
  const std::vector<int32_t> vec(jarr, jarr + length);
  return vec;
}

inline c10::Device GetDeviceFromJDevice(JNIEnv* env, jintArray jdevice) {
  jint* device = env->GetIntArrayElements(jdevice, JNI_FALSE);
  c10::DeviceType device_type = static_cast<c10::DeviceType>(*device);
  int device_idx = *(device + 1);
  if (device_type == c10::DeviceType::CPU) {
    device_idx = -1;
  }
  c10::Device c10_device(device_type, device_idx);
  return c10_device;
}

inline torch::TensorOptions CreateTensorOptions(JNIEnv* env,
                                                jint jdtype,
                                                jint jlayout,
                                                jintArray jdevice,
                                                jboolean jrequired_grad) {
  const auto device = utils::GetDeviceFromJDevice(env, jdevice);
  auto options = torch::TensorOptions()
    .dtype(GetScalarTypeFromDType(jdtype))
    .layout((jlayout == 0) ? torch::kStrided : torch::kSparse)
    .device(device)
    .requires_grad(JNI_TRUE == jrequired_grad);
  return options;
}

} // namespace utils

#endif //DJL_TORCH_DJL_PYTORCH_JNI_UTILS_H
