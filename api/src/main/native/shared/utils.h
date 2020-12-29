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

#ifndef DJL_UTILS_H
#define DJL_UTILS_H

#include <jni.h>
#include <vector>
#include <string>

namespace utils {
namespace jni {

static constexpr const jint RELEASE_MODE = JNI_ABORT;
static constexpr const jlong NULL_PTR = 0;

template<typename T>
inline std::vector <T> GetObjectVecFromJHandles(JNIEnv *env, jlongArray jhandles) {
  jsize length = env->GetArrayLength(jhandles);
  jlong *jptrs = env->GetLongArrayElements(jhandles, JNI_FALSE);
  std::vector <T> vec;
  vec.reserve(length);
  for (size_t i = 0; i < length; ++i) {
    vec.emplace_back(*(reinterpret_cast<T *>(jptrs[i])));
  }
  env->ReleaseLongArrayElements(jhandles, jptrs, RELEASE_MODE);
  return std::move(vec);
}

template <typename T1, typename T2>
inline jlongArray GetPtrArrayFromContainer(JNIEnv* env, T1 list) {
  size_t len = list.size();
  jlongArray jarray = env->NewLongArray(len);
  std::vector<jlong> jptrs;
  jptrs.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    const auto* element_ptr = new T2(list[i]);
    jptrs[i] = reinterpret_cast<uintptr_t>(element_ptr);
  }
  env->SetLongArrayRegion(jarray, 0, len, jptrs.data());
  return jarray;
}

inline std::vector<int64_t> GetVecFromJLongArray(JNIEnv* env, jlongArray jarray) {
  jlong* jarr = env->GetLongArrayElements(jarray, JNI_FALSE);
  jsize length = env->GetArrayLength(jarray);
  std::vector<int64_t> vec(jarr, jarr + length);
  env->ReleaseLongArrayElements(jarray, jarr, RELEASE_MODE);
  return std::move(vec);
}

inline std::vector<int32_t> GetVecFromJIntArray(JNIEnv* env, jintArray jarray) {
  jint* jarr = env->GetIntArrayElements(jarray, JNI_FALSE);
  jsize length = env->GetArrayLength(jarray);
  std::vector<int32_t> vec(jarr, jarr + length);
  env->ReleaseIntArrayElements(jarray, jarr, RELEASE_MODE);
  return std::move(vec);
}

inline std::vector<float> GetVecFromJFloatArray(JNIEnv* env, jfloatArray jarray) {
  jfloat* jarr = env->GetFloatArrayElements(jarray, JNI_FALSE);
  jsize length = env->GetArrayLength(jarray);
  std::vector<float> vec(jarr, jarr + length);
  env->ReleaseFloatArrayElements(jarray, jarr, RELEASE_MODE);
  return std::move(vec);
}

inline std::string GetStringFromJString(JNIEnv* env, jstring jstr) {
  if (jstr == nullptr) {
    return std::string();
  }
  const char* c_str = env->GetStringUTFChars(jstr, JNI_FALSE);
  std::string str = std::string(c_str);
  env->ReleaseStringUTFChars(jstr, c_str);
  return str;
}

} // namespace jni
} // namespace utils

#endif //DJL_UTILS_H
