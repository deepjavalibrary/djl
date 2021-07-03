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

namespace djl {
namespace utils {
namespace jni {

static constexpr const jint RELEASE_MODE = JNI_ABORT;
static constexpr const jlong NULL_PTR = 0;

inline std::string GetStringFromJString(JNIEnv* env, jstring jstr) {
  if (jstr == nullptr) {
    return std::string();
  }

  // TODO: cache reflection to improve performance
  const jclass string_class = env->GetObjectClass(jstr);
  const jmethodID getbytes_method = env->GetMethodID(string_class, "getBytes", "(Ljava/lang/String;)[B");

  const jstring charset = env->NewStringUTF("UTF-8");
  const jbyteArray jbytes = (jbyteArray) env->CallObjectMethod(jstr, getbytes_method, charset);
  env->DeleteLocalRef(charset);

  const jsize length = env->GetArrayLength(jbytes);
  jbyte* c_str = env->GetByteArrayElements(jbytes, NULL);
  std::string str = std::string(reinterpret_cast<const char *>(c_str), length);

  env->ReleaseByteArrayElements(jbytes, c_str, RELEASE_MODE);
  env->DeleteLocalRef(jbytes);
  return str;
}

template<typename T>
inline std::vector<T> GetObjectVecFromJHandles(JNIEnv* env, jlongArray jhandles) {
  jsize length = env->GetArrayLength(jhandles);
  jlong* jptrs = env->GetLongArrayElements(jhandles, JNI_FALSE);
  std::vector<T> vec;
  vec.reserve(length);
  for (size_t i = 0; i < length; ++i) {
    vec.emplace_back(*(reinterpret_cast<T*>(jptrs[i])));
  }
  env->ReleaseLongArrayElements(jhandles, jptrs, RELEASE_MODE);
  return std::move(vec);
}

template<typename T1, typename T2>
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
  std::vector <int32_t> vec(jarr, jarr + length);
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

inline std::vector<std::string> GetVecFromJStringArray(JNIEnv* env, jobjectArray array) {
  std::vector <std::string> vec;
  jsize len = env->GetArrayLength(array);
  vec.reserve(len);
  for (int i = 0; i < len; ++i) {
    vec.emplace_back(djl::utils::jni::GetStringFromJString(
            env, (jstring) env->GetObjectArrayElement(array, i)));
  }
  return std::move(vec);
}

// String[]
inline jobjectArray GetStringArrayFromVec(JNIEnv* env, const std::vector <std::string> &vec) {
  jobjectArray array = env->NewObjectArray(vec.size(), env->FindClass("Ljava/lang/String;"), nullptr);

  // TODO: cache reflection to improve performance
  const jclass string_class = env->FindClass("java/lang/String");
  const jmethodID ctor = env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V");
  const jstring charset = env->NewStringUTF("UTF-8");

  for (int i = 0; i < vec.size(); ++i) {
    const char* c_str = vec[i].c_str();
    int len = vec[i].length();
    auto jbytes = env->NewByteArray(len);
    env->SetByteArrayRegion(jbytes, 0, len, reinterpret_cast<const jbyte*>(c_str));
    jobject jstr = env->NewObject(string_class, ctor, jbytes, charset);
    env->DeleteLocalRef(jbytes);
    env->SetObjectArrayElement(array, i, jstr);
  }

  env->DeleteLocalRef(charset);
  return array;
}

inline jintArray GetIntArrayFromVec(JNIEnv* env, const std::vector<int> &vec) {
  jintArray array = env->NewIntArray(vec.size());
  env->SetIntArrayRegion(array, 0, vec.size(), reinterpret_cast<const jint*>(vec.data()));
  return array;
}

inline jobjectArray Get2DIntArrayFrom2DVec(JNIEnv* env, const std::vector<std::vector<int>> &vec) {
  jobjectArray array = env->NewObjectArray(vec.size(), env->FindClass("[I"), nullptr);
  for (size_t i = 0; i < vec.size(); ++i) {
    env->SetObjectArrayElement(array, i, djl::utils::jni::GetIntArrayFromVec(env, vec[i]));
  }
  return array;
}

// String[][]
inline jobjectArray Get2DStringArrayFrom2DVec(JNIEnv* env, const std::vector<std::vector<std::string>> &vec) {
  jobjectArray array = env->NewObjectArray(vec.size(), env->FindClass("[Ljava/lang/String;"), nullptr);
  for (int i = 0; i < vec.size(); ++i) {
    env->SetObjectArrayElement(array, i, GetStringArrayFromVec(env, vec[i]));
  }
  return array;
}

} // namespace jni
} // namespace utils
} // namespace djl

#endif //DJL_UTILS_H
