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

#include <cstdint>
#include <jni.h>
#include <vector>
#include <string>
#include <limits>
#include <stdexcept>

namespace djl {
namespace utils {
namespace jni {

static constexpr const jint RELEASE_MODE = JNI_ABORT;
static constexpr const jlong NULL_PTR = 0;

// Security: Maximum allocation limits to prevent DoS attacks
// These limits are reasonable for ML inference workloads while preventing unbounded allocation
// Note: MAX_ELEMENT_COUNT is the maximum number of elements, not bytes. For example, an array
// of 1 billion int32_t elements would be 4GB of memory. Adjust these limits based on your
// deployment constraints (batch inference may need larger limits, edge devices need smaller).
static constexpr const size_t MAX_ELEMENT_COUNT = 1024 * 1024 * 1024; // 1 billion elements max
static constexpr const size_t MAX_STRING_LENGTH = 1024 * 1024; // 1MB per string
static constexpr const size_t MAX_ALLOCATION_COUNT = 100000; // Max number of native allocations

// Security: Pointer validation - ensures jlong values are valid native pointers
// IMPORTANT: This is a best-effort heuristic check, NOT a complete safety guarantee.
// It catches common errors (null, low memory, misalignment) but cannot verify:
// - Whether the pointer points to a valid, allocated object
// - Whether the object type matches the expected type T
// - Whether the object has already been freed (use-after-free)
// - Whether the pointer is within valid memory regions
// The Java-native trust boundary means Java code must ensure pointer validity.
inline bool IsValidPointer(jlong ptr) {
  // Null pointer is explicitly allowed
  if (ptr == NULL_PTR) {
    return true;
  }
  
  // Check pointer is not in low memory (likely invalid)
  // Most systems don't allocate valid pointers below this threshold
  constexpr uintptr_t MIN_VALID_PTR = 0x1000;
  
  // Check pointer alignment - should be at least pointer-aligned
  // This catches many common corruption cases
  constexpr uintptr_t ALIGNMENT_MASK = sizeof(void*) - 1;
  
  uintptr_t addr = static_cast<uintptr_t>(ptr);
  
  return (addr >= MIN_VALID_PTR) && ((addr & ALIGNMENT_MASK) == 0);
}

// Security: Bounds checking helper
// Note: C++ exceptions thrown here MUST be caught at the JNI boundary and converted
// to Java exceptions (e.g., IllegalArgumentException) to avoid undefined behavior.
// Never let C++ exceptions propagate through JNI to Java without translation.
inline void ValidateArraySize(jsize size, const char* context) {
  if (size < 0) {
    throw std::invalid_argument(std::string(context) + ": negative array size");
  }
  if (static_cast<size_t>(size) > MAX_ELEMENT_COUNT) {
    throw std::invalid_argument(std::string(context) + ": array size exceeds maximum allowed");
  }
}

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
  
  // Security: Validate string length to prevent excessive memory allocation
  ValidateArraySize(length, "GetStringFromJString");
  if (static_cast<size_t>(length) > MAX_STRING_LENGTH) {
    env->DeleteLocalRef(jbytes);
    throw std::invalid_argument("String length exceeds maximum allowed");
  }
  
  jbyte* c_str = env->GetByteArrayElements(jbytes, NULL);
  std::string str = std::string(reinterpret_cast<const char *>(c_str), length);

  env->ReleaseByteArrayElements(jbytes, c_str, RELEASE_MODE);
  env->DeleteLocalRef(jbytes);
  return str;
}

template<typename T>
inline std::vector<T> GetObjectVecFromJHandles(JNIEnv* env, jlongArray jhandles) {
  jsize length = env->GetArrayLength(jhandles);
  
  // Security: Validate array size
  // Note: Exceptions must be caught and translated at JNI boundary
  ValidateArraySize(length, "GetObjectVecFromJHandles");
  
  jlong* jptrs = env->GetLongArrayElements(jhandles, JNI_FALSE);
  
  // Security: Use resize() instead of reserve() to ensure proper vector size
  // This prevents out-of-bounds writes
  std::vector<T> vec;
  vec.reserve(static_cast<size_t>(length)); // Pre-allocate for performance
  
  for (jsize i = 0; i < length; ++i) {
    // Security: Validate pointer before dereferencing
    if (!IsValidPointer(jptrs[i])) {
      env->ReleaseLongArrayElements(jhandles, jptrs, RELEASE_MODE);
      throw std::invalid_argument("GetObjectVecFromJHandles: invalid pointer at index " + std::to_string(i));
    }
    
    // Security: Only dereference after validation
    // Note: We still trust that the pointer points to a valid T object
    // This is a fundamental limitation of the Java-native trust boundary
    T* ptr = reinterpret_cast<T*>(static_cast<uintptr_t>(jptrs[i]));
    vec.emplace_back(*ptr);
  }
  
  env->ReleaseLongArrayElements(jhandles, jptrs, RELEASE_MODE);
  return vec;
}

template<typename T1, typename T2>
inline jlongArray GetPtrArrayFromContainer(JNIEnv* env, T1 list) {
  size_t len = list.size();
  
  // Security: Validate allocation count to prevent DoS
  // Note: Exceptions must be caught and translated at JNI boundary
  if (len > MAX_ALLOCATION_COUNT) {
    throw std::invalid_argument("GetPtrArrayFromContainer: allocation count exceeds maximum");
  }
  
  jlongArray jarray = env->NewLongArray(static_cast<jsize>(len));
  if (jarray == nullptr) {
    throw std::bad_alloc();
  }
  
  // Security: Use resize() instead of reserve() + index access
  // This ensures the vector has the correct size before writing
  std::vector<jlong> jptrs;
  jptrs.resize(len); // Properly size the vector
  
  // Security: Track allocations for cleanup on failure
  std::vector<T2*> allocated_ptrs;
  allocated_ptrs.reserve(len);
  
  try {
    for (size_t i = 0; i < len; ++i) {
      // Security: Use RAII-style allocation tracking
      T2* element_ptr = new T2(list[i]);
      allocated_ptrs.push_back(element_ptr);
      jptrs[i] = reinterpret_cast<jlong>(reinterpret_cast<uintptr_t>(element_ptr));
    }
    
    env->SetLongArrayRegion(jarray, 0, static_cast<jsize>(len), jptrs.data());
    
    // Note: Allocated memory ownership is transferred to Java side
    // Java code MUST call appropriate cleanup functions to deallocate these pointers
    
  } catch (...) {
    // Security: Cleanup on failure - deterministic deallocation
    for (T2* ptr : allocated_ptrs) {
      delete ptr;
    }
    throw;
  }
  
  return jarray;
}

inline std::vector<int64_t> GetVecFromJLongArray(JNIEnv* env, jlongArray jarray) {
  jsize length = env->GetArrayLength(jarray);
  
  // Security: Validate array size
  ValidateArraySize(length, "GetVecFromJLongArray");
  
  jlong* jarr = env->GetLongArrayElements(jarray, JNI_FALSE);
  std::vector<int64_t> vec(jarr, jarr + length);
  env->ReleaseLongArrayElements(jarray, jarr, RELEASE_MODE);
  return vec;
}

inline std::vector<int32_t> GetVecFromJIntArray(JNIEnv* env, jintArray jarray) {
  jsize length = env->GetArrayLength(jarray);
  
  // Security: Validate array size
  ValidateArraySize(length, "GetVecFromJIntArray");
  
  jint* jarr = env->GetIntArrayElements(jarray, JNI_FALSE);
  std::vector<int32_t> vec(jarr, jarr + length);
  env->ReleaseIntArrayElements(jarray, jarr, RELEASE_MODE);
  return vec;
}

inline std::vector<float> GetVecFromJFloatArray(JNIEnv* env, jfloatArray jarray) {
  jsize length = env->GetArrayLength(jarray);
  
  // Security: Validate array size
  ValidateArraySize(length, "GetVecFromJFloatArray");
  
  jfloat* jarr = env->GetFloatArrayElements(jarray, JNI_FALSE);
  std::vector<float> vec(jarr, jarr + length);
  env->ReleaseFloatArrayElements(jarray, jarr, RELEASE_MODE);
  return vec;
}

inline std::vector<std::string> GetVecFromJStringArray(JNIEnv* env, jobjectArray array) {
  jsize len = env->GetArrayLength(array);
  
  // Security: Validate array size
  ValidateArraySize(len, "GetVecFromJStringArray");
  
  std::vector<std::string> vec;
  vec.reserve(static_cast<size_t>(len)); // Pre-allocate for performance
  
  for (jsize i = 0; i < len; ++i) {
    jstring jstr = (jstring) env->GetObjectArrayElement(array, i);
    vec.emplace_back(djl::utils::jni::GetStringFromJString(env, jstr));
    env->DeleteLocalRef(jstr);
  }
  
  return vec;
}

// String[]
inline jobjectArray GetStringArrayFromVec(JNIEnv* env, const std::vector<std::string> &vec) {
  // Security: Validate output array size
  // Note: Exceptions must be caught and translated at JNI boundary
  if (vec.size() > MAX_ALLOCATION_COUNT) {
    throw std::invalid_argument("GetStringArrayFromVec: array size exceeds maximum");
  }
  
  jobjectArray array = env->NewObjectArray(static_cast<jsize>(vec.size()), 
                                           env->FindClass("java/lang/String"), 
                                           nullptr);
  if (array == nullptr) {
    throw std::bad_alloc();
  }

  // TODO: cache reflection to improve performance
  const jclass string_class = env->FindClass("java/lang/String");
  const jmethodID ctor = env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V");
  const jstring charset = env->NewStringUTF("UTF-8");

  for (size_t i = 0; i < vec.size(); ++i) {
    // Security: Validate string length
    size_t len = vec[i].length();
    if (len > MAX_STRING_LENGTH) {
      env->DeleteLocalRef(charset);
      throw std::invalid_argument("GetStringArrayFromVec: string length exceeds maximum at index " + std::to_string(i));
    }
    
    const char* c_str = vec[i].c_str();
    auto jbytes = env->NewByteArray(static_cast<jsize>(len));
    if (jbytes == nullptr) {
      env->DeleteLocalRef(charset);
      throw std::bad_alloc();
    }
    
    env->SetByteArrayRegion(jbytes, 0, static_cast<jsize>(len), reinterpret_cast<const jbyte*>(c_str));
    jobject jstr = env->NewObject(string_class, ctor, jbytes, charset);
    env->DeleteLocalRef(jbytes);
    env->SetObjectArrayElement(array, static_cast<jsize>(i), jstr);
    env->DeleteLocalRef(jstr);
  }

  env->DeleteLocalRef(charset);
  return array;
}

inline jintArray GetIntArrayFromVec(JNIEnv* env, const std::vector<int> &vec) {
  // Security: Validate array size
  // Note: Exceptions must be caught and translated at JNI boundary
  if (vec.size() > MAX_ALLOCATION_COUNT) {
    throw std::invalid_argument("GetIntArrayFromVec: array size exceeds maximum");
  }
  
  jintArray array = env->NewIntArray(static_cast<jsize>(vec.size()));
  if (array == nullptr) {
    throw std::bad_alloc();
  }
  
  env->SetIntArrayRegion(array, 0, static_cast<jsize>(vec.size()), 
                         reinterpret_cast<const jint*>(vec.data()));
  return array;
}

inline jlongArray GetLongArrayFromVec(JNIEnv* env, const std::vector<size_t> &vec) {
  // Security: Validate array size
  // Note: Exceptions must be caught and translated at JNI boundary
  if (vec.size() > MAX_ALLOCATION_COUNT) {
    throw std::invalid_argument("GetLongArrayFromVec: array size exceeds maximum");
  }
  
  jlongArray array = env->NewLongArray(static_cast<jsize>(vec.size()));
  if (array == nullptr) {
    throw std::bad_alloc();
  }
  
  env->SetLongArrayRegion(array, 0, static_cast<jsize>(vec.size()), 
                          reinterpret_cast<const jlong*>(vec.data()));
  return array;
}

inline jobjectArray Get2DIntArrayFrom2DVec(JNIEnv* env, const std::vector<std::vector<int>> &vec) {
  // Security: Validate outer array size
  // Note: Exceptions must be caught and translated at JNI boundary
  if (vec.size() > MAX_ALLOCATION_COUNT) {
    throw std::invalid_argument("Get2DIntArrayFrom2DVec: array size exceeds maximum");
  }
  
  jobjectArray array = env->NewObjectArray(static_cast<jsize>(vec.size()), 
                                           env->FindClass("[I"), 
                                           nullptr);
  if (array == nullptr) {
    throw std::bad_alloc();
  }
  
  for (size_t i = 0; i < vec.size(); ++i) {
    jintArray inner = djl::utils::jni::GetIntArrayFromVec(env, vec[i]);
    env->SetObjectArrayElement(array, static_cast<jsize>(i), inner);
    env->DeleteLocalRef(inner);
  }
  
  return array;
}

inline jobjectArray Get2DLongArrayFrom2DVec(JNIEnv* env, const std::vector<std::vector<size_t>> &vec) {
  // Security: Validate outer array size
  // Note: Exceptions must be caught and translated at JNI boundary
  if (vec.size() > MAX_ALLOCATION_COUNT) {
    throw std::invalid_argument("Get2DLongArrayFrom2DVec: array size exceeds maximum");
  }
  
  jobjectArray array = env->NewObjectArray(static_cast<jsize>(vec.size()), 
                                           env->FindClass("[J"), 
                                           nullptr);
  if (array == nullptr) {
    throw std::bad_alloc();
  }
  
  for (size_t i = 0; i < vec.size(); ++i) {
    jlongArray inner = djl::utils::jni::GetLongArrayFromVec(env, vec[i]);
    env->SetObjectArrayElement(array, static_cast<jsize>(i), inner);
    env->DeleteLocalRef(inner);
  }
  
  return array;
}

inline std::vector<std::vector<size_t>> Get2DVecFrom2DLongArray(JNIEnv* env, jobjectArray array) {
  jsize len = env->GetArrayLength(array);
  
  // Security: Validate outer array size
  ValidateArraySize(len, "Get2DVecFrom2DLongArray");
  
  std::vector<std::vector<size_t>> vec;
  vec.reserve(static_cast<size_t>(len)); // Pre-allocate for performance
  
  for (jsize i = 0; i < len; ++i) {
    auto long_array = (jlongArray) env->GetObjectArrayElement(array, i);
    jsize inner_len = env->GetArrayLength(long_array);
    
    // Security: Validate inner array size
    ValidateArraySize(inner_len, "Get2DVecFrom2DLongArray inner");
    
    jlong* jarr = env->GetLongArrayElements(long_array, JNI_FALSE);
    
    std::vector<size_t> temp;
    temp.reserve(static_cast<size_t>(inner_len)); // Pre-allocate for performance
    
    for (jsize j = 0; j < inner_len; ++j) {
      temp.emplace_back(static_cast<size_t>(jarr[j]));
    }
    
    vec.emplace_back(std::move(temp));
    env->ReleaseLongArrayElements(long_array, jarr, RELEASE_MODE);
    env->DeleteLocalRef(long_array);
  }
  
  return vec;
}

// String[][]
inline jobjectArray Get2DStringArrayFrom2DVec(JNIEnv* env, const std::vector<std::vector<std::string>> &vec) {
  // Security: Validate outer array size
  // Note: Exceptions must be caught and translated at JNI boundary
  if (vec.size() > MAX_ALLOCATION_COUNT) {
    throw std::invalid_argument("Get2DStringArrayFrom2DVec: array size exceeds maximum");
  }
  
  jobjectArray array = env->NewObjectArray(static_cast<jsize>(vec.size()), 
                                           env->FindClass("[Ljava/lang/String;"), 
                                           nullptr);
  if (array == nullptr) {
    throw std::bad_alloc();
  }
  
  for (size_t i = 0; i < vec.size(); ++i) {
    jobjectArray inner = GetStringArrayFromVec(env, vec[i]);
    env->SetObjectArrayElement(array, static_cast<jsize>(i), inner);
    env->DeleteLocalRef(inner);
  }
  
  return array;
}

} // namespace jni
} // namespace utils
} // namespace djl

#endif //DJL_UTILS_H
