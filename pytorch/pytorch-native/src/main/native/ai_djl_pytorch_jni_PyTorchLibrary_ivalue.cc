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

#include <djl/utils.h>
#include <torch/script.h>

#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_exception.h"
#include "djl_pytorch_utils.h"

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueFromTensor(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* ivalue_ptr = new torch::IValue(*reinterpret_cast<torch::Tensor*>(jhandle));
  return reinterpret_cast<uintptr_t>(ivalue_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueFromList(
    JNIEnv* env, jobject jthis, jlongArray jtensor_ptrs) {
  jsize len = env->GetArrayLength(jtensor_ptrs);
  jlong* jptrs = env->GetLongArrayElements(jtensor_ptrs, JNI_FALSE);
  torch::List<torch::Tensor> list;
  list.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    list.emplace_back(*reinterpret_cast<torch::Tensor*>(jptrs[i]));
  }
  env->ReleaseLongArrayElements(jtensor_ptrs, jptrs, JNI_ABORT);
  auto* result_ptr = new torch::IValue(list);
  return reinterpret_cast<uintptr_t>(result_ptr);
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueFromDict(
    JNIEnv* env, jobject jthis, jlongArray jtensor_ptrs, jobjectArray jnames) {
  auto len = static_cast<size_t>(env->GetArrayLength(jtensor_ptrs));
  jlong* jptrs = env->GetLongArrayElements(jtensor_ptrs, JNI_FALSE);
  torch::Dict<std::string, torch::Tensor> dict;
  dict.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    auto jname = (jstring) env->GetObjectArrayElement(jnames, i);
    std::string name = djl::utils::jni::GetStringFromJString(env, jname);
    dict.insert(name, *reinterpret_cast<torch::Tensor*>(jptrs[i]));
  }
  env->ReleaseLongArrayElements(jtensor_ptrs, jptrs, JNI_ABORT);
  env->DeleteLocalRef(jnames);
  auto* result_ptr = new torch::IValue(dict);
  return reinterpret_cast<uintptr_t>(result_ptr);
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToTensor(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* tensor_ptr = new torch::Tensor(reinterpret_cast<torch::IValue*>(jhandle)->toTensor());
  return reinterpret_cast<uintptr_t>(tensor_ptr);
  API_END_RETURN()
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToListFromTuple(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = reinterpret_cast<torch::IValue*>(jhandle);
  std::vector<torch::IValue> ivalue_vec = ivalue_ptr->toTuple()->elements();
  return djl::utils::jni::GetPtrArrayFromContainer<std::vector<torch::IValue>, torch::IValue>(env, ivalue_vec);
  API_END_RETURN()
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToTensorList(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = reinterpret_cast<torch::IValue*>(jhandle);
  torch::List<torch::Tensor> tensor_list = ivalue_ptr->toTensorList();
  return djl::utils::jni::GetPtrArrayFromContainer<torch::List<torch::Tensor>, torch::Tensor>(env, tensor_list);
  API_END_RETURN()
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToList(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = reinterpret_cast<torch::IValue*>(jhandle);
  torch::List<torch::IValue> ivalue_list = ivalue_ptr->toList();
  return djl::utils::jni::GetPtrArrayFromContainer<torch::List<torch::IValue>, torch::IValue>(env, ivalue_list);
  API_END_RETURN()
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToMap(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = reinterpret_cast<torch::IValue*>(jhandle);
  torch::Dict<torch::IValue, torch::IValue> dict = ivalue_ptr->toGenericDict();
  size_t len = dict.size() * 2;
  jlongArray jarray = env->NewLongArray(len);
  std::vector<jlong> jptrs;
  jptrs.reserve(len);
  size_t array_iter = 0;
  for (auto it = dict.begin(); it != dict.end(); ++it) {
    const auto* key_ptr = new torch::IValue(it->key());
    jptrs[array_iter++] = reinterpret_cast<uintptr_t>(key_ptr);
    const auto* value_ptr = new torch::IValue(it->value());
    jptrs[array_iter++] = reinterpret_cast<uintptr_t>(value_ptr);
  }
  env->SetLongArrayRegion(jarray, 0, len, jptrs.data());
  return jarray;
  API_END_RETURN()
}

JNIEXPORT jstring JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToString(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = reinterpret_cast<torch::IValue*>(jhandle);
  return env->NewStringUTF(ivalue_ptr->toString()->string().c_str());
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsString(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  return reinterpret_cast<torch::IValue*>(jhandle)->isString();
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsTensor(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  return reinterpret_cast<torch::IValue*>(jhandle)->isTensor();
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsTensorList(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  return reinterpret_cast<torch::IValue*>(jhandle)->isTensorList();
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsList(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  return reinterpret_cast<torch::IValue*>(jhandle)->isList();
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsMap(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  return reinterpret_cast<torch::IValue*>(jhandle)->isGenericDict();
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsTuple(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  return reinterpret_cast<torch::IValue*>(jhandle)->isTuple();
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteIValue(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = reinterpret_cast<torch::IValue*>(jhandle);
  delete ivalue_ptr;
  API_END()
}
