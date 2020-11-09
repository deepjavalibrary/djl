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

#include <torch/script.h>

#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_exception.h"
#include "djl_pytorch_jni_utils.h"

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueFromTensor(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN()
  const auto* ivalue_ptr = new torch::IValue(*utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle));
  return utils::CreatePointer<torch::IValue>(env, ivalue_ptr);
  API_END_RETURN()
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueFromList(
    JNIEnv* env, jobject jthis, jobjectArray jtensor_ptrs) {
  size_t len = static_cast<size_t>(env->GetArrayLength(jtensor_ptrs));
  torch::List<torch::Tensor> list;
  list.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    auto* tensor_ptr =
        utils::GetPointerFromJHandle<const torch::Tensor>(env, env->GetObjectArrayElement(jtensor_ptrs, i));
    list.emplace_back(*tensor_ptr);
  }
  env->DeleteLocalRef(jtensor_ptrs);
  auto* result_ptr = new torch::IValue(list);
  return utils::CreatePointer<torch::IValue>(env, result_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueFromDict(
    JNIEnv* env, jobject jthis, jobjectArray jtensor_ptrs, jobjectArray jnames) {
  size_t len = static_cast<size_t>(env->GetArrayLength(jtensor_ptrs));
  torch::Dict<std::string, torch::Tensor> dict;
  dict.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    auto jname = (jstring)env->GetObjectArrayElement(jnames, i);
    std::string name = utils::GetStringFromJString(env, jname);
    auto* tensor_ptr =
        utils::GetPointerFromJHandle<const torch::Tensor>(env, env->GetObjectArrayElement(jtensor_ptrs, i));
    dict.insert(name, *tensor_ptr);
  }
  env->DeleteLocalRef(jtensor_ptrs);
  env->DeleteLocalRef(jnames);
  auto* result_ptr = new torch::IValue(dict);
  return utils::CreatePointer<torch::IValue>(env, result_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToTensor(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN()
  auto* tensor_ptr = new torch::Tensor(utils::GetPointerFromJHandle<torch::IValue>(env, jhandle)->toTensor());
  return utils::CreatePointer<torch::Tensor>(env, tensor_ptr);
  API_END_RETURN()
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToListFromTuple(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = utils::GetPointerFromJHandle<torch::IValue>(env, jhandle);
  auto ivalue_list = ivalue_ptr->toTuple()->elements();
  jobjectArray jarray = env->NewObjectArray(ivalue_list.size(), env->FindClass(utils::POINTER_CLASS), nullptr);
  for (size_t i = 0; i < ivalue_list.size(); ++i) {
    const auto* element_ptr = new torch::IValue(ivalue_list.at(i));
    auto ptr = utils::CreatePointer<torch::IValue>(env, element_ptr);
    env->SetObjectArrayElement(jarray, i, ptr);
  }
  return jarray;
  API_END_RETURN()
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToTensorList(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = utils::GetPointerFromJHandle<torch::IValue>(env, jhandle);
  auto ivalue_list = ivalue_ptr->toTensorList();
  jobjectArray jarray = env->NewObjectArray(ivalue_list.size(), env->FindClass(utils::POINTER_CLASS), nullptr);
  for (size_t i = 0; i < ivalue_list.size(); ++i) {
    const auto* element_ptr = new torch::Tensor(ivalue_list.get(i));
    auto ptr = utils::CreatePointer<torch::Tensor>(env, element_ptr);
    env->SetObjectArrayElement(jarray, i, ptr);
  }
  return jarray;
  API_END_RETURN()
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToList(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = utils::GetPointerFromJHandle<torch::IValue>(env, jhandle);
  auto ivalue_list = ivalue_ptr->toList();
  jobjectArray jarray = env->NewObjectArray(ivalue_list.size(), env->FindClass(utils::POINTER_CLASS), nullptr);
  for (size_t i = 0; i < ivalue_list.size(); ++i) {
    const auto* element_ptr = new torch::IValue(ivalue_list.get(i));
    auto ptr = utils::CreatePointer<torch::IValue>(env, element_ptr);
    env->SetObjectArrayElement(jarray, i, ptr);
  }
  return jarray;
  API_END_RETURN()
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToMap(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = utils::GetPointerFromJHandle<torch::IValue>(env, jhandle);
  auto dict = ivalue_ptr->toGenericDict();
  jobjectArray jarray = env->NewObjectArray(dict.size() * 2, env->FindClass(utils::POINTER_CLASS), nullptr);
  int array_iter = 0;
  for (auto it = dict.begin(); it != dict.end(); ++it) {
    const auto* key_ptr = new torch::IValue(it->key());
    auto ptr = utils::CreatePointer<torch::IValue>(env, key_ptr);
    env->SetObjectArrayElement(jarray, array_iter++, ptr);
    const auto* value_ptr = new torch::IValue(it->value());
    ptr = utils::CreatePointer<torch::IValue>(env, value_ptr);
    env->SetObjectArrayElement(jarray, array_iter++, ptr);
  }
  return jarray;
  API_END_RETURN()
}

JNIEXPORT jstring JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToString(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = utils::GetPointerFromJHandle<torch::IValue>(env, jhandle);
  return env->NewStringUTF(ivalue_ptr->toString()->string().c_str());
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsString(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN()
  return utils::GetPointerFromJHandle<torch::IValue>(env, jhandle)->isString();
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsTensor(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN()
  return utils::GetPointerFromJHandle<torch::IValue>(env, jhandle)->isTensor();
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsTensorList(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN()
  return utils::GetPointerFromJHandle<torch::IValue>(env, jhandle)->isTensorList();
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsList(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN()
  return utils::GetPointerFromJHandle<torch::IValue>(env, jhandle)->isList();
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsMap(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN()
  return utils::GetPointerFromJHandle<torch::IValue>(env, jhandle)->isGenericDict();
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsTuple(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN()
  return utils::GetPointerFromJHandle<torch::IValue>(env, jhandle)->isTuple();
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteIValue(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = utils::GetPointerFromJHandle<torch::IValue>(env, jhandle);
  delete ivalue_ptr;
  API_END()
}
