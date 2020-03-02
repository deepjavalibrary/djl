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
#include <torch/torch.h>

#include "../build/include/ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_utils.h"

// The file is the implementation for PyTorch inference operations

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleLoad(
    JNIEnv* env, jobject jthis, jstring jpath, jintArray jarray) {
  const std::string path_string((env)->GetStringUTFChars(jpath, JNI_FALSE));
  const c10::Device device = utils::GetDeviceFromJDevice(env, jarray);
  const torch::jit::script::Module module = torch::jit::load(path_string, device);
  const auto* module_ptr = new torch::jit::script::Module(module);
  return utils::CreatePointer<torch::jit::script::Module>(env, module_ptr);
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleEval(
    JNIEnv* env, jobject jthis, jobject module_handle) {
  auto* module_ptr = utils::GetPointerFromJHandle<torch::jit::script::Module>(env, module_handle);
  module_ptr->eval();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleForward(
    JNIEnv* env, jobject jthis, jobject module_handle, jobjectArray jivalue_ptr_array) {
  auto ivalue_vec = std::vector<c10::IValue>();
  for (auto i = 0; i < env->GetArrayLength(jivalue_ptr_array); ++i) {
    auto ivalue = utils::GetPointerFromJHandle<c10::IValue>(env, env->GetObjectArrayElement(jivalue_ptr_array, i));
    ivalue_vec.emplace_back(*ivalue);
  }
  auto* module_ptr = utils::GetPointerFromJHandle<torch::jit::script::Module>(env, module_handle);
  const auto* result_ptr = new c10::IValue(module_ptr->forward(ivalue_vec));
  return utils::CreatePointer<c10::IValue>(env, result_ptr);
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteModule(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  const auto* module_ptr = utils::GetPointerFromJHandle<const torch::jit::script::Module>(env, jhandle);
  delete module_ptr;
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueCreateFromTensor(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  const auto* ivalue_ptr = new c10::IValue(*utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle));
  return utils::CreatePointer<c10::IValue>(env, ivalue_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToTensor(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  auto* tensor_ptr = new torch::Tensor(utils::GetPointerFromJHandle<c10::IValue>(env, jhandle)->toTensor());
  return utils::CreatePointer<torch::Tensor>(env, tensor_ptr);
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToList(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  auto* ivalue_ptr = utils::GetPointerFromJHandle<c10::IValue>(env, jhandle);
  auto ivalue_list = ivalue_ptr->toGenericList();
  jobjectArray jarray = env->NewObjectArray(ivalue_list.size(), env->FindClass(utils::POINTER_CLASS), nullptr);
  for (size_t i = 0; i < ivalue_list.size(); ++i) {
    const auto* element_ptr = new c10::IValue(ivalue_list.get(i));
    auto ptr = utils::CreatePointer<c10::IValue>(env, element_ptr);
    env->SetObjectArrayElement(jarray, i, ptr);
  }
  return jarray;
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToListFromTuple(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  auto* ivalue_ptr = utils::GetPointerFromJHandle<c10::IValue>(env, jhandle);
  auto ivalue_list = ivalue_ptr->toTuple()->elements();
  jobjectArray jarray = env->NewObjectArray(ivalue_list.size(), env->FindClass(utils::POINTER_CLASS), nullptr);
  for (size_t i = 0; i < ivalue_list.size(); ++i) {
    const auto* element_ptr = new c10::IValue(ivalue_list.at(i));
    auto ptr = utils::CreatePointer<c10::IValue>(env, element_ptr);
    env->SetObjectArrayElement(jarray, i, ptr);
  }
  return jarray;
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToTensorList(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  auto* ivalue_ptr = utils::GetPointerFromJHandle<c10::IValue>(env, jhandle);
  auto ivalue_list = ivalue_ptr->toTensorList();
  jobjectArray jarray = env->NewObjectArray(ivalue_list.size(), env->FindClass(utils::POINTER_CLASS), nullptr);
  for (size_t i = 0; i < ivalue_list.size(); ++i) {
    const auto* element_ptr = new torch::Tensor(ivalue_list.get(i));
    auto ptr = utils::CreatePointer<torch::Tensor>(env, element_ptr);
    env->SetObjectArrayElement(jarray, i, ptr);
  }
  return jarray;
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToMap(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  auto* ivalue_ptr = utils::GetPointerFromJHandle<c10::IValue>(env, jhandle);
  auto dict = ivalue_ptr->toGenericDict();
  jobjectArray jarray = env->NewObjectArray(dict.size() * 2, env->FindClass(utils::POINTER_CLASS), nullptr);
  int array_iter = 0;
  for (auto it = dict.begin(); it != dict.end(); ++it) {
    const auto* key_ptr = new c10::IValue(it->key());
    auto ptr = utils::CreatePointer<c10::IValue>(env, key_ptr);
    env->SetObjectArrayElement(jarray, array_iter++, ptr);
    const auto* value_ptr = new c10::IValue(it->value());
    ptr = utils::CreatePointer<c10::IValue>(env, value_ptr);
    env->SetObjectArrayElement(jarray, array_iter++, ptr);
  }
  return jarray;
}

JNIEXPORT jstring JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToString(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  auto* ivalue_ptr = utils::GetPointerFromJHandle<c10::IValue>(env, jhandle);
  return env->NewStringUTF(ivalue_ptr->toString()->string().c_str());
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsString(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  return utils::GetPointerFromJHandle<c10::IValue>(env, jhandle)->isString();
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsTensor(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  return utils::GetPointerFromJHandle<c10::IValue>(env, jhandle)->isTensor();
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsTensorList(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  return utils::GetPointerFromJHandle<c10::IValue>(env, jhandle)->isTensorList();
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsList(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  return utils::GetPointerFromJHandle<c10::IValue>(env, jhandle)->isGenericList();
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsMap(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  return utils::GetPointerFromJHandle<c10::IValue>(env, jhandle)->isGenericDict();
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsTuple(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  return utils::GetPointerFromJHandle<c10::IValue>(env, jhandle)->isTuple();
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteIValue(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  auto* ivalue_ptr = utils::GetPointerFromJHandle<c10::IValue>(env, jhandle);
  delete ivalue_ptr;
}
