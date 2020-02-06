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
#include <torch/torch.h>
#include <torch/script.h>

// The file is the implementation for PyTorch inference operations

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleLoad
  (JNIEnv* env, jobject jthis, jstring jpath) {
  const std::string path_string((env)->GetStringUTFChars(jpath, JNI_FALSE));
  const torch::jit::script::Module module = torch::jit::load(path_string);
  const auto* module_ptr = new torch::jit::script::Module(module);
  return utils::CreatePointer<torch::jit::script::Module>(env, module_ptr);
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleEval
  (JNIEnv* env, jobject jthis, jobject module_handle) {
  auto* module_ptr = utils::GetPointerFromJHandle<torch::jit::script::Module>(env, module_handle);
  module_ptr->eval();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleForward
  (JNIEnv* env, jobject jthis, jobject module_handle, jobjectArray jivalue_ptr_array) {
  auto ivalue_vec = std::vector<c10::IValue>();
  for (auto i = 0; i < env->GetArrayLength(jivalue_ptr_array); ++i) {
    auto ivalue = utils::GetPointerFromJHandle<c10::IValue>(env, env->GetObjectArrayElement(jivalue_ptr_array, i));
    ivalue_vec.emplace_back(*ivalue);
  }
  auto* module_ptr = utils::GetPointerFromJHandle<torch::jit::script::Module>(env, module_handle);
  const auto* result_ptr = new c10::IValue(module_ptr->forward(ivalue_vec));
  return utils::CreatePointer<c10::IValue>(env, result_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueCreateFromTensor
  (JNIEnv* env, jobject jthis, jobject jhandle) {
  const auto* ivalue_ptr = new c10::IValue(
    *utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle));
  return utils::CreatePointer<c10::IValue>(env, ivalue_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToTensor
  (JNIEnv* env, jobject jthis, jobject jhandle) {
  auto* tensor_ptr = new torch::Tensor(
    utils::GetPointerFromJHandle<c10::IValue>(env, jhandle)->toTensor());
  return utils::CreatePointer<torch::Tensor>(env, tensor_ptr);
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteModule
  (JNIEnv* env, jobject jthis, jobject jhandle) {
  const auto* module_ptr = utils::GetPointerFromJHandle<const torch::jit::script::Module>(env, jhandle);
  delete module_ptr;
}
