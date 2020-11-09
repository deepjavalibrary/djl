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

// The file is the implementation for PyTorch inference operations

struct JITCallGuard {
  torch::autograd::AutoGradMode no_autograd_guard{false};
  torch::NoGradGuard no_grad;
};

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleLoad(
    JNIEnv* env, jobject jthis, jstring jpath, jintArray jarray) {
  API_BEGIN()
  const std::string path = utils::GetStringFromJString(env, jpath);
  const torch::Device device = utils::GetDeviceFromJDevice(env, jarray);
  const torch::jit::script::Module module = torch::jit::load(path, device);
  const auto* module_ptr = new torch::jit::script::Module(module);
  return utils::CreatePointer<torch::jit::script::Module>(env, module_ptr);
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleEval(
    JNIEnv* env, jobject jthis, jobject module_handle) {
  API_BEGIN()
  auto* module_ptr = utils::GetPointerFromJHandle<torch::jit::script::Module>(env, module_handle);
  module_ptr->eval();
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleTrain(
    JNIEnv* env, jobject jthis, jobject module_handle) {
  API_BEGIN()
  auto* module_ptr = utils::GetPointerFromJHandle<torch::jit::script::Module>(env, module_handle);
  module_ptr->train(true);
  API_END()
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleForward(
    JNIEnv* env, jobject jthis, jobject module_handle, jobjectArray jivalue_ptrs, jboolean jis_train) {
  API_BEGIN()
  auto* module_ptr = utils::GetPointerFromJHandle<torch::jit::script::Module>(env, module_handle);
  auto len = static_cast<size_t>(env->GetArrayLength(jivalue_ptrs));
  std::vector<torch::IValue> inputs;
  inputs.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    auto* ivalue_ptr =
        utils::GetPointerFromJHandle<const torch::IValue>(env, env->GetObjectArrayElement(jivalue_ptrs, i));
    inputs.emplace_back(*ivalue_ptr);
  }
  auto output = [&]() {
    if (jis_train) {
      return module_ptr->forward(inputs);
    }
    // disable autograd
    JITCallGuard guard;
    return module_ptr->forward(inputs);
  }();

  // release resource
  // each IValue is created by new, free the memory after the inference
  for (size_t i = 0; i < len; ++i) {
    auto* ivalue_ptr =
        utils::GetPointerFromJHandle<const torch::IValue>(env, env->GetObjectArrayElement(jivalue_ptrs, i));
    delete ivalue_ptr;
  }
  env->DeleteLocalRef(jivalue_ptrs);

  const auto* result_ptr = new torch::IValue(output);
  return utils::CreatePointer<torch::IValue>(env, result_ptr);
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteModule(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN()
  const auto* module_ptr = utils::GetPointerFromJHandle<const torch::jit::script::Module>(env, jhandle);
  delete module_ptr;
  API_END()
}
