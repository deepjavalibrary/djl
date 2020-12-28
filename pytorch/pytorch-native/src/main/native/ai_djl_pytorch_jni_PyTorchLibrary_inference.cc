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

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleLoad(
    JNIEnv* env, jobject jthis, jstring jpath, jintArray jarray, jobjectArray jefnames, jobjectArray jefvalues) {
  API_BEGIN()
  const std::string path = utils::jni::GetStringFromJString(env, jpath);
  const torch::Device device = utils::GetDeviceFromJDevice(env, jarray);
  std::unordered_map<std::string, std::string> map;
  size_t len = static_cast<size_t>(env->GetArrayLength(jefnames));
  for (size_t i = 0; i < len; ++i) {
    auto jname = (jstring)env->GetObjectArrayElement(jefnames, i);
    auto name = utils::jni::GetStringFromJString(env, jname);
    map[name] = "";
  }
  const torch::jit::script::Module module = torch::jit::load(path, device, map);
  const auto* module_ptr = new torch::jit::script::Module(module);
  for (size_t i = 0; i < len; ++i) {
    auto jname = (jstring)env->GetObjectArrayElement(jefnames, i);
    auto name = utils::jni::GetStringFromJString(env, jname);
    env->SetObjectArrayElement(jefvalues, i, env->NewStringUTF(map[name].c_str()));
  }
  return reinterpret_cast<uintptr_t>(module_ptr);
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleEval(
    JNIEnv* env, jobject jthis, jlong module_handle) {
  API_BEGIN()
  auto* module_ptr = reinterpret_cast<torch::jit::script::Module*>(module_handle);
  module_ptr->eval();
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleTrain(
    JNIEnv* env, jobject jthis, jlong module_handle) {
  API_BEGIN()
  auto* module_ptr = reinterpret_cast<torch::jit::script::Module*>(module_handle);
  module_ptr->train(true);
  API_END()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleForward(
    JNIEnv* env, jobject jthis, jlong module_handle, jlongArray jivalue_ptrs, jboolean jis_train) {
  API_BEGIN()
  auto* module_ptr = reinterpret_cast<torch::jit::script::Module*>(module_handle);
  size_t len = env->GetArrayLength(jivalue_ptrs);
  jlong* jptrs = env->GetLongArrayElements(jivalue_ptrs, JNI_FALSE);
  std::vector<torch::IValue> inputs;
  inputs.reserve(len);
  for (auto i = 0; i < len; ++i) {
    inputs.emplace_back(*reinterpret_cast<torch::IValue*>(jptrs[i]));
  }
  torch::IValue output = [&]() {
    if (jis_train) {
      return module_ptr->forward(inputs);
    }
    // disable autograd
    JITCallGuard guard;
    return module_ptr->forward(inputs);
  }();
  // release resource
  // each IValue is created by new, free the memory after the inference
  for (auto i = 0; i < len; ++i) {
    delete reinterpret_cast<torch::IValue*>(jptrs[i]);
  }
  env->ReleaseLongArrayElements(jivalue_ptrs, jptrs, utils::jni::RELEASE_MODE);
  const auto* result_ptr = new torch::IValue(output);
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteModule(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* module_ptr = reinterpret_cast<torch::jit::script::Module*>(jhandle);
  delete module_ptr;
  API_END()
}
