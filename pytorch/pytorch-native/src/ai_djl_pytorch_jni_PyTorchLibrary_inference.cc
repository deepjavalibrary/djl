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

#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_error.h"
#include "djl_pytorch_jni_utils.h"

// The file is the implementation for PyTorch inference operations

struct JITCallGuard {
  // disable autograd by default
  torch::autograd::AutoGradMode no_autograd_guard{false};
};

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleLoad(
    JNIEnv* env, jobject jthis, jstring jpath, jintArray jarray) {
  API_BEGIN();
  const std::string path_string((env)->GetStringUTFChars(jpath, JNI_FALSE));
  const c10::Device device = utils::GetDeviceFromJDevice(env, jarray);
  const torch::jit::script::Module module = torch::jit::load(path_string, device);
  const auto* module_ptr = new torch::jit::script::Module(module);
  return utils::CreatePointer<torch::jit::script::Module>(env, module_ptr);
  API_END();
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleEval(
    JNIEnv* env, jobject jthis, jobject module_handle) {
  auto* module_ptr = utils::GetPointerFromJHandle<torch::jit::script::Module>(env, module_handle);
  module_ptr->eval();
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleTrain(
  JNIEnv* env, jobject jthis, jobject module_handle) {
  auto* module_ptr = utils::GetPointerFromJHandle<torch::jit::script::Module>(env, module_handle);
  module_ptr->train(true);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleForward(
    JNIEnv* env, jobject jthis, jobject module_handle, jobjectArray tensor_ptrs, jboolean isTrain) {
  API_BEGIN();
  auto ivalue_vec = std::vector<c10::IValue>();
  size_t len = static_cast<size_t>(env->GetArrayLength(tensor_ptrs));
  ivalue_vec.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    auto* tensor_ptr =
        utils::GetPointerFromJHandle<const torch::Tensor>(env, env->GetObjectArrayElement(tensor_ptrs, i));
    // IValue and Tensor are interchangeable
    ivalue_vec.emplace_back(*tensor_ptr);
  }
  env->DeleteLocalRef(tensor_ptrs);
  auto* module_ptr = utils::GetPointerFromJHandle<torch::jit::script::Module>(env, module_handle);
  auto output = [&]() {
    if (isTrain) {
        return module_ptr->forward(ivalue_vec);
    }
    // disable autograd
    JITCallGuard guard;
    return module_ptr->forward(ivalue_vec);
  }();
  const auto* result_ptr = new c10::IValue(output);
  return utils::CreatePointer<c10::IValue>(env, result_ptr);
  API_END();
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteModule(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  const auto* module_ptr = utils::GetPointerFromJHandle<const torch::jit::script::Module>(env, jhandle);
  delete module_ptr;
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueCreateFromTensor(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* ivalue_ptr = new c10::IValue(*utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle));
  return utils::CreatePointer<c10::IValue>(env, ivalue_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToTensor(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  auto* tensor_ptr = new torch::Tensor(utils::GetPointerFromJHandle<c10::IValue>(env, jhandle)->toTensor());
  return utils::CreatePointer<torch::Tensor>(env, tensor_ptr);
  API_END();
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToListFromTuple(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  auto* ivalue_ptr = utils::GetPointerFromJHandle<c10::IValue>(env, jhandle);
  auto ivalue_list = ivalue_ptr->toTuple()->elements();
  jobjectArray jarray = env->NewObjectArray(ivalue_list.size(), env->FindClass(utils::POINTER_CLASS), nullptr);
  for (size_t i = 0; i < ivalue_list.size(); ++i) {
    const auto* element_ptr = new c10::IValue(ivalue_list.at(i));
    auto ptr = utils::CreatePointer<c10::IValue>(env, element_ptr);
    env->SetObjectArrayElement(jarray, i, ptr);
  }
  return jarray;
  API_END();
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToTensorList(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  auto* ivalue_ptr = utils::GetPointerFromJHandle<c10::IValue>(env, jhandle);
  auto ivalue_list = ivalue_ptr->toTensorList();
  jobjectArray jarray = env->NewObjectArray(ivalue_list.size(), env->FindClass(utils::POINTER_CLASS), nullptr);
  for (size_t i = 0; i < ivalue_list.size(); ++i) {
    const auto* element_ptr = new torch::Tensor(ivalue_list.get(i));
    auto ptr = utils::CreatePointer<torch::Tensor>(env, element_ptr);
    env->SetObjectArrayElement(jarray, i, ptr);
  }
  return jarray;
  API_END();
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToList(
  JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
    auto* ivalue_ptr = utils::GetPointerFromJHandle<c10::IValue>(env, jhandle);
    auto ivalue_list = ivalue_ptr->toList();
    jobjectArray jarray = env->NewObjectArray(ivalue_list.size(), env->FindClass(utils::POINTER_CLASS), nullptr);
    for (size_t i = 0; i < ivalue_list.size(); ++i) {
      const auto* element_ptr = new c10::IValue(ivalue_list.get(i));
      auto ptr = utils::CreatePointer<c10::IValue>(env, element_ptr);
      env->SetObjectArrayElement(jarray, i, ptr);
    }
    return jarray;
  API_END();
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToMap(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
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
  API_END();
}

JNIEXPORT jstring JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToString(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  auto* ivalue_ptr = utils::GetPointerFromJHandle<c10::IValue>(env, jhandle);
  return env->NewStringUTF(ivalue_ptr->toString()->string().c_str());
  API_END();
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsString(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  return utils::GetPointerFromJHandle<c10::IValue>(env, jhandle)->isString();
  API_END();
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsTensor(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  return utils::GetPointerFromJHandle<c10::IValue>(env, jhandle)->isTensor();
  API_END();
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsTensorList(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  return utils::GetPointerFromJHandle<c10::IValue>(env, jhandle)->isTensorList();
  API_END();
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsList(
  JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
    return utils::GetPointerFromJHandle<c10::IValue>(env, jhandle)->isList();
  API_END();
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsMap(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  return utils::GetPointerFromJHandle<c10::IValue>(env, jhandle)->isGenericDict();
  API_END();
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsTuple(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  return utils::GetPointerFromJHandle<c10::IValue>(env, jhandle)->isTuple();
  API_END();
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteIValue(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  auto* ivalue_ptr = utils::GetPointerFromJHandle<c10::IValue>(env, jhandle);
  delete ivalue_ptr;
}
