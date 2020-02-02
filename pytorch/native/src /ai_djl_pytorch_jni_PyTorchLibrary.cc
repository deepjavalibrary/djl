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


JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSizes
  (JNIEnv* env, jobject jthis, jobject jhandle) {
  const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  jlongArray size = env->NewLongArray(tensor_ptr->dim());
  env->SetLongArrayRegion(size, 0, tensor_ptr->dim(),
    reinterpret_cast<const jlong*>(tensor_ptr->sizes().data()));
  return size;
}

JNIEXPORT jint JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDType
  (JNIEnv* env, jobject jthis, jobject jhandle) {
  const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  return utils::GetDTypeFromScalarType(tensor_ptr->scalar_type());
}

JNIEXPORT jintArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDevice
  (JNIEnv* env, jobject jthis, jobject jhandle) {
  const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  jintArray result = env->NewIntArray(2);
  if (nullptr == result) {
    return nullptr;
  }
  int temp_device[] = {static_cast<int>(tensor_ptr->device().type()), tensor_ptr->device().index()};
  env->SetIntArrayRegion(result, 0, 2, temp_device);
  return result;
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchEmpty(
  JNIEnv* env,
  jobject jthis,
  jlongArray jshape,
  jint jdtype,
  jint jlayout,
  jintArray jdevice,
  jboolean jrequired_grad) {
  const auto shape_vec = utils::GetShapeVecFromJShape(env, jshape);
  const auto options = utils::GetTensorOptions(env, jdtype, jlayout, jdevice, jrequired_grad);
  const torch::Tensor* tensor_ptr = new torch::Tensor(torch::empty(shape_vec, options));
  return utils::CreatePointer<torch::Tensor>(env, tensor_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchZeros(
  JNIEnv* env,
  jobject jthis,
  jlongArray jshape,
  jint jdtype,
  jint jlayout,
  jintArray jdevice,
  jboolean jrequired_grad) {
  const auto shape_vec = utils::GetShapeVecFromJShape(env, jshape);
  const auto options = utils::GetTensorOptions(env, jdtype, jlayout, jdevice, jrequired_grad);
  const torch::Tensor* tensor_ptr = new torch::Tensor(torch::zeros(shape_vec, options));
  return utils::CreatePointer<torch::Tensor>(env, tensor_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchOnes(
  JNIEnv* env,
  jobject jthis,
  jlongArray jshape,
  jint jdtype,
  jint jlayout,
  jintArray jdevice,
  jboolean jrequired_grad) {
  const auto shape_vec = utils::GetShapeVecFromJShape(env, jshape);
  const auto options = utils::GetTensorOptions(env, jdtype, jlayout, jdevice, jrequired_grad);
  const auto* tensor_ptr = new torch::Tensor(torch::ones(shape_vec, options));
  return utils::CreatePointer<torch::Tensor>(env, tensor_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchFromBlob(
  JNIEnv* env,
  jobject jthis,
  jobject jbuffer,
  jlongArray jshape,
  jint jdtype,
  jint jlayout,
  jintArray jdevice,
  jboolean jrequired_grad) {
  const auto shape_vec = utils::GetShapeVecFromJShape(env, jshape);
  const auto options = utils::GetTensorOptions(env, jdtype, jlayout, jdevice, jrequired_grad);
  const torch::Tensor* tensor_ptr =
    new torch::Tensor(torch::from_blob(
      env->GetDirectBufferAddress(jbuffer),
      shape_vec,
      options));
  return utils::CreatePointer<torch::Tensor>(env, tensor_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDataPtr
  (JNIEnv* env, jobject jthis, jobject jhandle) {
  const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  jobject buf = env->NewDirectByteBuffer(tensor_ptr->data_ptr(), tensor_ptr->nbytes());
  return buf;
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleLoad
  (JNIEnv* env, jobject jthis, jstring jpath) {
  std::string path_string((env)->GetStringUTFChars(jpath, JNI_FALSE));
  torch::jit::script::Module module = torch::jit::load(path_string);
  const auto* module_ptr = new torch::jit::script::Module(module);
  return utils::CreatePointer<torch::jit::script::Module>(env, module_ptr);
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleEval
  (JNIEnv* env, jobject jthis, jobject module_handle) {
  auto* module_ptr = utils::GetPointerFromJHandle<torch::jit::script::Module>(env, module_handle);
  module_ptr->eval();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleForward
  (JNIEnv* env, jobject jthis, jobject module_handle, jobjectArray ivalue_handle_array) {
  auto ivalue_array = std::vector<c10::IValue>();
  for (int i = 0; i < env->GetArrayLength(ivalue_handle_array); ++i) {
    auto ivalue = utils::GetPointerFromJHandle<c10::IValue>(env, env->GetObjectArrayElement(ivalue_handle_array, i));
    ivalue_array.emplace_back(*ivalue);
  }
  auto* module_ptr = utils::GetPointerFromJHandle<torch::jit::script::Module>(env, module_handle);
  const auto* result_handle = new c10::IValue(module_ptr->forward(ivalue_array));
  return utils::CreatePointer<c10::IValue>(env, result_handle);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueCreateFromTensor
  (JNIEnv* env, jobject jthis, jobject tensor_handle) {
  const auto* ivalue_ptr = new c10::IValue(
    *utils::GetPointerFromJHandle<torch::Tensor>(env, tensor_handle));
  return utils::CreatePointer<c10::IValue>(env, ivalue_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToTensor
  (JNIEnv* env, jobject jthis, jobject ivalue_handle) {
  auto* tensor_ptr = new torch::Tensor(
    utils::GetPointerFromJHandle<c10::IValue>(env, ivalue_handle)->toTensor());
  return utils::CreatePointer<torch::Tensor>(env, tensor_ptr);
}
