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


JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchVersion
  (JNIEnv *env, jobject jthis) {
  auto tensor = torch::empty(1);
  return tensor._version();
}

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
  const auto shape_vec = utils::GetVecFromJLongArray(env, jshape);
  const auto options = utils::CreateTensorOptions(env, jdtype, jlayout, jdevice, jrequired_grad);
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
  const auto shape_vec = utils::GetVecFromJLongArray(env, jshape);
  const auto options = utils::CreateTensorOptions(env, jdtype, jlayout, jdevice, jrequired_grad);
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
  const auto shape_vec = utils::GetVecFromJLongArray(env, jshape);
  const auto options = utils::CreateTensorOptions(env, jdtype, jlayout, jdevice, jrequired_grad);
  const auto* tensor_ptr = new torch::Tensor(torch::ones(shape_vec, options));
  return utils::CreatePointer<torch::Tensor>(env, tensor_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchReshape
  (JNIEnv* env, jobject jthis, jobject jhandle, jlongArray jshape) {
  const auto shape_vec = utils::GetVecFromJLongArray(env, jshape);
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->reshape(shape_vec));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSoftmax
  (JNIEnv* env, jobject jthis, jobject jhandle, jint jdim, jint jdtype) {
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->softmax(jdim, utils::GetScalarTypeFromDType(jdtype)));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchArgMax__Lai_djl_pytorch_jni_Pointer_2
  (JNIEnv* env, jobject jthis, jobject jhandle) {
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->argmax());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchArgMax__Lai_djl_pytorch_jni_Pointer_2IZ
  (JNIEnv* env, jobject jthis, jobject jhandle, jint jdim, jboolean jkeep_dim) {
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->argmax(jdim, jkeep_dim == JNI_TRUE));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchUpsampleBilinear2d
  (JNIEnv* env, jobject jthis, jobject jhandle, jlongArray jsize, jboolean jalign_corners) {
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto size_vec = utils::GetVecFromJLongArray(env, jsize);
  const auto* result_ptr = new torch::Tensor(torch::upsample_bilinear2d(*tensor_ptr, size_vec, jalign_corners == JNI_TRUE));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSplit__Lai_djl_pytorch_jni_Pointer_2JJ
  (JNIEnv *env, jobject jthis, jobject jhandle, jlong jsize, jlong jdim) {
  const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  auto tensors = tensor_ptr->split(jsize, jdim);
  jobjectArray jarray = env->NewObjectArray(tensors.size(), env->FindClass(utils::POINTER_CLASS), nullptr);
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto* element_ptr = new torch::Tensor(tensors.at(i));
    auto ptr = utils::CreatePointer<torch::Tensor>(env, element_ptr);
    env->SetObjectArrayElement(jarray, i, ptr);
  }
  return jarray;
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSplit__Lai_djl_pytorch_jni_Pointer_2_3IJ
  (JNIEnv *env, jobject jthis, jobject jhandle, jlongArray jindices, jlong jdim) {
  const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  auto indices = env->GetLongArrayElements(jindices, JNI_FALSE);
  auto tensors = tensor_ptr->split_with_sizes(c10::IntArrayRef(*indices), jdim);
  jobjectArray jarray = env->NewObjectArray(tensors.size(), env->FindClass(utils::POINTER_CLASS), nullptr);
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto* element_ptr = new torch::Tensor(tensors.at(i));
    auto ptr = utils::CreatePointer<torch::Tensor>(env, element_ptr);
    env->SetObjectArrayElement(jarray, i, ptr);
  }
  return jarray;
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
  const auto shape_vec = utils::GetVecFromJLongArray(env, jshape);
  const auto options = utils::CreateTensorOptions(env, jdtype, jlayout, jdevice, jrequired_grad);
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

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteTensor
  (JNIEnv* env, jobject jthis, jobject jhandle) {
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  delete tensor_ptr;
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteModule
  (JNIEnv* env, jobject jthis, jobject jhandle) {
  const auto* module_ptr = utils::GetPointerFromJHandle<const torch::jit::script::Module>(env, jhandle);
  delete module_ptr;
}

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
