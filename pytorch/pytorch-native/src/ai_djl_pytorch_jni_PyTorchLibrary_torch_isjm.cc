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
#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_error.h"
#include "djl_pytorch_jni_utils.h"

// The file is the implementation for PyTorch tensor indexing, slicing, joining, mutating ops

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchReshape(
    JNIEnv* env, jobject jthis, jobject jhandle, jlongArray jshape) {
  API_BEGIN();
  const auto shape_vec = utils::GetVecFromJLongArray(env, jshape);
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->reshape(shape_vec));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSqueeze__Lai_djl_pytorch_jni_Pointer_2(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->squeeze());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSqueeze__Lai_djl_pytorch_jni_Pointer_2J(
    JNIEnv* env, jobject jthis, jobject jhandle, jlong jdim) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->squeeze(jdim));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchUnsqueeze(
    JNIEnv* env, jobject jthis, jobject jhandle, jlong jdim) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->unsqueeze(jdim));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchExpand(
    JNIEnv* env, jobject jthis, jobject jhandle, jlongArray jshape) {
  API_BEGIN();
  const auto shape_vec = utils::GetVecFromJLongArray(env, jshape);
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->expand(shape_vec));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchStack(
    JNIEnv* env, jobject jthis, jobjectArray jhandles, jlong jdim) {
  API_BEGIN();
  const std::vector<torch::Tensor> tensor_vec = utils::GetObjectVecFromJHandles<torch::Tensor>(env, jhandles);
  const torch::Tensor* result_ptr = new torch::Tensor(torch::stack(tensor_vec, jdim));
  return utils::CreatePointer<const torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCat(
    JNIEnv* env, jobject jthis, jobjectArray jhandles, jlong jdim) {
  API_BEGIN();
  const std::vector<torch::Tensor> tensor_vec = utils::GetObjectVecFromJHandles<torch::Tensor>(env, jhandles);
  const torch::Tensor* result_ptr = new torch::Tensor(torch::cat(tensor_vec, jdim));
  return utils::CreatePointer<const torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSplit__Lai_djl_pytorch_jni_Pointer_2JJ(
    JNIEnv* env, jobject jthis, jobject jhandle, jlong jsize, jlong jdim) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  auto tensors = tensor_ptr->split(jsize, jdim);
  jobjectArray jarray = env->NewObjectArray(tensors.size(), env->FindClass(utils::POINTER_CLASS), nullptr);
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto* element_ptr = new torch::Tensor(tensors.at(i));
    auto ptr = utils::CreatePointer<torch::Tensor>(env, element_ptr);
    env->SetObjectArrayElement(jarray, i, ptr);
  }
  return jarray;
  API_END();
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSplit__Lai_djl_pytorch_jni_Pointer_2_3JJ(
    JNIEnv* env, jobject jthis, jobject jhandle, jlongArray jindices, jlong jdim) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  const std::vector<int64_t> indices = utils::GetVecFromJLongArray(env, jindices);
  auto tensors = tensor_ptr->split_with_sizes(indices, jdim);
  jobjectArray jarray = env->NewObjectArray(tensors.size(), env->FindClass(utils::POINTER_CLASS), nullptr);
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto* element_ptr = new torch::Tensor(tensors.at(i));
    auto ptr = utils::CreatePointer<torch::Tensor>(env, element_ptr);
    env->SetObjectArrayElement(jarray, i, ptr);
  }
  return jarray;
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchPermute(
    JNIEnv* env, jobject jthis, jobject jhandle, jlongArray jdims) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const std::vector<int64_t> dims = utils::GetVecFromJLongArray(env, jdims);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->permute(dims));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchFlip(
        JNIEnv* env, jobject jthis, jobject jhandle, jlongArray jdims) {
    API_BEGIN();
        const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
        const std::vector<int64_t> dims = utils::GetVecFromJLongArray(env, jdims);
        const auto* result_ptr = new torch::Tensor(tensor_ptr->flip(dims));
        return utils::CreatePointer<torch::Tensor>(env, result_ptr);
    API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchTranspose(
    JNIEnv* env, jobject jthis, jobject jhandle, jlong jdim1, jlong jdim2) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->transpose(jdim1, jdim2));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchRepeat(
  JNIEnv* env, jobject jthis, jobject jhandle, jlongArray jrepeats) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
    const std::vector<int64_t> repeats = utils::GetVecFromJLongArray(env, jrepeats);
    const torch::Tensor* result_ptr = new torch::Tensor(tensor_ptr->repeat(repeats));
    return utils::CreatePointer<const torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchRepeatInterleave(
  JNIEnv* env, jobject jthis, jobject jhandle, jlong jrepeats, jlong jdim) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
    const torch::Tensor* result_ptr = new torch::Tensor(tensor_ptr->repeat_interleave(jrepeats, jdim));
    return utils::CreatePointer<const torch::Tensor>(env, result_ptr);
  API_END();
}
