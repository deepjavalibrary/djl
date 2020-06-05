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

// The file is the implementation for PyTorch tensor core functionality operation

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSizes(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  jlongArray size = env->NewLongArray(tensor_ptr->dim());
  env->SetLongArrayRegion(size, 0, tensor_ptr->dim(), reinterpret_cast<const jlong*>(tensor_ptr->sizes().data()));
  return size;
  API_END();
}

JNIEXPORT jint JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDType(JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  return utils::GetDTypeFromScalarType(tensor_ptr->scalar_type());
  API_END();
}

JNIEXPORT jintArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDevice(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  jclass jexception = env->FindClass("java/lang/NullPointerException");
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  jintArray result = env->NewIntArray(2);
  if (result == nullptr) {
    env->ThrowNew(jexception, "Unable to create int array");
  }
  jint temp_device[] = {static_cast<int>(tensor_ptr->device().type()), tensor_ptr->device().index()};
  env->SetIntArrayRegion(result, 0, 2, temp_device);
  return result;
  API_END();
}

JNIEXPORT jint JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLayout(JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  jclass jexception = env->FindClass("java/lang/IllegalStateException");
  const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  auto layout = tensor_ptr->layout();
  switch (layout) {
    case torch::kStrided:
      return 0;
    case torch::kSparse:
      return 1;
    case torch::kMkldnn:
      return 2;
    default:
      env->ThrowNew(jexception, "Internal PyTorch error, layout should only have kStrided, kSparse or kMkldnn");
  }
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchTo(
    JNIEnv* env, jobject jthis, jobject jhandle, jint jdtype, jintArray jdevice, jboolean jcopy) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  const auto device = utils::GetDeviceFromJDevice(env, jdevice);
  const auto* result_ptr =
      new torch::Tensor(tensor_ptr->to(device, utils::GetScalarTypeFromDType(jdtype), false, jcopy == JNI_TRUE));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_tensorClone(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->clone());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchIndex(JNIEnv* env, jobject jthis, jobject jhandle,
    jlongArray jmin_indices, jlongArray jmax_indices, jlongArray jstep_indices) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  auto indices = utils::CreateTensorIndex(env, jmin_indices, jmax_indices, jstep_indices);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->index(indices));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchIndexPut(
  JNIEnv* env, jobject jthis, jobject jhandle, jobject jvalue_handle, jlongArray jmin_indices, jlongArray jmax_indices, jlongArray jstep_indices) {
    auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
    const auto* value_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jvalue_handle);
    auto indices = utils::CreateTensorIndex(env, jmin_indices, jmax_indices, jstep_indices);
    tensor_ptr->index_put_(indices, *value_ptr);
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSet(
  JNIEnv* env, jobject jthis, jobject jself, jobject jreplace) {
    const auto* self_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jself);
    const auto* replace_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jreplace);
    self_ptr->set_(*replace_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSlice(
    JNIEnv* env, jobject jthis, jobject jhandle, jlong jdim, jlong jstart, jlong jend, jlong jstep) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->slice(jdim, jstart, jend, jstep));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchGather(
  JNIEnv* env, jobject jthis, jobject jhandle, jobject jindex_handle, jlong dim, jboolean sparse_grad) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
    const auto* index_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jindex_handle);
    const auto* result_ptr = new torch::Tensor(tensor_ptr->gather(dim, *index_ptr, sparse_grad));
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}


JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMaskedSelect(
    JNIEnv* env, jobject jthis, jobject jhandle, jobject jmasked_handle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  const auto* index_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jmasked_handle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->masked_select(*index_ptr));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMaskedPut(
  JNIEnv* env, jobject jthis, jobject jhandle, jobject jvalue_handle, jobject jmasked_handle) {
    const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
    const auto* index_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jmasked_handle);
    const auto* value_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jvalue_handle);
    tensor_ptr->masked_fill_(*index_ptr, *value_ptr);
}

JNIEXPORT jbyteArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDataPtr(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  // sparse and mkldnn are required to be converted to dense to access data ptr
  auto tensor = (tensor_ptr->is_sparse() || tensor_ptr->is_mkldnn()) ? tensor_ptr->to_dense() : *tensor_ptr;
  tensor = (tensor.is_contiguous()) ? tensor : tensor.contiguous();
  jbyteArray result = env->NewByteArray(tensor.nbytes());
  env->SetByteArrayRegion(result, 0, tensor.nbytes(), static_cast<const jbyte*>(tensor.data_ptr()));
  return result;
  API_END();
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteTensor(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  delete tensor_ptr;
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchToSparse(
  JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->to_sparse());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchToDense(
  JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
    const auto* result_ptr = new torch::Tensor(tensor_ptr->to_dense());
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchRequiresGrad(
  JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
    return tensor_ptr->requires_grad();
  API_END();
}

JNIEXPORT jstring JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchGradFnName(
  JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
    return env->NewStringUTF(tensor_ptr->grad_fn()->name().c_str());
  API_END();
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAttachGrad(
  JNIEnv* env, jobject jthis, jobject jhandle) {
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
    tensor_ptr->requires_grad_(true);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchGrad(
  JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
    const auto* result_ptr = new torch::Tensor(tensor_ptr->grad());
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDetachGrad(
  JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
    const auto* result_ptr = new torch::Tensor(tensor_ptr->detach());
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchBackward(
  JNIEnv* env, jobject jthis, jobject jhandle, jobject jgrad_handle, jboolean keep_graph, jboolean create_graph) {
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
    const auto* grad_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jgrad_handle);
    tensor_ptr->backward(*grad_ptr, keep_graph, create_graph);
}
