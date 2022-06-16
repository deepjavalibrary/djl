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
#include "ai_djl_pytorch_jni_cache.h"
#include "djl_pytorch_jni_exception.h"
#include "djl_pytorch_utils.h"

// The file is the implementation for PyTorch tensor core functionality operation

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSizes(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  jlongArray size = env->NewLongArray(tensor_ptr->dim());
  env->SetLongArrayRegion(size, 0, tensor_ptr->dim(), reinterpret_cast<const jlong*>(tensor_ptr->sizes().data()));
  return size;
  API_END_RETURN()
}

JNIEXPORT jint JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDType(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  return utils::GetDTypeFromScalarType(tensor_ptr->scalar_type());
  API_END_RETURN()
}

JNIEXPORT jintArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDevice(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  jintArray result = env->NewIntArray(2);
  if (result == nullptr) {
    env->ThrowNew(NULL_PTR_EXCEPTION_CLASS, "Unable to create int array");
    return nullptr;
  }
  jint temp_device[] = {static_cast<int>(tensor_ptr->device().type()), tensor_ptr->device().index()};
  env->SetIntArrayRegion(result, 0, 2, temp_device);
  return result;
  API_END_RETURN()
}

JNIEXPORT jint JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLayout(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  auto layout = tensor_ptr->layout();
  switch (layout) {
    case torch::kStrided:
      return 0;
    case torch::kSparse:
      return 1;
    case torch::kMkldnn:
      return 2;
    default:
      env->ThrowNew(ILLEGAL_STATE_EXCEPTION_CLASS,
          "Internal PyTorch error, layout should only have kStrided, kSparse or kMkldnn");
  }
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchTo(
    JNIEnv* env, jobject jthis, jlong jhandle, jint jdtype, jintArray jdevice) {
  API_BEGIN()
  torch::NoGradGuard NoGradGuard;
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto device = utils::GetDeviceFromJDevice(env, jdevice);
  auto result = tensor_ptr->to(device, utils::GetScalarTypeFromDType(jdtype), false);
  const auto* result_ptr = new torch::Tensor(result);
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchGetItem__JJ(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jindex) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->index({static_cast<int64_t>(jindex)}));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchGetItem__J_3J(
    JNIEnv* env, jobject jthis, jlong jhandle, jlongArray jindices) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  std::vector<int64_t> vec = djl::utils::jni::GetVecFromJLongArray(env, jindices);
  std::vector<torch::indexing::TensorIndex> indices;
  indices.reserve(vec.size());
  std::transform(vec.begin(), vec.end(), std::back_inserter(indices),
      [](int64_t index) { return torch::indexing::TensorIndex{index}; });
  const auto* result_ptr = new torch::Tensor(tensor_ptr->index(indices));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_tensorClone(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  torch::NoGradGuard NoGradGuard;
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->clone());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchIndex(JNIEnv* env, jobject jthis, jlong jhandle,
    jlongArray jmin_indices, jlongArray jmax_indices, jlongArray jstep_indices) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  auto indices = utils::CreateTensorIndex(env, jmin_indices, jmax_indices, jstep_indices);
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchIndexAdv(JNIEnv* env, jobject jthis, jlong jhandle, jobject index) {
  API_BEGIN()
  std::cout << "Reached here!" << std::endl;
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  using namespace std;

  jclass java_util_ArrayList = static_cast<jclass>(env->FindClass("java/util/ArrayList"));
  cout << "temp check:" << java_util_ArrayList << endl;
  jmethodID java_util_ArrayList_get = env->GetMethodID(java_util_ArrayList, "get", "(I)Ljava/lang/Object;");
  jmethodID java_util_ArrayList_size = env->GetMethodID(java_util_ArrayList, "size", "()I");
  cout << "Give me correct size() methodID:" << java_util_ArrayList_get << endl;
  cout << "jclass: " << java_util_ArrayList << endl;

  jint size = 0;
  cout << "size:" << size << endl;
//    size = env->CallIntMethod(java_util_ArrayList, java_util_ArrayList_size);
  jobject index_element = env->CallObjectMethod(java_util_ArrayList, java_util_ArrayList_get, 0);
//    cout << "size" << size << endl;
  cout << index_element << endl;

  cout << "Dev ends here!" << endl;
  return reinterpret_cast<uintptr_t>(tensor_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchIndexInit(JNIEnv* env, jobject jthis, jint jsize) {
  API_BEGIN()
  //const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  using namespace std;
  using namespace torch::indexing;

  std::vector<at::indexing::TensorIndex> *index_ptr = new std::vector<at::indexing::TensorIndex>;
  index_ptr->reserve(jsize);

  index_ptr->emplace_back(Slice(0, 2));
  cout << (*index_ptr)[0] << endl;
  //cout << tensor_ptr->index((*index_ptr)[0]) << endl;

  return reinterpret_cast<uintptr_t>(index_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchIndexReturn(JNIEnv* env, jobject jthis,
    jlong jhandle, jlong jtorch_index_handle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  using namespace std;
  using namespace torch::indexing;

  std::cout << "IndexReturn:" << jtorch_index_handle << std::endl;
  auto* index_ptr = reinterpret_cast<std::vector<at::indexing::TensorIndex> *>(jtorch_index_handle);
  std::cout << "*index_ptr[0] " << sizeof(index_ptr->at(0)) << " " << index_ptr->at(0) << std::endl;
  std::cout << "*index_ptr " << sizeof(*index_ptr) << " " << *index_ptr << std::endl;
  cout << "output: " << tensor_ptr->index(index_ptr->at(0)) << endl;
  torch::Tensor* ret_ptr = new torch::Tensor(tensor_ptr->index(index_ptr->at(0)));
  return reinterpret_cast<uintptr_t>(ret_ptr);
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchIndexAppendNoneEllipsis(JNIEnv* env, jobject jthis,
    jlong jtorch_index_handle, jboolean jis_ellipsis) {
  API_BEGIN()
  auto* index_ptr = reinterpret_cast<std::vector<at::indexing::TensorIndex> *>(jtorch_index_handle);
  if (jis_ellipsis) {
      index_ptr->emplace_back(torch::indexing::Ellipsis);
  } else {
      index_ptr->emplace_back(torch::indexing::None);
  }
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchIndexAppendSlice(JNIEnv* env, jobject jthis,
    jlong jtorch_index_handle, jlong jmin, jlong jmax, jlong jstep, jint jnull_slice_bin) {
  API_BEGIN()
  auto* index_ptr = reinterpret_cast<std::vector<at::indexing::TensorIndex> *>(jtorch_index_handle);
  if (jnull_slice_bin == 0) {
      index_ptr->emplace_back(torch::indexing::Slice(jmin, jmax, jstep));
  } else if (jnull_slice_bin == 1) {
      index_ptr->emplace_back(torch::indexing::Slice(jmin, torch::indexing::None, jstep));
  } else if (jnull_slice_bin == 2) {
      index_ptr->emplace_back(torch::indexing::Slice(torch::indexing::None, jmax, jstep));
  } else if (jnull_slice_bin == 3) {
      index_ptr->emplace_back(torch::indexing::Slice(torch::indexing::None, torch::indexing::None, jstep));
  }
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchIndexAppendFixed(JNIEnv* env, jobject jthis,
    jlong jtorch_index_handle, jlong jidx) {
  API_BEGIN()
  auto* index_ptr = reinterpret_cast<std::vector<at::indexing::TensorIndex> *>(jtorch_index_handle);
  index_ptr->emplace_back((long) jidx);
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchIndexAppendArray(JNIEnv* env, jobject jthis,
    jlong jtorch_index_handle, jlong jarray) {
API_BEGIN()
  auto* index_ptr = reinterpret_cast<std::vector<at::indexing::TensorIndex> *>(jtorch_index_handle);
  auto* array_ptr = reinterpret_cast<torch::Tensor*>(jarray);
  index_ptr->emplace_back(*array_ptr);
API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchIndexPut(JNIEnv* env, jobject jthis, jlong jhandle,
    jlong jvalue_handle, jlongArray jmin_indices, jlongArray jmax_indices, jlongArray jstep_indices) {
  API_BEGIN()
  auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* value_ptr = reinterpret_cast<torch::Tensor*>(jvalue_handle);
  auto indices = utils::CreateTensorIndex(env, jmin_indices, jmax_indices, jstep_indices);
  tensor_ptr->index_put_(indices, *value_ptr);
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSet(
    JNIEnv* env, jobject jthis, jlong jhandle, jobject jbuffer) {
  API_BEGIN()
  torch::AutoNonVariableTypeMode guard;
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  torch::ArrayRef<int64_t> sizes = tensor_ptr->sizes();
  // the jbuffer is on the CPU. Although storage can specify GPU device,
  // it still doesn't move the data to GPU for us. So we
  // handle the gpu case explicitly here
  if (tensor_ptr->device().is_cuda()) {
    auto options = torch::TensorOptions().dtype(tensor_ptr->dtype()).requires_grad(false);
    torch::Tensor temp = torch::from_blob(env->GetDirectBufferAddress(jbuffer), sizes, options);
    temp = temp.to(tensor_ptr->device());
    tensor_ptr->set_(temp);
    return;
  }
  std::vector<int64_t> strides = at::detail::defaultStrides(sizes);
  torch::Storage storage(torch::Storage::use_byte_size_t(),
      at::detail::computeStorageNbytes(sizes, strides, tensor_ptr->dtype().itemsize()),
      torch::DataPtr(env->GetDirectBufferAddress(jbuffer), tensor_ptr->device()),
      /*allocator=*/nullptr,
      /*resizable=*/false);
  tensor_ptr->set_(storage, 0, sizes, strides);
  API_END()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSlice(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jdim, jlong jstart, jlong jend, jlong jstep) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->slice(jdim, jstart, jend, jstep));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchGather(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jindex_handle, jlong dim, jboolean sparse_grad) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* index_ptr = reinterpret_cast<torch::Tensor*>(jindex_handle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->gather(dim, *index_ptr, sparse_grad));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchTake(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jindex_handle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* index_ptr = reinterpret_cast<torch::Tensor*>(jindex_handle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->take(*index_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMaskedSelect(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jmasked_handle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* index_ptr = reinterpret_cast<torch::Tensor*>(jmasked_handle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->masked_select(*index_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMaskedPut(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jvalue_handle, jlong jmasked_handle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* index_ptr = reinterpret_cast<torch::Tensor*>(jmasked_handle);
  const auto* value_ptr = reinterpret_cast<torch::Tensor*>(jvalue_handle);
  tensor_ptr->masked_fill_(*index_ptr, *value_ptr);
  API_END()
}

JNIEXPORT jbyteArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDataPtr(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  // sparse and mkldnn are required to be converted to dense to access data ptr
  auto tensor = (tensor_ptr->is_sparse() || tensor_ptr->is_mkldnn()) ? tensor_ptr->to_dense() : *tensor_ptr;
  tensor = (tensor.is_contiguous()) ? tensor : tensor.contiguous();
  jbyteArray result = env->NewByteArray(tensor.nbytes());
  env->SetByteArrayRegion(result, 0, tensor.nbytes(), static_cast<const jbyte*>(tensor.data_ptr()));
  return result;
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteTensor(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  delete tensor_ptr;
  API_END()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchToSparse(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  torch::NoGradGuard NoGradGuard;
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->to_sparse());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchToDense(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  torch::NoGradGuard NoGradGuard;
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->to_dense());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchRequiresGrad(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  return tensor_ptr->requires_grad();
  API_END_RETURN()
}

JNIEXPORT jstring JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchGradFnName(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  return env->NewStringUTF(tensor_ptr->grad_fn()->name().c_str());
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAttachGrad(
    JNIEnv* env, jobject jthis, jlong jhandle, jboolean jrequires_grad) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  tensor_ptr->requires_grad_(jrequires_grad == JNI_TRUE);
  API_END()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchGrad(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->grad());
  if (!tensor_ptr->grad().defined()) {
    return djl::utils::jni::NULL_PTR;
  }
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDetachGrad(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->detach());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchBackward(
    JNIEnv* env, jobject jthis, jlong jhandle, jlong jgrad_handle, jboolean keep_graph, jboolean create_graph) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* grad_ptr = reinterpret_cast<torch::Tensor*>(jgrad_handle);
  tensor_ptr->backward(*grad_ptr, keep_graph, create_graph);
  API_END()
}
