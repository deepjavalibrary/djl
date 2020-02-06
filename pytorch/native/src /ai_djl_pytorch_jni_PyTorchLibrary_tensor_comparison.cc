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

// The file is the implementation for PyTorch tensor comparision ops

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_contentEqual
  (JNIEnv* env, jobject jthis, jobject jhandle1, jobject jhandle2) {
  auto tensor_ptr1 = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle1);
  auto tensor_ptr2 = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle2);
  return tensor_ptr1->equal(*tensor_ptr2);
}