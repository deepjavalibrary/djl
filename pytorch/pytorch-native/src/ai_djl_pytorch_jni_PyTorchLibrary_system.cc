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
#include <torch/torch.h>

#include "../build/include/ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_utils.h"

// The file is the implementation for PyTorch system-wide operations

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchManualSeed(JNIEnv* env, jobject jthis, jlong jseed) {
  torch::manual_seed(jseed);
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchShowConfig(
    JNIEnv* env, jobject jthis, jobject jset) {
  jclass jexception = env->FindClass("java/lang/NullPointerException");
  jclass set_class = env->GetObjectClass(jset);
  if (set_class == nullptr) {
    env->ThrowNew(jexception, "Java Set class is not found");
  }
  jmethodID add_method_id = env->GetMethodID(set_class, "add", "(Ljava/lang/Object;)Z");
  if (add_method_id == nullptr) {
    env->ThrowNew(jexception, "The add method in Set is not found");
  }
  std::string feature;
  jstring jfeature;
  if (torch::cuda::is_available()) {
    feature = "CUDA";
    jfeature = env->NewStringUTF(feature.c_str());
    env->CallBooleanMethod(jset, add_method_id, jfeature);
    env->DeleteLocalRef(jfeature);
  }
  if (torch::cuda::cudnn_is_available()) {
    feature = "CUDNN";
    jfeature = env->NewStringUTF(feature.c_str());
    env->CallBooleanMethod(jset, add_method_id, jfeature);
    env->DeleteLocalRef(jfeature);
  }
  if (torch::hasMKL()) {
    feature = "MKL";
    jfeature = env->NewStringUTF(feature.c_str());
    env->CallBooleanMethod(jset, add_method_id, jfeature);
    env->DeleteLocalRef(jfeature);
  }
  if (torch::hasMKLDNN()) {
    feature = "MKLDNN";
    jfeature = env->NewStringUTF(feature.c_str());
    env->CallBooleanMethod(jset, add_method_id, jfeature);
    env->DeleteLocalRef(jfeature);
  }
  if (torch::hasOpenMP()) {
    feature = "OPENMP";
    jfeature = env->NewStringUTF(feature.c_str());
    env->CallBooleanMethod(jset, add_method_id, jfeature);
    env->DeleteLocalRef(jfeature);
  }
}
