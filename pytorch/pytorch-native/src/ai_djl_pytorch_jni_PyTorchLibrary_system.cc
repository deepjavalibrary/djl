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

#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_error.h"
#include "djl_pytorch_jni_utils.h"

#if defined(__ANDROID__)
#ifndef USE_PTHREADPOOL
#define USE_PTHREADPOOL
#endif /* USE_PTHREADPOOL */
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#endif

// The file is the implementation for PyTorch system-wide operations

JNIEXPORT jint JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchGetNumInteropThreads(JNIEnv* env, jobject jthis) {
  API_BEGIN()
  return torch::get_num_interop_threads();
  API_END_RETURN()
}

JNIEXPORT jint JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchGetNumThreads(JNIEnv* env, jobject jthis) {
  API_BEGIN()
  return torch::get_num_threads();
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSetNumInteropThreads(
    JNIEnv* env, jobject jthis, jint jthreads) {
  API_BEGIN()
#if defined(__ANDROID__)
  Log log(env);
  log.info("Android didn't support this interop config, please use intra-op instead");
#else
  torch::set_num_interop_threads(jthreads);
#endif
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSetNumThreads(
    JNIEnv* env, jobject jthis, jint jthreads) {
  API_BEGIN()
#if defined(__ANDROID__)
  caffe2::pthreadpool()->set_thread_count(jthreads);
#else
  torch::set_num_threads(jthreads);
#endif
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchManualSeed(JNIEnv* env, jobject jthis, jlong jseed) {
  API_BEGIN()
  torch::manual_seed(jseed);
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchShowConfig(
    JNIEnv* env, jobject jthis, jobject jset) {
  API_BEGIN()
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
#if !defined(__ANDROID__)
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
#endif
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
  API_END()
}
