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
#include <djl/utils.h>
#include <paddle_inference_api.h>

#include "ai_djl_paddlepaddle_jni_PaddleLibrary.h"
#include "djl_paddle_utils.h"

JNIEXPORT jlong JNICALL Java_ai_djl_paddlepaddle_jni_PaddleLibrary_createAnalysisConfig(
    JNIEnv* env, jobject jthis, jstring jmodel_dir, jstring jparam_dir, jint device_id) {
  paddle::AnalysisConfig* config;
  if (jparam_dir == nullptr) {
    config = new paddle::AnalysisConfig(djl::utils::jni::GetStringFromJString(env, jmodel_dir));
  } else {
    config = new paddle::AnalysisConfig(
        djl::utils::jni::GetStringFromJString(env, jmodel_dir), djl::utils::jni::GetStringFromJString(env, jparam_dir));
  }
  if (device_id == -1) {
    config->DisableGpu();
  } else {
    config->EnableUseGpu(100, device_id);
  }
  config->SwitchUseFeedFetchOps(false);
  // optional: config->SwitchIrOptim(false); optimize performance
  // config->EnableMKLDNN(); not supporting multi-thread
  // config->SetCpuMathLibraryNumThreads(); set to 1 for multi-thread
  return reinterpret_cast<uintptr_t>(config);
}

JNIEXPORT void JNICALL Java_ai_djl_paddlepaddle_jni_PaddleLibrary_loadExtraDir(
    JNIEnv* env, jobject jthis, jobjectArray jargs) {
  // like --mkl-lib="libpath"
  auto vec_arg = djl::utils::jni::GetVecFromJStringArray(env, jargs);
  std::vector<char*> argv;
  int size = vec_arg.size();
  argv.reserve(vec_arg.size());
  for (auto& arg : vec_arg) {
    argv.push_back(const_cast<char*>(arg.data()));
  }
  char** array = argv.data();
  std::cout << "Pending Paddle fix to proceed with the option" << std::endl;
}

JNIEXPORT void JNICALL Java_ai_djl_paddlepaddle_jni_PaddleLibrary_analysisConfigEnableMKLDNN(
        JNIEnv* env, jobject jthis, jlong jhandle) {
    auto* config_ptr = reinterpret_cast<paddle::AnalysisConfig*>(jhandle);
    config_ptr->EnableMKLDNN();
}

JNIEXPORT void JNICALL Java_ai_djl_paddlepaddle_jni_PaddleLibrary_deleteAnalysisConfig(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  const auto* config_ptr = reinterpret_cast<paddle::AnalysisConfig*>(jhandle);
  delete config_ptr;
}

JNIEXPORT jlong JNICALL Java_ai_djl_paddlepaddle_jni_PaddleLibrary_createPredictor(
    JNIEnv* env, jobject jthis, jlong jconfig) {
  const auto* config_ptr = reinterpret_cast<paddle::AnalysisConfig*>(jconfig);
  auto predictor = paddle::CreatePaddlePredictor(*config_ptr).release();
  return reinterpret_cast<uintptr_t>(predictor);
}

JNIEXPORT jlong JNICALL Java_ai_djl_paddlepaddle_jni_PaddleLibrary_clonePredictor(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  const auto predictor = reinterpret_cast<paddle::PaddlePredictor*>(jhandle);
  auto cloned = predictor->Clone().release();
  return reinterpret_cast<uintptr_t>(cloned);
}

JNIEXPORT void JNICALL Java_ai_djl_paddlepaddle_jni_PaddleLibrary_deletePredictor(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  auto* predictor = reinterpret_cast<paddle::PaddlePredictor*>(jhandle);
  delete predictor;
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_paddlepaddle_jni_PaddleLibrary_getInputNames(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  auto* predictor = reinterpret_cast<paddle::PaddlePredictor*>(jhandle);
  return djl::utils::jni::GetStringArrayFromVec(env, predictor->GetInputNames());
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_paddlepaddle_jni_PaddleLibrary_runInference(
    JNIEnv* env, jobject jthis, jlong jhandle, jlongArray jtensor_ptrs) {
  auto* predictor = reinterpret_cast<paddle::PaddlePredictor*>(jhandle);
  jsize len = env->GetArrayLength(jtensor_ptrs);
  jlong* jptrs = env->GetLongArrayElements(jtensor_ptrs, JNI_FALSE);
  for (size_t i = 0; i < len; ++i) {
    auto tensor = reinterpret_cast<paddle::PaddleTensor*>(jptrs[i]);
    auto z_tensor = predictor->GetInputTensor(tensor->name);
    utils::GetZTensorFromTensor(z_tensor.get(), tensor);
  }
  predictor->ZeroCopyRun();
  auto output_names = predictor->GetOutputNames();
  int output_len = output_names.size();
  jlongArray jarray = env->NewLongArray(output_len);
  std::vector<jlong> ptr_vec;
  for (const auto& output_name : output_names) {
    auto out_ztensor = predictor->GetOutputTensor(output_name);
    auto tensor_ptr = new paddle::PaddleTensor{};
    utils::GetTensorFromZTensor(out_ztensor.get(), tensor_ptr);
    ptr_vec.emplace_back(reinterpret_cast<uintptr_t>(tensor_ptr));
  }
  env->SetLongArrayRegion(jarray, 0, output_len, ptr_vec.data());
  return jarray;
}
