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
#include "ai_djl_paddlepaddle_jni_PaddleLibrary.h"
#include "djl_paddle_jni_utils.h"
#include<paddle_api.h>
#include<paddle_inference_api.h>
#include <numeric>

JNIEXPORT jlong JNICALL Java_ai_djl_paddlepaddle_jni_PaddleLibrary_createAnalysisConfig
        (JNIEnv *env, jobject jthis, jstring jmodel_dir, jstring jparam_dir, jint device_id) {
    auto config = new paddle::AnalysisConfig;
    if (jparam_dir == nullptr) {
        config->SetModel(utils::GetStringFromJString(env, jmodel_dir));
    } else {
        config->SetModel(utils::GetStringFromJString(env, jmodel_dir), utils::GetStringFromJString(env, jparam_dir));
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

JNIEXPORT void JNICALL Java_ai_djl_paddlepaddle_jni_PaddleLibrary_deleteAnalysisConfig
        (JNIEnv *env, jobject jthis, jlong jhandle) {
    const auto* config_ptr = reinterpret_cast<paddle::AnalysisConfig*>(jhandle);
    delete config_ptr;
}

JNIEXPORT jlong JNICALL Java_ai_djl_paddlepaddle_jni_PaddleLibrary_createPredictor
        (JNIEnv *env, jobject jthis, jlong jconfig) {
    const auto* config_ptr = reinterpret_cast<paddle::AnalysisConfig*>(jconfig);
    auto predictor = paddle::CreatePaddlePredictor(*config_ptr).release();
    return reinterpret_cast<uintptr_t>(predictor);
}

JNIEXPORT jlong JNICALL Java_ai_djl_paddlepaddle_jni_PaddleLibrary_clonePredictor
        (JNIEnv *env, jobject jthis, jlong jhandle) {
    const auto predictor = reinterpret_cast<paddle::PaddlePredictor*>(jhandle);
    auto cloned = predictor->Clone().release();
    return reinterpret_cast<uintptr_t>(cloned);
}

JNIEXPORT void JNICALL Java_ai_djl_paddlepaddle_jni_PaddleLibrary_deletePredictor
        (JNIEnv *env, jobject jthis, jlong jhandle) {
    const auto* predictor = reinterpret_cast<paddle::PaddlePredictor*>(jhandle);
    delete predictor;
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_paddlepaddle_jni_PaddleLibrary_getInputNames
        (JNIEnv *env, jobject jthis, jlong jhandle) {
    const auto predictor = reinterpret_cast<paddle::PaddlePredictor*>(jhandle);
    return utils::GetStringArrayFromVector(env, predictor->GetInputNames());
}

void tensor_to_ztensor(paddle::ZeroCopyTensor* z_tensor, paddle::PaddleTensor* tensor) {
    z_tensor->Reshape(tensor->shape);
    switch (tensor->dtype) {
        case paddle::PaddleDType::FLOAT32:
            z_tensor->copy_from_cpu(static_cast<float*>(tensor->data.data()));
            break;
        case paddle::PaddleDType::INT32:
            z_tensor->copy_from_cpu(static_cast<int32_t*>(tensor->data.data()));
            break;
        case paddle::PaddleDType::INT64:
            z_tensor->copy_from_cpu(static_cast<int64_t*>(tensor->data.data()));
            break;
        case paddle::PaddleDType::UINT8:
            z_tensor->copy_from_cpu(static_cast<uint8_t*>(tensor->data.data()));
            break;
    }
}

void ztensor_to_tensor(paddle::ZeroCopyTensor* z_tensor, paddle::PaddleTensor* tensor) {
    tensor->name = z_tensor->name();
    tensor->dtype = z_tensor->type();
    tensor->shape = z_tensor->shape();
    std::vector<int> output_shape = z_tensor->shape();
    int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    auto dtype = z_tensor->type();
    if (dtype == paddle::PaddleDType::FLOAT32) {
        int size = sizeof(float);
        tensor->data.Resize(out_num * size);
        z_tensor->copy_to_cpu(static_cast<float*>(tensor->data.data()));
    } else if (dtype == paddle::PaddleDType::INT32) {
        int size = sizeof(int32_t);
        tensor->data.Resize(out_num * size);
        z_tensor->copy_to_cpu(static_cast<int32_t *>(tensor->data.data()));
    } else if (dtype == paddle::PaddleDType::INT64) {
        int size = sizeof(int64_t);
        tensor->data.Resize(out_num * size);
        z_tensor->copy_to_cpu(static_cast<int64_t *>(tensor->data.data()));
    } else if (dtype == paddle::PaddleDType::UINT8) {
        int size = sizeof(uint8_t);
        tensor->data.Resize(out_num * size);
        z_tensor->copy_to_cpu(static_cast<uint8_t *>(tensor->data.data()));
    } else {
        // throw error
    }
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_paddlepaddle_jni_PaddleLibrary_runInference
        (JNIEnv *env, jobject jthis, jlong jhandle, jlongArray jtensor_ptrs) {
    const auto predictor = reinterpret_cast<paddle::PaddlePredictor*>(jhandle);
    jsize len = env->GetArrayLength(jtensor_ptrs);
    jlong* jptrs = env->GetLongArrayElements(jtensor_ptrs, JNI_FALSE);
    for (size_t i = 0; i < len; ++i) {
        auto tensor = reinterpret_cast<paddle::PaddleTensor*>(jptrs[i]);
        auto z_tensor = predictor->GetInputTensor(tensor->name);
        tensor_to_ztensor(z_tensor.get(), tensor);
    }
    predictor->ZeroCopyRun();
    auto output_names = predictor->GetOutputNames();
    int output_len = output_names.size();
    jlongArray jarray = env->NewLongArray(output_len);
    std::vector<jlong> ptr_vec;
    for (auto & output_name : output_names) {
        auto out_ztensor = predictor->GetOutputTensor(output_name);
        auto tensor_ptr = new paddle::PaddleTensor{};
        ztensor_to_tensor(out_ztensor.get(), tensor_ptr);
        ptr_vec.push_back(reinterpret_cast<uintptr_t>(tensor_ptr));
    }
    env->SetLongArrayRegion(jarray, 0, output_len, ptr_vec.data());
    return jarray;
}
