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

JNIEXPORT jlong JNICALL Java_ai_djl_paddlepaddle_jni_PaddleLibrary_createAnalysisConfig
        (JNIEnv *env, jobject jthis, jstring jmodel_dir, jstring jparam_dir, jint device_id) {
    auto config = new paddle::AnalysisConfig;
    if (jparam_dir == NULL) {
        config->SetModel(utils::GetStringFromJString(env, jmodel_dir));
    } else {
        config->SetModel(utils::GetStringFromJString(env, jmodel_dir), utils::GetStringFromJString(env, jparam_dir));
    }
    if (device_id == -1) {
        config->DisableGpu();
    } else {
        config->EnableUseGpu(100, device_id);
    }
    return reinterpret_cast<uintptr_t>(config);
}
