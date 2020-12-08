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

JNIEXPORT jlong JNICALL Java_ai_djl_paddlepaddle_jni_PaddleLibrary_paddleCreateTensor
        (JNIEnv *env, jobject jthis, jobject jbuffer, jlong jlength, jintArray jshape, jint jdtype) {
    auto paddleBuf = paddle::PaddleBuf(env->GetDirectBufferAddress(jbuffer), jlength);
    auto tensor_ptr = new paddle::PaddleTensor{};
    tensor_ptr->data = paddleBuf;
    tensor_ptr->dtype = static_cast<paddle::PaddleDType>(jdtype);
    tensor_ptr->shape = utils::GetVecFromJIntArray(env, jshape);
    return reinterpret_cast<uintptr_t>(tensor_ptr);
}

JNIEXPORT void JNICALL Java_ai_djl_paddlepaddle_jni_PaddleLibrary_deleteTensor
        (JNIEnv *env, jobject jthis, jlong jhandle) {
    const auto* tensor_ptr = reinterpret_cast<paddle::PaddleTensor*>(jhandle);
    delete tensor_ptr;
}
