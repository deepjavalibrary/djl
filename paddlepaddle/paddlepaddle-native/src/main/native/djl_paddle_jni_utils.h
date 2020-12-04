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

#ifndef DJL_PADDLE_DJL_PADDLE_JNI_UTILS_H
#define DJL_PADDLE_DJL_PADDLE_JNI_UTILS_H

#include <jni.h>
#include <iostream>
#include <vector>


namespace utils {

    static constexpr const jint RELEASE_MODE = JNI_ABORT;

    inline std::vector<int> GetVecFromJIntArray(JNIEnv* env, jintArray jarray) {
        jint* jarr = env->GetIntArrayElements(jarray, JNI_FALSE);
        jsize length = env->GetArrayLength(jarray);
        std::vector<int> vec(jarr, jarr + length);
        env->ReleaseIntArrayElements(jarray, jarr, RELEASE_MODE);
        return std::move(vec);
    }

    inline std::string GetStringFromJString(JNIEnv* env, jstring jstr) {
        if (jstr == nullptr) {
            return std::string();
        }
        const char* c_str = env->GetStringUTFChars(jstr, JNI_FALSE);
        std::string str = std::string(c_str);
        env->ReleaseStringUTFChars(jstr, c_str);
        return str;
    }
}

#endif //DJL_PADDLE_DJL_PADDLE_JNI_UTILS_H
