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
#include "djl_pytorch_jni_log.h"

#include "ai_djl_pytorch_jni_cache.h"

static inline jobject get_log_object(JNIEnv* env) { return env->GetStaticObjectField(JNI_UTILS_CLASS, LOGGER_FIELD); }

Log::Log(JNIEnv* env) : env(env), logger(get_log_object(env)) {}

void Log::info(const std::string& message) {
  env->CallVoidMethod(logger, INFO_METHOD, env->NewStringUTF(message.c_str()));
}

void Log::debug(const std::string& message) {
  env->CallVoidMethod(logger, DEBUG_METHOD, env->NewStringUTF(message.c_str()));
}

void Log::error(const std::string& message) {
  env->CallVoidMethod(logger, ERROR_METHOD, env->NewStringUTF(message.c_str()));
}
