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

static inline jobject get_log_object(JNIEnv* env) {
  auto cls = env->FindClass(JNIUTILS_CLASS);
  assert(cls);
  auto log_field = env->GetStaticFieldID(cls, "logger", "Lorg/slf4j/Logger;");
  assert(log_field);
  auto log_obj = env->GetStaticObjectField(cls, log_field);
  assert(log_obj);
  return log_obj;
}

static inline jmethodID get_info_method(JNIEnv* env, jobject log) {
  auto method_id = env->GetMethodID(env->GetObjectClass(log), "error", "(Ljava/lang/String;)V");
  assert(method_id);
  return method_id;
}

Log::Log(JNIEnv* env) : env(env), logger(get_log_object(env)), error_method(get_info_method(env, logger)) {
}

void Log::error(const std::string& message) {
  env->CallVoidMethod(logger, error_method, env->NewStringUTF(message.c_str()));
}
