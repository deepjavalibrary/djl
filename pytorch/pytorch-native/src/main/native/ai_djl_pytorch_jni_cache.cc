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

#include "ai_djl_pytorch_jni_cache.h"

jclass NULL_PTR_EXCEPTION_CLASS;
jclass ILLEGAL_STATE_EXCEPTION_CLASS;
jclass ENGINE_EXCEPTION_CLASS;
jclass JNI_UTILS_CLASS;
jclass LOG4J_LOGGER_CLASS;
jfieldID LOGGER_FIELD;
jmethodID INFO_METHOD;
jmethodID DEBUG_METHOD;
jmethodID ERROR_METHOD;

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
  // Obtain the JNIEnv from the VM and confirm JNI_VERSION
  JNIEnv* env;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION) != JNI_OK) {
    return JNI_ERR;
  }
  jclass temp_class;
  temp_class = env->FindClass("java/lang/NullPointerException");
  NULL_PTR_EXCEPTION_CLASS = (jclass) env->NewGlobalRef(temp_class);

  temp_class = env->FindClass("java/lang/IllegalStateException");
  ILLEGAL_STATE_EXCEPTION_CLASS = (jclass) env->NewGlobalRef(temp_class);

  temp_class = env->FindClass("ai/djl/engine/EngineException");
  ENGINE_EXCEPTION_CLASS = (jclass) env->NewGlobalRef(temp_class);

  temp_class = env->FindClass("ai/djl/pytorch/jni/JniUtils");
  JNI_UTILS_CLASS = (jclass) env->NewGlobalRef(temp_class);
  LOGGER_FIELD = env->GetStaticFieldID(JNI_UTILS_CLASS, "logger", "Lorg/slf4j/Logger;");

  temp_class = env->FindClass("org/slf4j/Logger");
  LOG4J_LOGGER_CLASS = (jclass) env->NewGlobalRef(temp_class);

  INFO_METHOD = env->GetMethodID(LOG4J_LOGGER_CLASS, "info", "(Ljava/lang/String;)V");
  DEBUG_METHOD = env->GetMethodID(LOG4J_LOGGER_CLASS, "debug", "(Ljava/lang/String;)V");
  ERROR_METHOD = env->GetMethodID(LOG4J_LOGGER_CLASS, "error", "(Ljava/lang/String;)V");

  return JNI_VERSION;
}

void JNI_OnUnload(JavaVM* vm, void* reserved) {
  JNIEnv* env;
  vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION);

  env->DeleteGlobalRef(NULL_PTR_EXCEPTION_CLASS);
  env->DeleteGlobalRef(ILLEGAL_STATE_EXCEPTION_CLASS);
  env->DeleteGlobalRef(ENGINE_EXCEPTION_CLASS);
  env->DeleteGlobalRef(JNI_UTILS_CLASS);
  env->DeleteGlobalRef(LOG4J_LOGGER_CLASS);
}
