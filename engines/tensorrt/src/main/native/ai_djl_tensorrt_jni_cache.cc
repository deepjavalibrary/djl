/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

#include "ai_djl_tensorrt_jni_cache.h"

jclass ENGINE_EXCEPTION_CLASS;

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
  // Obtain the JNIEnv from the VM and confirm JNI_VERSION
  JNIEnv* env;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION) != JNI_OK) {
    return JNI_ERR;
  }
  jclass temp_class = env->FindClass("ai/djl/engine/EngineException");
  ENGINE_EXCEPTION_CLASS = (jclass) env->NewGlobalRef(temp_class);

  return JNI_VERSION;
}

void JNI_OnUnload(JavaVM* vm, void* reserved) {
  JNIEnv* env;
  vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION);

  env->DeleteGlobalRef(ENGINE_EXCEPTION_CLASS);
}
