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

#include <dlfcn.h>
#include "../build/include/ai_djl_pytorch_jni_NativeLoader.h"

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_NativeLoader_loadGlobal
  (JNIEnv *env, jobject jthis, jstring jfilePath) {
  const char *nativePath = env->GetStringUTFChars(jfilePath, JNI_FALSE);
  dlopen(nativePath, RTLD_LAZY | RTLD_GLOBAL);
}
