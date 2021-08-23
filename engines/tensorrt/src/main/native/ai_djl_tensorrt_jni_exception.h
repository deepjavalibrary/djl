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
#ifndef DJL_TRT_JNI_EXCEPTION_H
#define DJL_TRT_JNI_EXCEPTION_H

#include "ai_djl_tensorrt_jni_cache.h"

/*
 * Macros to guard beginning and end section of all functions
 * every function starts with API_BEGIN()
 * and finishes with API_END()
 */
#define API_BEGIN() try {
#define API_END()                                     \
  }                                                   \
  catch (const std::exception& e_) {                  \
    env->ThrowNew(ENGINE_EXCEPTION_CLASS, e_.what()); \
  }

// TODO refactor all jni functions to c style function which mean
//  return value should be unified to function execution status code
#define API_END_RETURN() \
  API_END()              \
  return 0;

#endif  // DJL_TRT_JNI_EXCEPTION_H
