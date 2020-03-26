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

#ifndef DJL_TORCH_DJL_PYTORCH_JNI_ERROR_H
#define DJL_TORCH_DJL_PYTORCH_JNI_ERROR_H

/*
 * Macros to guard beginning and end section of all functions
 * every function starts with API_BEGIN()
 * and finishes with API_END()
 */
#define API_BEGIN() \
  try {             \
  __func__
#define API_END()                                                        \
  }                                                                      \
  catch (const c10::Error& e) {                                          \
    jclass jexception = env->FindClass("ai/djl/engine/EngineException"); \
    std::vector<std::string> stake_trace = e.msg_stack();                \
    for (std::string & msg : stake_trace) {                              \
      std::cerr << msg << std::endl;                                     \
    }                                                                    \
    env->ThrowNew(jexception, e.what_without_backtrace());               \
  }                                                                      \
  catch (const std::exception& e_) {                                     \
    jclass jexception = env->FindClass("ai/djl/engine/EngineException"); \
    env->ThrowNew(jexception, e_.what());                                \
  }                                                                      \
  return 0;

#endif  // DJL_TORCH_DJL_PYTORCH_JNI_ERROR_H
