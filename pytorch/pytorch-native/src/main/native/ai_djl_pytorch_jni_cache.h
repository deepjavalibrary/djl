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

#ifndef DJL_TORCH_AI_DJL_PYTORCH_JNI_CACHE_H
#define DJL_TORCH_AI_DJL_PYTORCH_JNI_CACHE_H

#include <jni.h>

extern jclass NULL_PTR_EXCEPTION_CLASS;
extern jclass ILLEGAL_STATE_EXCEPTION_CLASS;
extern jclass ENGINE_EXCEPTION_CLASS;
extern jclass JNI_UTILS_CLASS;
extern jclass LOG4J_LOGGER_CLASS;
extern jfieldID LOGGER_FIELD;
extern jmethodID INFO_METHOD;
extern jmethodID DEBUG_METHOD;
extern jmethodID ERROR_METHOD;
// the highest version Android JNI version is 1.6
static jint JNI_VERSION = JNI_VERSION_1_6;

#endif  // DJL_TORCH_AI_DJL_PYTORCH_JNI_CACHE_H
