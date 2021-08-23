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
#ifndef DJL_TRT_JNI_CACHE_H
#define DJL_TRT_JNI_CACHE_H

#include <jni.h>

extern jclass ENGINE_EXCEPTION_CLASS;
// the highest version Android JNI version is 1.6
static jint JNI_VERSION = JNI_VERSION_1_6;

#endif  // DJL_TRT_JNI_CACHE_H
