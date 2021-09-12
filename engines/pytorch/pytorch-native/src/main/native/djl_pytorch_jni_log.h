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
#ifndef DJL_TORCH_DJL_PYTORCH_JNI_LOG_H
#define DJL_TORCH_DJL_PYTORCH_JNI_LOG_H

#include <jni.h>

#include <cassert>
#include <iostream>
#include <string>

/**
 * The Log class is for logging the C++ message with slf4j
 * which helps the Java developer to debug.
 */
class Log {
 public:
  explicit Log(JNIEnv* env);
  void info(const std::string& message);
  void debug(const std::string& message);
  void error(const std::string& message);

 private:
  JNIEnv* env;
  const jobject logger;
};

#endif  // DJL_TORCH_DJL_PYTORCH_JNI_LOG_H
