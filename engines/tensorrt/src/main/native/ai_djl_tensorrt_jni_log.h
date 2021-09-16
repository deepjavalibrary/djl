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
#ifndef DJL_TRT_JNI_LOG_H
#define DJL_TRT_JNI_LOG_H

#include <jni.h>

#include <cassert>
#include <iostream>
#include <string>

#include "NvInferRuntime.h"

using Severity = nvinfer1::ILogger::Severity;

class Logger : public nvinfer1::ILogger {
 public:
  Logger() : mSeverity(Severity::kINFO) {}
  void setSeverity(Severity severity) { mSeverity = severity; }
  void log(Severity severity, const char* msg) noexcept override;
  nvinfer1::ILogger& getTrtLogger() { return *this; }

 private:
  Severity mSeverity;
};

#endif  // DJL_TRT_JNI_LOG_H
