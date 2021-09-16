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
#include "ai_djl_tensorrt_jni_log.h"

static std::string severityPrefix(Severity severity) {
  switch (severity) {
    case Severity::kINTERNAL_ERROR:
      return "FATAL: ";
    case Severity::kERROR:
      return "ERROR: ";
    case Severity::kWARNING:
      return "WARN:  ";
    case Severity::kINFO:
      return "INFO:  ";
    case Severity::kVERBOSE:
      return "TRACE: ";
    default:
      assert(0);
      return "";
  }
}

void Logger::log(Severity severity, const char* msg) noexcept {
  if (severity <= mSeverity) {
    std::ostream& output = severity >= Severity::kINFO ? std::cout : std::cerr;
    output << "[TRT] " << severityPrefix(severity) << msg << std::endl;
  }
}

namespace djl_trt {
Logger gLogger{};
}
