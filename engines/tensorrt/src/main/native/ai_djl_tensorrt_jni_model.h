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
#ifndef DJL_TRT_JNI_MODEL_H
#define DJL_TRT_JNI_MODEL_H

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>

#include "NvInfer.h"
#include "ai_djl_tensorrt_jni_common.h"
#include "ai_djl_tensorrt_jni_log.h"

namespace djl_trt {

struct ModelParams {
  int modelType;
  std::string modelPath;
  int device{0};
  bool int8{false};
  bool fp16{false};
  int32_t maxBatchSize{0};
  int32_t dlaCore{-1};
  std::vector<std::pair<std::string, nvinfer1::Dims>> uffInputs;
  std::vector<std::string> uffOutputs;
  bool uffNHWC{false};
};

class TrtSession {
 public:
  explicit TrtSession(std::shared_ptr<nvinfer1::ICudaEngine> engine, int deviceId, int batchSize)
      : mEngine(std::move(engine)), mDeviceId(deviceId), mBatchSize(batchSize), mContext(nullptr) {}

  void init();
  nvinfer1::Dims getShape(const char* name);
  void bind(const char* name, void* buffer, size_t size);
  void predict();

  ~TrtSession() {
    for (auto& deviceBuffer : mDeviceBuffers) {
      cudaFree(deviceBuffer);
    }
  }

 private:
  std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
  int mDeviceId;
  nvinfer1::IExecutionContext* mContext;
  int32_t mBatchSize;
  std::vector<size_t> mBufferSizes;
  std::vector<void*> mHostBuffers;
  std::vector<void*> mDeviceBuffers;

  void copyInputs();
  void copyOutputs();
};

class TrtModel {
 public:
  explicit TrtModel(ModelParams params) : mParams(std::move(params)), mEngine(nullptr) {}
  void buildModel();
  TrtSession* createSession();
  std::vector<std::string> getInputNames() const { return mInputNames; }
  std::vector<std::string> getOutputNames() const { return mOutputNames; }
  std::vector<nvinfer1::DataType> getInputTypes() { return mInputTypes; }
  std::vector<nvinfer1::DataType> getOutputTypes() { return mOutputTypes; }

 private:
  ModelParams mParams;
  std::shared_ptr<nvinfer1::ICudaEngine> mEngine;

  std::vector<nvinfer1::DataType> mInputTypes;
  std::vector<nvinfer1::DataType> mOutputTypes;
  std::vector<std::string> mInputNames;
  std::vector<std::string> mOutputNames;

  void loadSerializedEngine();
};
}  // namespace djl_trt

#endif  // DJL_TRT_JNI_MODEL_H
