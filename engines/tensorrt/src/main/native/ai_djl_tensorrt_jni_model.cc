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

#include "ai_djl_tensorrt_jni_model.h"

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <strstream>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"

using djl_trt::TrtUniquePtr;

namespace djl_trt {

struct UffBufferShutter {
  ~UffBufferShutter() { nvuffparser::shutdownProtobufLibrary(); }
};

void TrtModel::buildModel() {
  auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTrtLogger()));
  if (!builder) {
    throw std::invalid_argument("Failed to call createInferBuilder.");
  }
  if (mParams.maxBatchSize > 0) {
    builder->setMaxBatchSize(mParams.maxBatchSize);
  }

  auto networkFlags = 0U;
  if (mParams.maxBatchSize == 0) {
    networkFlags |= 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  }
  auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(networkFlags));
  if (!network) {
    throw std::invalid_argument("Failed to call createNetworkV2.");
  }

  auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    throw std::invalid_argument("Failed to call builder->createBuilderConfig.");
  }

  TrtUniquePtr<nvuffparser::IUffParser> uffParser;  // must in outer scope
  if (mParams.modelType == 0) {
    // ONNX model
    auto onnxParser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTrtLogger()));
    if (!onnxParser) {
      throw std::invalid_argument("Failed create ONNX Parser.");
    }

    auto parsed = onnxParser->parseFromFile(mParams.modelPath.c_str(), static_cast<int>(Severity::kINFO));
    if (!parsed) {
      throw std::invalid_argument("Failed parse ONNX model file: ");
    }
  } else if (mParams.modelType == 1) {
    // UFF model
    using namespace nvuffparser;
    uffParser = TrtUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());
    if (!uffParser) {
      throw std::invalid_argument("Failed create UFF Parser.");
    }

    UffBufferShutter bufferShutter;
    for (const auto &s : mParams.uffInputs) {
      UffInputOrder inputOrder = mParams.uffNHWC ? UffInputOrder::kNHWC : UffInputOrder::kNCHW;
      if (!uffParser->registerInput(s.first.c_str(), s.second, inputOrder)) {
        uffParser.reset();
        throw std::invalid_argument("Failed to register input: " + s.first);
      }
    }

    for (const auto &s : mParams.uffOutputs) {
      if (!uffParser->registerOutput(s.c_str())) {
        uffParser.reset();
        throw std::invalid_argument("Failed to register output " + s);
      }
    }

    if (!uffParser->parse(mParams.modelPath.c_str(), *network)) {
      uffParser.reset();
      throw std::invalid_argument("Failed to parse uff file");
    }
  } else {
    throw std::invalid_argument("Unsupported model type: " + std::to_string(mParams.modelType));
  }

  config->setMaxWorkspaceSize(16_MiB);
  if (mParams.fp16) {
    config->setFlag(BuilderFlag::kFP16);
  } else if (mParams.int8) {
    config->setFlag(BuilderFlag::kINT8);
    setAllDynamicRanges(network.get(), 127.0f, 127.0f);
  }
  config->setFlag(BuilderFlag::kGPU_FALLBACK);
  enableDLA(builder.get(), config.get(), mParams.dlaCore);

  // CUDA stream used for profiling by the builder.
  auto profileStream = makeCudaStream();
  if (!profileStream) {
    throw std::invalid_argument("Failed to call makeCudaStream.");
  }
  config->setProfileStream(*profileStream);

  TrtUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  if (!plan) {
    throw std::invalid_argument("Failed to call buildSerializedNetwork.");
  }

  TrtUniquePtr<IRuntime> runtime{createInferRuntime(gLogger.getTrtLogger())};
  if (!runtime) {
    throw std::invalid_argument("Failed to call createInferRuntime.");
  }

  mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
  if (!mEngine) {
    throw std::invalid_argument("Failed to call deserializeCudaEngine.");
  }

  int32_t mNumInputs = network->getNbInputs();
  mInputTypes.reserve(mNumInputs);
  for (int i = 0; i < mNumInputs; ++i) {
    ITensor *tensor = network->getInput(i);
    mInputNames.emplace_back(tensor->getName());
    mInputTypes.emplace_back(tensor->getType());
  }

  int32_t mNumOutputs = network->getNbOutputs();
  for (int i = 0; i < mNumOutputs; ++i) {
    ITensor *tensor = network->getOutput(i);
    mOutputNames.emplace_back(tensor->getName());
    mOutputTypes.emplace_back(tensor->getType());
  }
}

TrtSession *TrtModel::createSession() {
  auto *session = new TrtSession(mEngine, mParams.maxBatchSize);

  CHECK(cudaSetDevice(mParams.device));

  session->init();
  return session;
}

void TrtSession::init() {
  mContext = mEngine->createExecutionContext();
  int bindings = mEngine->getNbBindings();
  mBufferSizes.reserve(bindings);
  mHostBuffers.reserve(bindings);
  mDeviceBuffers.reserve(bindings);

  for (int i = 0; i < bindings; i++) {
    auto dims = mContext->getBindingDimensions(i);
    size_t vol = 1;
    int vecDim = mEngine->getBindingVectorizedDim(i);
    if (-1 != vecDim)  // i.e., 0 != lgScalarsPerVector
    {
      int scalarsPerVec = mEngine->getBindingComponentsPerElement(i);
      dims.d[vecDim] = divUp(dims.d[vecDim], scalarsPerVec);
      vol *= scalarsPerVec;
    }
    vol *= volume(dims);
    vol *= getElementSize(mEngine->getBindingDataType(i));
    mBufferSizes.emplace_back(vol);

    void *ptr = safeCudaMalloc(vol);
    mDeviceBuffers.push_back(ptr);
  }
}

nvinfer1::Dims TrtSession::getShape(const char *name) {
  int index = mEngine->getBindingIndex(name);
  return mContext->getBindingDimensions(index);
}

void TrtSession::bind(const char *name, void *buffer, size_t size) {
  int index = mEngine->getBindingIndex(name);
  if (size != mBufferSizes[index]) {
    std::stringstream ss;
    ss << "Invalid binding size: " << size << ". binding[" << index << "], name: " << name
       << ", shape: " << mEngine->getBindingDimensions(index)
       << ", type: " << static_cast<int>(mEngine->getBindingDataType(index)) << ", size: " << mBufferSizes[index];
    throw std::invalid_argument(ss.str());
  }
  mHostBuffers[index] = buffer;
}

void TrtSession::copyInputs() {
  for (int i = 0; i < mEngine->getNbBindings(); i++) {
    if (mEngine->bindingIsInput(i)) {
      size_t byteSize = mBufferSizes[i];
      CHECK(cudaMemcpy(mDeviceBuffers[i], mHostBuffers[i], byteSize, cudaMemcpyHostToDevice));
    }
  }
}

void TrtSession::copyOutputs() {
  for (int i = 0; i < mEngine->getNbBindings(); i++) {
    if (!mEngine->bindingIsInput(i)) {
      size_t byteSize = mBufferSizes[i];
      CHECK(cudaMemcpy(mHostBuffers[i], mDeviceBuffers[i], byteSize, cudaMemcpyDeviceToHost));
    }
  }
}

void TrtSession::predict() {
  copyInputs();

  bool status;
  if (mBatchSize > 0) {
    status = mContext->execute(mBatchSize, mDeviceBuffers.data());
  } else {
    status = mContext->executeV2(mDeviceBuffers.data());
  }
  if (!status) {
    throw std::invalid_argument("Session execution failed, code: " + std::to_string(status));
  }

  copyOutputs();
}
}  // namespace djl_trt