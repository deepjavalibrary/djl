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
#ifndef DJL_TRT_JNI_COMMON_H
#define DJL_TRT_JNI_COMMON_H

#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "ai_djl_tensorrt_jni_log.h"

using namespace nvinfer1;
using namespace plugin;

#define CHECK(status)                                    \
  do {                                                   \
    auto ret = (status);                                 \
    if (ret != 0) {                                      \
      std::cerr << "Cuda failure: " << ret << std::endl; \
      abort();                                           \
    }                                                    \
  } while (0)

constexpr long double operator"" _GiB(long double val) { return val * (1 << 30); }
constexpr long double operator"" _MiB(long double val) { return val * (1 << 20); }
constexpr long double operator"" _KiB(long double val) { return val * (1 << 10); }

// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator"" _GiB(unsigned long long val) { return val * (1 << 30); }
constexpr long long int operator"" _MiB(unsigned long long val) { return val * (1 << 20); }
constexpr long long int operator"" _KiB(unsigned long long val) { return val * (1 << 10); }

namespace djl_trt {

extern Logger gLogger;

inline void* safeCudaMalloc(size_t memSize) {
  void* deviceMem;
  CHECK(cudaMalloc(&deviceMem, memSize));
  if (deviceMem == nullptr) {
    std::cerr << "Out of memory" << std::endl;
    exit(1);
  }
  return deviceMem;
}

struct InferDeleter {
  template <typename T>
  void operator()(T* obj) const {
    delete obj;
  }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, InferDeleter>;

static auto StreamDeleter = [](cudaStream_t* pStream) {
  if (pStream) {
    cudaStreamDestroy(*pStream);
    delete pStream;
  }
};

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream() {
  std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
  if (cudaStreamCreate(pStream.get()) != cudaSuccess) {
    pStream.reset(nullptr);
  }

  return pStream;
}

inline void setAllTensorScales(INetworkDefinition* network, float inScales = 2.0f, float outScales = 4.0f) {
  // Ensure that all layer inputs have a scale.
  for (int i = 0; i < network->getNbLayers(); i++) {
    auto layer = network->getLayer(i);
    for (int j = 0; j < layer->getNbInputs(); j++) {
      ITensor* input{layer->getInput(j)};
      // Optional inputs are nullptr here and are from RNN layers.
      if (input != nullptr && !input->dynamicRangeIsSet()) {
        input->setDynamicRange(-inScales, inScales);
      }
    }
  }

  // Ensure that all layer outputs have a scale.
  // Tensors that are also inputs to layers are ingored here
  // since the previous loop nest assigned scales to them.
  for (int i = 0; i < network->getNbLayers(); i++) {
    auto layer = network->getLayer(i);
    for (int j = 0; j < layer->getNbOutputs(); j++) {
      ITensor* output{layer->getOutput(j)};
      // Optional outputs are nullptr here and are from RNN layers.
      if (output != nullptr && !output->dynamicRangeIsSet()) {
        // Pooling must have the same input and output scales.
        if (layer->getType() == LayerType::kPOOLING) {
          output->setDynamicRange(-inScales, inScales);
        } else {
          output->setDynamicRange(-outScales, outScales);
        }
      }
    }
  }
}

inline void setAllDynamicRanges(INetworkDefinition* network, float inRange = 2.0f, float outRange = 4.0f) {
  return setAllTensorScales(network, inRange, outRange);
}

inline void enableDLA(IBuilder* builder, IBuilderConfig* config, int useDLACore, bool allowGPUFallback = true) {
  if (useDLACore >= 0) {
    if (builder->getNbDLACores() == 0) {
      throw std::invalid_argument(
          "Trying to use DLA core " + std::to_string(useDLACore) + " on a platform that doesn't have any DLA cores");
    }
    if (allowGPUFallback) {
      config->setFlag(BuilderFlag::kGPU_FALLBACK);
    }
    if (!config->getFlag(BuilderFlag::kINT8)) {
      // User has not requested INT8 Mode.
      // By default run in FP16 mode. FP32 mode is not permitted.
      config->setFlag(BuilderFlag::kFP16);
      throw std::invalid_argument("DataType is not set, default to fp16.");
    }
    config->setDefaultDeviceType(DeviceType::kDLA);
    config->setDLACore(useDLACore);
    config->setFlag(BuilderFlag::kSTRICT_TYPES);
  }
}

inline uint32_t getElementSize(nvinfer1::DataType t) noexcept {
  switch (t) {
    case nvinfer1::DataType::kINT32:
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
      return 1;
  }
  return 0;
}

inline int64_t volume(const nvinfer1::Dims& d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

template <typename A, typename B>
inline A divUp(A x, B n) {
  return (x + n - 1) / n;
}

inline std::vector<std::string> splitString(const std::string& str, char delimiter = ',') {
  std::vector<std::string> splitVect;
  std::stringstream ss(str);
  std::string substr;

  while (ss.good()) {
    getline(ss, substr, delimiter);
    splitVect.emplace_back(std::move(substr));
  }
  return splitVect;
}

inline std::vector<std::pair<std::string, nvinfer1::Dims>> parseUffInputs(const std::string& uffInputs) {
  std::vector<std::pair<std::string, nvinfer1::Dims>> ret;
  std::vector<std::string> inputs{splitString(uffInputs, ';')};
  for (const auto& i : inputs) {
    std::vector<std::string> values{splitString(i)};
    if (values.size() == 4) {
      nvinfer1::Dims3 dims{std::stoi(values[1]), std::stoi(values[2]), std::stoi(values[3])};
      ret.emplace_back(values[0], dims);
    } else {
      throw std::invalid_argument("Invalid uffInput " + i);
    }
  }
  return ret;
}

}  // namespace djl_trt

inline std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& dims) {
  os << "(";
  for (int i = 0; i < dims.nbDims; ++i) {
    os << (i ? ", " : "") << dims.d[i];
  }
  return os << ")";
}

#endif  // DJL_TRT_JNI_COMMON_H
