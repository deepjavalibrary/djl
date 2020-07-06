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
#include <torch/torch.h>
#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_error.h"
#include "djl_pytorch_jni_utils.h"

// The file is the implementation for PyTorch neural network functional ops

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSoftmax(
    JNIEnv* env, jobject jthis, jobject jhandle, jlong jdim, jint jdtype) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->softmax(jdim, utils::GetScalarTypeFromDType(jdtype)));
  return utils::CreatePointer<const torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLogSoftmax(
  JNIEnv* env, jobject jthis, jobject jhandle, jlong jdim, jint jdtype) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
    const auto* result_ptr = new torch::Tensor(tensor_ptr->log_softmax(jdim, utils::GetScalarTypeFromDType(jdtype)));
    return utils::CreatePointer<const torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchUpsampleBilinear2d(
    JNIEnv* env, jobject jthis, jobject jhandle, jlongArray jsize, jboolean jalign_corners) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  const auto size_vec = utils::GetVecFromJLongArray(env, jsize);
  const auto* result_ptr =
      new torch::Tensor(torch::upsample_bilinear2d(*tensor_ptr, size_vec, jalign_corners == JNI_TRUE));
  return utils::CreatePointer<const torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNLinear
  (JNIEnv* env, jobject jthis, jobject jinput, jobject jweight, jobject jbias) {
  API_BEGIN();
    auto* input_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jinput);
    auto* weight_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jweight);
    torch::Tensor bias = {};
    if (!env->IsSameObject(jbias, nullptr)) {
      bias = *utils::GetPointerFromJHandle<const torch::Tensor>(env, jbias);
    }
    const auto* result_ptr = new torch::Tensor(torch::nn::functional::linear(*input_ptr, *weight_ptr, bias));
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNConvNd
  (JNIEnv* env, jobject jthis, jobject jinput, jobject jweight, jobject jbias, jlongArray jstride, jlongArray jpadding, jlongArray jdilation, jint jgroups) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jinput);
    const auto* weigtht_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jweight);
    torch::Tensor bias = {};
    if (!env->IsSameObject(jbias, nullptr)) {
      bias = *utils::GetPointerFromJHandle<const torch::Tensor>(env, jbias);
    }
    const std::vector<int64_t> strideVec = utils::GetVecFromJLongArray(env, jstride);
    const std::vector<int64_t> paddingVec = utils::GetVecFromJLongArray(env, jpadding);
    const std::vector<int64_t> dilationVec = utils::GetVecFromJLongArray(env, jdilation);

    torch::Tensor* result_ptr = nullptr;
    long dim = weigtht_ptr->dim() - 2;
    if (dim == 1) {
      result_ptr = new torch::Tensor(
        torch::conv1d(*tensor_ptr, *weigtht_ptr, bias, strideVec, paddingVec, dilationVec, jgroups));
    } else if (dim == 2) {
      result_ptr = new torch::Tensor(
        torch::conv2d(*tensor_ptr, *weigtht_ptr, bias, strideVec, paddingVec, dilationVec, jgroups));
    } else if (dim == 3) {
      result_ptr = new torch::Tensor(
        torch::conv3d(*tensor_ptr, *weigtht_ptr, bias, strideVec, paddingVec, dilationVec, jgroups));
    }
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNBatchNorm
  (JNIEnv* env, jobject jthis, jobject jinput, jobject jrunning_mean, jobject jrunning_var, jobject jweight, jobject jbias, jboolean jtraining, jdouble jmomentum, jdouble jeps) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jinput);
  const auto* running_mean_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jrunning_mean);
  const auto* running_var_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jrunning_var);
  torch::Tensor weight = {};
  torch::Tensor bias = {};
  if (!env->IsSameObject(jweight, nullptr)) {
    weight = *utils::GetPointerFromJHandle<const torch::Tensor>(env, jweight);
  }
  if (!env->IsSameObject(jbias, nullptr)) {
      bias = *utils::GetPointerFromJHandle<const torch::Tensor>(env, jbias);
  }
  const auto* result_ptr = new torch::Tensor(torch::nn::functional::batch_norm(
      *tensor_ptr, *running_mean_ptr, *running_var_ptr, torch::nn::functional::BatchNormFuncOptions().weight(weight).bias(bias).momentum(jmomentum).eps(jeps).training(jtraining)));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNDropout(
  JNIEnv* env, jobject jthis, jobject jinput, jdouble probability, jboolean jtraining) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jinput);
  const auto* result_ptr = new torch::Tensor(torch::nn::functional::dropout(*tensor_ptr, torch::nn::functional::DropoutFuncOptions().p(probability).training(jtraining)));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNRelu(
  JNIEnv* env, jobject jthis, jobject jinput) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jinput);
  // FIIXME the compiled libtorch have reference error
  // use torch::relu() for now until the fix
  const auto* result_ptr = new torch::Tensor(torch::relu(*tensor_ptr));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNSoftPlus(
  JNIEnv* env, jobject jthis, jobject jinput) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jinput);
  const auto* result_ptr = new torch::Tensor(torch::nn::functional::softplus(*tensor_ptr));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNSoftSign(
  JNIEnv* env, jobject jthis, jobject jinput) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jinput);
    const auto* result_ptr = new torch::Tensor(torch::nn::functional::softsign(*tensor_ptr));
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNLeakyRelu(
  JNIEnv* env, jobject jthis, jobject jinput, jdouble jnegative_slope) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jinput);
    const auto* result_ptr = new torch::Tensor(torch::nn::functional::leaky_relu(*tensor_ptr, torch::nn::functional::LeakyReLUFuncOptions().negative_slope(jnegative_slope)));
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNElu(
  JNIEnv* env, jobject jthis, jobject jinput, jdouble jalpha) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jinput);
    const auto* result_ptr = new torch::Tensor(torch::nn::functional::elu(*tensor_ptr, torch::nn::functional::ELUFuncOptions().alpha(jalpha)));
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNSelu(
  JNIEnv* env, jobject jthis, jobject jinput) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jinput);
    // FIIXME the compiled libtorch have reference error
    // use torch::selu() for now until the fix
    const auto* result_ptr = new torch::Tensor(torch::selu(*tensor_ptr));
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNGelu(
  JNIEnv* env, jobject jthis, jobject jinput) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jinput);
    const auto* result_ptr = new torch::Tensor(torch::nn::functional::gelu(*tensor_ptr));
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNMaxPool(
  JNIEnv* env, jobject jthis, jobject jhandle, jlongArray jkernel, jlongArray jstride, jlongArray jpadding, jboolean jceil_mode) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
    const std::vector<int64_t> kernel_vec = utils::GetVecFromJLongArray(env, jkernel);
    const std::vector<int64_t> stride_vec = utils::GetVecFromJLongArray(env, jstride);
    const std::vector<int64_t> padding_vec = utils::GetVecFromJLongArray(env, jpadding);
    torch::Tensor* result_ptr = nullptr;
    long dim = tensor_ptr->dim() - 2;
    if (dim == 1) {
      result_ptr = new torch::Tensor(torch::nn::functional::max_pool1d(*tensor_ptr, torch::nn::functional::MaxPool1dFuncOptions(kernel_vec).stride(stride_vec).padding(padding_vec).ceil_mode(jceil_mode)));
    } else if (dim == 2) {
      result_ptr = new torch::Tensor(torch::nn::functional::max_pool2d(*tensor_ptr, torch::nn::functional::MaxPool2dFuncOptions(kernel_vec).stride(stride_vec).padding(padding_vec).ceil_mode(jceil_mode)));
    } else if (dim == 3) {
      result_ptr = new torch::Tensor(torch::nn::functional::max_pool3d(*tensor_ptr, torch::nn::functional::MaxPool3dFuncOptions(kernel_vec).stride(stride_vec).padding(padding_vec).ceil_mode(jceil_mode)));
    }
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNAvgPool(
  JNIEnv* env, jobject jthis, jobject jinput, jlongArray jkernel_size, jlongArray jstride, jlongArray jpaddiing, jboolean jceil_mode, jboolean jcount_include_pad) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jinput);
    const std::vector<int64_t> kernel_vec = utils::GetVecFromJLongArray(env, jkernel_size);
    const std::vector<int64_t> stride_vec = utils::GetVecFromJLongArray(env, jstride);
    const std::vector<int64_t> padding_vec = utils::GetVecFromJLongArray(env, jpaddiing);

    torch::Tensor* result_ptr = nullptr;
    long dim = tensor_ptr->dim() - 2;
    if (dim == 1) {
      result_ptr = new torch::Tensor(torch::nn::functional::avg_pool1d(*tensor_ptr, torch::nn::functional::AvgPool1dFuncOptions(kernel_vec).stride(stride_vec).padding(padding_vec).ceil_mode(jceil_mode)));
    } else if (dim == 2) {
      result_ptr = new torch::Tensor(torch::nn::functional::avg_pool2d(*tensor_ptr, torch::nn::functional::AvgPool2dFuncOptions(kernel_vec).stride(stride_vec).padding(padding_vec).ceil_mode(jceil_mode)));
    } else if (dim == 3) {
      result_ptr = new torch::Tensor(torch::nn::functional::avg_pool3d(*tensor_ptr, torch::nn::functional::AvgPool3dFuncOptions(kernel_vec).stride(stride_vec).padding(padding_vec).ceil_mode(jceil_mode)));
    }
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNAdaptiveAvgPool(
  JNIEnv* env, jobject jthis, jobject jhandle, jlongArray joutput_size) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
    const std::vector<int64_t> output_vec = utils::GetVecFromJLongArray(env, joutput_size);

    torch::Tensor* result_ptr = nullptr;
    long dim = tensor_ptr->dim() - 2;
    if (dim == 1) {
      result_ptr = new torch::Tensor(torch::nn::functional::adaptive_avg_pool1d(*tensor_ptr, torch::nn::functional::AdaptiveAvgPool1dFuncOptions(output_vec)));
    } else if (dim == 2) {
      result_ptr = new torch::Tensor(torch::nn::functional::adaptive_avg_pool2d(*tensor_ptr, torch::nn::functional::AdaptiveAvgPool2dFuncOptions(output_vec)));
    } else if (dim == 3) {
      result_ptr = new torch::Tensor(torch::nn::functional::adaptive_avg_pool3d(*tensor_ptr, torch::nn::functional::AdaptiveAvgPool3dFuncOptions(output_vec)));
    }
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNAdaptiveMaxPool(
  JNIEnv* env, jobject jthis, jobject jhandle, jlongArray joutput_size) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
    const std::vector<int64_t> output_vec = utils::GetVecFromJLongArray(env, joutput_size);

    torch::Tensor* result_ptr = nullptr;
    long dim = tensor_ptr->dim() - 2;
    if (dim == 1) {
      result_ptr = new torch::Tensor(torch::nn::functional::adaptive_max_pool1d(*tensor_ptr, torch::nn::functional::AdaptiveMaxPool1dFuncOptions(output_vec)));
    } else if (dim == 2) {
      result_ptr = new torch::Tensor(torch::nn::functional::adaptive_max_pool2d(*tensor_ptr, torch::nn::functional::AdaptiveMaxPool2dFuncOptions(output_vec)));
    } else if (dim == 3) {
      result_ptr = new torch::Tensor(torch::nn::functional::adaptive_max_pool3d(*tensor_ptr, torch::nn::functional::AdaptiveMaxPool3dFuncOptions(output_vec)));
    }
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNNLpPool
  (JNIEnv* env, jobject jthis, jobject jinput, jdouble jnorm_type, jlongArray jkernel_size, jlongArray jstride, jboolean jceil_mode) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jinput);
    const std::vector<int64_t> kernel_vec = utils::GetVecFromJLongArray(env, jkernel_size);
    const std::vector<int64_t> stride_vec = utils::GetVecFromJLongArray(env, jstride);

    torch::Tensor* result_ptr = nullptr;
    long dim = tensor_ptr->dim() - 2;
    if (dim == 1) {
      result_ptr = new torch::Tensor(torch::nn::functional::lp_pool1d(*tensor_ptr, torch::nn::functional::LPPool1dFuncOptions(jnorm_type, kernel_vec).stride(stride_vec).ceil_mode(jceil_mode)));
    } else if (dim == 2) {
      result_ptr = new torch::Tensor(torch::nn::functional::lp_pool2d(*tensor_ptr, torch::nn::functional::LPPool2dFuncOptions(jnorm_type, kernel_vec).stride(stride_vec).ceil_mode(jceil_mode)));
    }
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}