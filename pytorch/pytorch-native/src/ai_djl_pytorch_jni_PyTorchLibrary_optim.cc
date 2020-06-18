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

#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_utils.h"

// The file is the implementation for PyTorch training operations

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_adamUpdate(
  JNIEnv* env, jobject jthis, jobject jweight, jobject jgrad, jobject jmean, jobject jvariance, jfloat learning_rate, jfloat weight_decay,
  jfloat rescale_grad, jfloat clip_grad, jfloat beta1, jfloat beta2, jfloat eps) {
  torch::autograd::AutoGradMode no_autograd_guard{false};
  const auto* weight_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jweight);
  const auto grad = utils::GetPointerFromJHandle<torch::Tensor>(env, jgrad)->clone();
  const auto* mean_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jmean);
  const auto* variance_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jvariance);
  // following this formula: rescaled_grad = clip(rescale_grad * grad, clip_gradient)) + wd * weight
  if (rescale_grad != 1.0) {
    grad.mul_(rescale_grad);
  }
  if (clip_grad >= 0.0) {
    // Add clip grad option
    grad.clamp_max_(clip_grad);
  }
  grad.add_(*weight_ptr, weight_decay);
  mean_ptr->mul_(beta1).add_(grad, 1 - beta1);
  variance_ptr->mul_(beta2).addcmul_(grad, grad, 1 - beta2);
  weight_ptr->sub_(mean_ptr->mul(learning_rate).div(variance_ptr->sqrt().add(eps)));
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_sgdUpdate(
  JNIEnv* env, jobject jthis, jobject jweight, jobject jgrad, jobject jstate, jfloat learning_rate, jfloat weight_decay,
  jfloat rescale_grad, jfloat clip_grad, jfloat momentum) {
  // disable gradient calculation
  torch::autograd::AutoGradMode no_autograd_guard{false};
  const auto* weight_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jweight);
  // use clone to avoid input grad change
  auto grad = utils::GetPointerFromJHandle<torch::Tensor>(env, jgrad)->clone();
  // following this formula: rescaled_grad = clip(rescale_grad * grad, clip_gradient)) + wd * weight
  if (rescale_grad != 1.0) {
    grad.mul_(rescale_grad);
  }
  // TODO: MXNet convension, if < 0, it won't clip
  if (clip_grad >= 0.0) {
    // Add clip grad option
    grad.clamp_max_(clip_grad);
  }
  grad.add_(*weight_ptr, weight_decay).mul_(learning_rate);
  // TODO: implementation in DJL is different than PyTorch with missing dampening and nesterov
  if (momentum == 0.0) {
    weight_ptr->sub_(grad);
  } else {
    const auto* state_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jstate);
    state_ptr->mul_(momentum).add_(grad);
    weight_ptr->sub_(*state_ptr);
  }
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_zeroGrad
  (JNIEnv* env, jobject jthis, jobject jhandle) {
  const auto* weight_ptr = utils::GetPointerFromJHandle<torch::Tensor>(env, jhandle);
  weight_ptr->grad().detach_();
  weight_ptr->grad().zero_();
}
