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
#include "djl_pytorch_jni_exception.h"
#include "djl_pytorch_utils.h"

// The file is the implementation for PyTorch training operations

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_adamUpdate(JNIEnv* env, jobject jthis, jlong jweight,
    jlong jgrad, jlong jmean, jlong jvariance, jfloat learning_rate, jfloat learning_rate_bias_correction,
    jfloat weight_decay, jfloat rescale_grad, jfloat clip_grad, jfloat beta1, jfloat beta2, jfloat eps,
    jboolean adamw) {
  API_BEGIN()
  const auto* weight_ptr = reinterpret_cast<torch::Tensor*>(jweight);
  const auto grad = reinterpret_cast<torch::Tensor*>(jgrad)->clone();
  const auto* mean_ptr = reinterpret_cast<torch::Tensor*>(jmean);
  const auto* variance_ptr = reinterpret_cast<torch::Tensor*>(jvariance);
  // following this formula: rescaled_grad = clip(rescale_grad * grad, clip_gradient)) + wd * weight
  if (rescale_grad != 1.0) {
    grad.mul_(rescale_grad);
  }
  if (clip_grad >= 0.0) {
    // Add clip grad option
    grad.clamp_max_(clip_grad);
  }
  if (!adamw) {
    // rescaled_grad is obtained here
    grad.add_(*weight_ptr, weight_decay);
  } else {
    weight_ptr->sub_(weight_ptr->mul(learning_rate).mul(weight_decay));
  }
  mean_ptr->mul_(beta1).add_(grad, 1 - beta1);
  variance_ptr->mul_(beta2).addcmul_(grad, grad, 1 - beta2);
  weight_ptr->sub_(mean_ptr->mul(learning_rate_bias_correction).div(variance_ptr->sqrt().add(eps)));
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_sgdUpdate(JNIEnv* env, jobject jthis, jlong jweight,
    jlong jgrad, jlong jstate, jfloat learning_rate, jfloat weight_decay, jfloat rescale_grad, jfloat clip_grad,
    jfloat momentum) {
  API_BEGIN()
  // disable gradient calculation
  const auto* weight_ptr = reinterpret_cast<torch::Tensor*>(jweight);
  // use clone to avoid input grad change
  auto grad = reinterpret_cast<torch::Tensor*>(jgrad)->clone();
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
    const auto* state_ptr = reinterpret_cast<torch::Tensor*>(jstate);
    state_ptr->mul_(momentum).add_(grad);
    weight_ptr->sub_(*state_ptr);
  }
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_zeroGrad(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  torch::NoGradGuard NoGradGuard;
  const auto* weight_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  // the check is only for batch_size < # of gpus
  // where some required_grad weights never call backward
  // TODO we should avoid the create parameter but not applying backward
  if (weight_ptr->grad().defined()) {
    weight_ptr->grad().zero_();
  }
  API_END()
}
