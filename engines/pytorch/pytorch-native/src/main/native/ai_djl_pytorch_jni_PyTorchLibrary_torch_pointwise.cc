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

// The file is the implementation for PyTorch tensor pointwise ops

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAdd(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(self_ptr->add(*other_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAddi(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  self_ptr->add_(*other_ptr);
  API_END()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSub(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(self_ptr->sub(*other_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSubi(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  self_ptr->sub_(*other_ptr);
  API_END()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMul(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(self_ptr->mul(*other_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMuli(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  self_ptr->mul_(*other_ptr);
  API_END()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchClamp(
    JNIEnv* env, jobject jthis, jlong jself, jlong jmin, jlong jmax) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* min_ptr = reinterpret_cast<torch::Tensor*>(jmin);
  const auto* max_ptr = reinterpret_cast<torch::Tensor*>(jmax);
  const auto* result_ptr = new torch::Tensor(self_ptr->clamp(min_ptr->item(), max_ptr->item()));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchTrueDivide(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(self_ptr->div(*other_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchTrueDividei(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  self_ptr->div_(*other_ptr);
  API_END()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchWhere(
    JNIEnv* env, jobject jthis, jlong jcondition, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* condition_ptr = reinterpret_cast<torch::Tensor*>(jcondition);
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(torch::where(*condition_ptr, *self_ptr, *other_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchRemainder(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(self_ptr->remainder(*other_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchRemainderi(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  self_ptr->remainder_(*other_ptr);
  API_END()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchPow(
    JNIEnv* env, jobject jthis, jlong jself, jlong jexponent) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* exponent_ptr = reinterpret_cast<torch::Tensor*>(jexponent);
  const auto* result_ptr = new torch::Tensor(self_ptr->pow(*exponent_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchPowi(
    JNIEnv* env, jobject jthis, jlong jself, jlong jexponent) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jexponent);
  self_ptr->pow_(*other_ptr);
  API_END()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMatmul(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(self_ptr->matmul(*other_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchBmm(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(self_ptr->bmm(*other_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchXLogY(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(self_ptr->xlogy(*other_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDot(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(self_ptr->dot(*other_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMaximum(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(self_ptr->max(*other_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMinimum(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(self_ptr->min(*other_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMedian(
    JNIEnv* env, jobject jthis, jlong jself, jlong jdim, jboolean keep_dim) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto result = self_ptr->median(jdim, keep_dim);
  const auto* value_ptr = new torch::Tensor(std::get<0>(result));
  const auto* indices_ptr = new torch::Tensor(std::get<1>(result));
  std::vector<uintptr_t> vect;
  vect.push_back(reinterpret_cast<uintptr_t>(value_ptr));
  vect.push_back(reinterpret_cast<uintptr_t>(indices_ptr));
  return djl::utils::jni::GetLongArrayFromVec(env, vect);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchQuantile(
    JNIEnv* env, jobject jthis, jlong jself, jfloat q, jlong jdim, jboolean keep_dim) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* result_ptr = new torch::Tensor(torch::quantile(*self_ptr, q, jdim, keep_dim));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAbs(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->abs());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSquare(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->square());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchFloor(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->floor());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCeil(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->ceil());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchRound(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->round());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchTrunc(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->trunc());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchExp(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->exp());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLgamma(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->lgamma());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLog(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->log());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLog10(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->log10());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLog2(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->log2());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSin(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->sin());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCos(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->cos());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchTan(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->tan());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchASin(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->asin());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAcos(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->acos());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAtan(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->atan());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAtan2(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(self_ptr->atan2(*other_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSqrt(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->sqrt());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSinh(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->sinh());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCosh(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->cosh());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchTanh(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->tanh());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSigmoid(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->sigmoid());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAll(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->all());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAny(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->any());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNone(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->any().logical_not());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNeg(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->neg());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNegi(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  tensor_ptr->neg_();
  API_END()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLogicalAnd(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(torch::logical_and(*self_ptr, *other_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLogicalOr(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(torch::logical_or(*self_ptr, *other_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLogicalXor(
    JNIEnv* env, jobject jthis, jlong jself, jlong jother) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* other_ptr = reinterpret_cast<torch::Tensor*>(jother);
  const auto* result_ptr = new torch::Tensor(torch::logical_xor(*self_ptr, *other_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLogicalNot(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->logical_not());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSign(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->sign());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSigni(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  tensor_ptr->sign_();
  API_END()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchErfinv(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->erfinv());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchErf(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* tensor_ptr = reinterpret_cast<torch::Tensor*>(jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->erf());
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchInverse(JNIEnv* env, jobject jthis, jlong jself) {
  API_BEGIN()
  const auto* self_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* result_ptr = new torch::Tensor(torch::linalg_inv(*self_ptr));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDiff(JNIEnv* env, jobject jthis, jlong jself, jint n, jint dim) {
  API_BEGIN()
  const auto* input_ptr = reinterpret_cast<torch::Tensor*>(jself);
  const auto* result_ptr = new torch::Tensor(torch::diff(*input_ptr, n, dim));
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}
