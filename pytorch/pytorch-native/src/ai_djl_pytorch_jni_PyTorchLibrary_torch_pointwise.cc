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
#include "djl_pytorch_jni_error.h"
#include "djl_pytorch_jni_utils.h"

// The file is the implementation for PyTorch tensor pointwise ops

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAdd(
    JNIEnv* env, jobject jthis, jobject jself, jobject jother) {
  API_BEGIN();
  const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
  const auto* other_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jother);
  const auto* result_ptr = new torch::Tensor(self_ptr->add(*other_ptr));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAddi(
    JNIEnv* env, jobject jthis, jobject jself, jobject jother) {
  const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
  const auto* other_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jother);
  self_ptr->add_(*other_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSub(
    JNIEnv* env, jobject jthis, jobject jself, jobject jother) {
  API_BEGIN();
  const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
  const auto* other_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jother);
  const auto* result_ptr = new torch::Tensor(self_ptr->sub(*other_ptr));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSubi(
    JNIEnv* env, jobject jthis, jobject jself, jobject jother) {
  const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
  const auto* other_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jother);
  self_ptr->sub_(*other_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMul(
    JNIEnv* env, jobject jthis, jobject jself, jobject jother) {
  API_BEGIN();
  const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
  const auto* other_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jother);
  const auto* result_ptr = new torch::Tensor(self_ptr->mul(*other_ptr));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMuli(
    JNIEnv* env, jobject jthis, jobject jself, jobject jother) {
  const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
  const auto* other_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jother);
  self_ptr->mul_(*other_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchClamp(
    JNIEnv* env, jobject jthis, jobject jself, jobject jmin, jobject jmax) {
  API_BEGIN();
  const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
  const auto* min_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jmin);
  const auto* max_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jmax);
  const auto* result_ptr = new torch::Tensor(self_ptr->clamp(min_ptr->item(), max_ptr->item()));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchTrueDivide(
    JNIEnv* env, jobject jthis, jobject jself, jobject jother) {
  API_BEGIN();
  const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
  const auto* other_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jother);
  const auto* result_ptr = new torch::Tensor(self_ptr->div(*other_ptr));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchTrueDividei(
    JNIEnv* env, jobject jthis, jobject jself, jobject jother) {
  const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
  const auto* other_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jother);
  self_ptr->div_(*other_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchWhere(
  JNIEnv* env, jobject jthis, jobject jcondition, jobject jself, jobject jother) {
  API_BEGIN();
  const auto* condition_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jcondition);
  const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
  const auto* other_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jother);
  const auto* result_ptr = new torch::Tensor(torch::where(*condition_ptr, *self_ptr, *other_ptr));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchRemainder(
    JNIEnv* env, jobject jthis, jobject jself, jobject jother) {
  API_BEGIN();
  const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
  const auto* other_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jother);
  const auto* result_ptr = new torch::Tensor(self_ptr->remainder(*other_ptr));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchRemainderi(
    JNIEnv* env, jobject jthis, jobject jself, jobject jother) {
  const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
  const auto* other_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jother);
  self_ptr->remainder_(*other_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchPow(
    JNIEnv* env, jobject jthis, jobject jself, jobject jexponent) {
  API_BEGIN();
  const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
  const auto* exponent_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jexponent);
  const auto* result_ptr = new torch::Tensor(self_ptr->pow(*exponent_ptr));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchPowi(
    JNIEnv* env, jobject jthis, jobject jself, jobject jexponent) {
  const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
  const auto* other_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jexponent);
  self_ptr->pow_(*other_ptr);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMatmul(
    JNIEnv* env, jobject jthis, jobject jself, jobject jother) {
  API_BEGIN();
  const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
  const auto* other_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jother);
  const auto* result_ptr = new torch::Tensor(self_ptr->matmul(*other_ptr));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDot(
  JNIEnv* env, jobject jthis, jobject jself, jobject jother) {
  API_BEGIN();
    const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
    const auto* other_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jother);
    const auto* result_ptr = new torch::Tensor(self_ptr->dot(*other_ptr));
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMM(
  JNIEnv* env, jobject jthis, jobject jself, jobject jother) {
  API_BEGIN();
    const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
    const auto* other_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jother);
    const auto* result_ptr = new torch::Tensor(self_ptr->mm(*other_ptr));
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMax__Lai_djl_pytorch_jni_Pointer_2Lai_djl_pytorch_jni_Pointer_2(
    JNIEnv* env, jobject jthis, jobject jself, jobject jother) {
  API_BEGIN();
  const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
  const auto* other_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jother);
  const auto* result_ptr = new torch::Tensor(self_ptr->max(*other_ptr));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_torchMin__Lai_djl_pytorch_jni_Pointer_2Lai_djl_pytorch_jni_Pointer_2(
    JNIEnv* env, jobject jthis, jobject jself, jobject jother) {
  API_BEGIN();
  const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
  const auto* other_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jother);
  const auto* result_ptr = new torch::Tensor(self_ptr->min(*other_ptr));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAbs(JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->abs());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSquare(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->square());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchFloor(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->floor());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCeil(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->ceil());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchRound(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->round());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchTrunc(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->trunc());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchExp(JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->exp());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLog(JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->log());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLog10(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->log10());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLog2(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->log2());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSin(JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->sin());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCos(JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->cos());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchTan(JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->tan());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchASin(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->asin());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAcos(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->acos());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAtan(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->atan());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSqrt(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->sqrt());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSinh(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->sinh());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchCosh(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->cosh());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchTanh(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->tanh());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchSigmoid(
  JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
    const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
    const auto* result_ptr = new torch::Tensor(tensor_ptr->sigmoid());
    return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAll(JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->all());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchAny(JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->any());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNone(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->any().logical_not());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNeg(JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->neg());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchNegi(JNIEnv* env, jobject jthis, jobject jhandle) {
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  tensor_ptr->neg_();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLogicalAnd(
    JNIEnv* env, jobject jthis, jobject jself, jobject jother) {
  API_BEGIN();
  const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
  const auto* other_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jother);
  const auto* result_ptr = new torch::Tensor(torch::logical_and(*self_ptr, *other_ptr));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLogicalOr(
    JNIEnv* env, jobject jthis, jobject jself, jobject jother) {
  API_BEGIN();
  const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
  const auto* other_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jother);
  const auto* result_ptr = new torch::Tensor(torch::logical_or(*self_ptr, *other_ptr));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLogicalXor(
    JNIEnv* env, jobject jthis, jobject jself, jobject jother) {
  API_BEGIN();
  const auto* self_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jself);
  const auto* other_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jother);
  const auto* result_ptr = new torch::Tensor(torch::logical_xor(*self_ptr, *other_ptr));
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchLogicalNot(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  API_BEGIN();
  const auto* tensor_ptr = utils::GetPointerFromJHandle<const torch::Tensor>(env, jhandle);
  const auto* result_ptr = new torch::Tensor(tensor_ptr->logical_not());
  return utils::CreatePointer<torch::Tensor>(env, result_ptr);
  API_END();
}
