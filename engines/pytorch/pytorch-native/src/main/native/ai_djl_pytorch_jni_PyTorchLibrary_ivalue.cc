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

#include <djl/utils.h>
#include <torch/script.h>

#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_exception.h"
#include "djl_pytorch_utils.h"

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueFromTensor(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  const auto* ivalue_ptr = new torch::IValue(*reinterpret_cast<torch::Tensor*>(jhandle));
  return reinterpret_cast<uintptr_t>(ivalue_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueFromBool(
    JNIEnv* env, jobject jthis, jboolean jvalue) {
  API_BEGIN()
  const auto* ivalue_ptr = new torch::IValue((bool) jvalue);
  return reinterpret_cast<uintptr_t>(ivalue_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueFromLong(
    JNIEnv* env, jobject jthis, jlong jvalue) {
  API_BEGIN()
  const auto* ivalue_ptr = new torch::IValue((int64_t) jvalue);
  return reinterpret_cast<uintptr_t>(ivalue_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueFromDouble(
    JNIEnv* env, jobject jthis, jdouble jvalue) {
  API_BEGIN()
  const auto* ivalue_ptr = new torch::IValue(jvalue);
  return reinterpret_cast<uintptr_t>(ivalue_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueFromString(
    JNIEnv* env, jobject jthis, jstring jvalue) {
  API_BEGIN()
  const std::string value = djl::utils::jni::GetStringFromJString(env, jvalue);
  const auto* ivalue_ptr = new torch::IValue(value);
  return reinterpret_cast<uintptr_t>(ivalue_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueFromBoolList(
    JNIEnv* env, jobject jthis, jbooleanArray jvalues) {
  API_BEGIN()
  jsize len = env->GetArrayLength(jvalues);
  jboolean* jptrs = env->GetBooleanArrayElements(jvalues, JNI_FALSE);
  torch::List<bool> list;
  list.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    list.emplace_back(jptrs[i]);
  }
  env->ReleaseBooleanArrayElements(jvalues, jptrs, JNI_ABORT);
  const auto* ivalue_ptr = new torch::IValue(list);
  return reinterpret_cast<uintptr_t>(ivalue_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueFromLongList(
    JNIEnv* env, jobject jthis, jlongArray jvalues) {
  API_BEGIN()
  jsize len = env->GetArrayLength(jvalues);
  jlong* jptrs = env->GetLongArrayElements(jvalues, JNI_FALSE);
  torch::List<int64_t> list;
  list.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    list.emplace_back(jptrs[i]);
  }
  env->ReleaseLongArrayElements(jvalues, jptrs, JNI_ABORT);
  const auto* ivalue_ptr = new torch::IValue(list);
  return reinterpret_cast<uintptr_t>(ivalue_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueFromDoubleList(
    JNIEnv* env, jobject jthis, jdoubleArray jvalues) {
  API_BEGIN()
  jsize len = env->GetArrayLength(jvalues);
  jdouble* jptrs = env->GetDoubleArrayElements(jvalues, JNI_FALSE);
  torch::List<double> list;
  list.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    list.emplace_back(jptrs[i]);
  }
  env->ReleaseDoubleArrayElements(jvalues, jptrs, JNI_ABORT);
  const auto* ivalue_ptr = new torch::IValue(list);
  return reinterpret_cast<uintptr_t>(ivalue_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueFromTensorList(
    JNIEnv* env, jobject jthis, jlongArray jvalues) {
  API_BEGIN()
  jsize len = env->GetArrayLength(jvalues);
  jlong* jptrs = env->GetLongArrayElements(jvalues, JNI_FALSE);
  torch::List<torch::Tensor> list;
  list.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    list.emplace_back(*reinterpret_cast<torch::Tensor*>(jptrs[i]));
  }
  env->ReleaseLongArrayElements(jvalues, jptrs, JNI_ABORT);
  const auto* ivalue_ptr = new torch::IValue(list);
  return reinterpret_cast<uintptr_t>(ivalue_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueFromList(
    JNIEnv* env, jobject jthis, jlongArray jvalues) {
  API_BEGIN()
  jsize len = env->GetArrayLength(jvalues);
  jlong* jptrs = env->GetLongArrayElements(jvalues, JNI_FALSE);
  auto* head = reinterpret_cast<torch::IValue*>(jptrs[0]);
  c10::impl::GenericList list{c10::unshapedType(head->type())};
  list.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    list.emplace_back(*reinterpret_cast<torch::IValue*>(jptrs[i]));
  }
  env->ReleaseLongArrayElements(jvalues, jptrs, JNI_ABORT);
  const auto* ivalue_ptr = new torch::IValue{list};
  return reinterpret_cast<uintptr_t>(ivalue_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueFromTuple(
    JNIEnv* env, jobject jthis, jlongArray jvalues) {
  API_BEGIN()
  jsize len = env->GetArrayLength(jvalues);
  jlong* jptrs = env->GetLongArrayElements(jvalues, JNI_FALSE);
  std::vector<torch::IValue> elements;
  elements.reserve(len);
  for (auto i = 0; i < len; ++i) {
    elements.emplace_back(*reinterpret_cast<torch::IValue*>(jptrs[i]));
  }
  c10::intrusive_ptr<c10::ivalue::Tuple> tuple = c10::ivalue::Tuple::create(std::move(elements));
  env->ReleaseLongArrayElements(jvalues, jptrs, JNI_ABORT);
  const auto* ivalue_ptr = new torch::IValue{tuple};
  return reinterpret_cast<uintptr_t>(ivalue_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueFromStringMap(
    JNIEnv* env, jobject jthis, jobjectArray jkeys, jlongArray jvalues) {
  API_BEGIN()
  auto len = static_cast<size_t>(env->GetArrayLength(jvalues));
  jlong* jptrs = env->GetLongArrayElements(jvalues, JNI_FALSE);
  torch::Dict<std::string, torch::Tensor> dict;
  dict.reserve(len);
  for (size_t i = 0; i < len; ++i) {
    auto jname = (jstring) env->GetObjectArrayElement(jkeys, i);
    std::string name = djl::utils::jni::GetStringFromJString(env, jname);
    dict.insert(name, *reinterpret_cast<torch::Tensor*>(jptrs[i]));
  }
  env->ReleaseLongArrayElements(jvalues, jptrs, JNI_ABORT);
  env->DeleteLocalRef(jkeys);
  const auto* ivalue_ptr = new torch::IValue(dict);
  return reinterpret_cast<uintptr_t>(ivalue_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueFromStringIValueMap(
    JNIEnv* env, jobject jthis, jobjectArray jkeys, jlongArray jvalues) {
  API_BEGIN()
  auto len = static_cast<size_t>(env->GetArrayLength(jvalues));
  jlong* jptrs = env->GetLongArrayElements(jvalues, JNI_FALSE);
  if (len == 0) {
    const auto* ivalue_ptr = new torch::IValue{c10::impl::GenericDict(c10::StringType::get(), c10::TensorType::get())};
    return reinterpret_cast<uintptr_t>(ivalue_ptr);
  }

  auto* firstEntryValue = reinterpret_cast<torch::IValue*>(jptrs[0]);
  c10::impl::GenericDict dict(c10::StringType::get(), c10::unshapedType(firstEntryValue->type()));
  for (size_t i = 0; i < len; ++i) {
    auto jname = (jstring) env->GetObjectArrayElement(jkeys, i);
    std::string name = djl::utils::jni::GetStringFromJString(env, jname);
    dict.insert(name, *reinterpret_cast<torch::IValue*>(jptrs[i]));
  }
  env->ReleaseLongArrayElements(jvalues, jptrs, JNI_ABORT);
  env->DeleteLocalRef(jkeys);
  const auto* ivalue_ptr = new torch::IValue{dict};
  return reinterpret_cast<uintptr_t>(ivalue_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToTensor(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* tensor_ptr = new torch::Tensor(reinterpret_cast<torch::IValue*>(jhandle)->toTensor());
  return reinterpret_cast<uintptr_t>(tensor_ptr);
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToBool(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = reinterpret_cast<torch::IValue*>(jhandle);
  return ivalue_ptr->toBool();
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToLong(JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = reinterpret_cast<torch::IValue*>(jhandle);
  return ivalue_ptr->toInt();
  API_END_RETURN()
}

JNIEXPORT jdouble JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToDouble(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = reinterpret_cast<torch::IValue*>(jhandle);
  return ivalue_ptr->toDouble();
  API_END_RETURN()
}

JNIEXPORT jstring JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToString(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = reinterpret_cast<torch::IValue*>(jhandle);
  return env->NewStringUTF(ivalue_ptr->toString()->string().c_str());
  API_END_RETURN()
}

JNIEXPORT jbooleanArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToBoolList(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = reinterpret_cast<torch::IValue*>(jhandle);
  torch::List<bool> list = ivalue_ptr->toBoolList();
  size_t len = list.size();
  jbooleanArray jarray = env->NewBooleanArray(len);
  std::vector<jboolean> jptrs;
  jptrs.resize(len);
  for (size_t i = 0; i < len; ++i) {
    jptrs[i] = list[i];
  }
  env->SetBooleanArrayRegion(jarray, 0, len, jptrs.data());
  return jarray;
  API_END_RETURN()
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToLongList(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = reinterpret_cast<torch::IValue*>(jhandle);
  torch::List<int64_t> list = ivalue_ptr->toIntList();
  size_t len = list.size();
  jlongArray jarray = env->NewLongArray(len);
  std::vector<jlong> jptrs;
  jptrs.resize(len);
  for (size_t i = 0; i < len; ++i) {
    jptrs[i] = list[i];
  }
  env->SetLongArrayRegion(jarray, 0, len, jptrs.data());
  return jarray;
  API_END_RETURN()
}

JNIEXPORT jdoubleArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToDoubleList(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = reinterpret_cast<torch::IValue*>(jhandle);
  torch::List<double> list = ivalue_ptr->toDoubleList();
  size_t len = list.size();
  jdoubleArray jarray = env->NewDoubleArray(len);
  std::vector<jdouble> jptrs;
  jptrs.resize(len);
  for (size_t i = 0; i < len; ++i) {
    jptrs[i] = list[i];
  }
  env->SetDoubleArrayRegion(jarray, 0, len, jptrs.data());
  return jarray;
  API_END_RETURN()
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToTensorList(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = reinterpret_cast<torch::IValue*>(jhandle);
  torch::List<torch::Tensor> tensor_list = ivalue_ptr->toTensorList();
  return djl::utils::jni::GetPtrArrayFromContainer<torch::List<torch::Tensor>, torch::Tensor>(env, tensor_list);
  API_END_RETURN()
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToIValueList(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = reinterpret_cast<torch::IValue*>(jhandle);
  torch::List<torch::IValue> ivalue_list = ivalue_ptr->toList();
  return djl::utils::jni::GetPtrArrayFromContainer<torch::List<torch::IValue>, torch::IValue>(env, ivalue_list);
  API_END_RETURN()
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToIValueTuple(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = reinterpret_cast<torch::IValue*>(jhandle);
  std::vector<torch::IValue> ivalue_vec = ivalue_ptr->toTuple()->elements();
  return djl::utils::jni::GetPtrArrayFromContainer<std::vector<torch::IValue>, torch::IValue>(env, ivalue_vec);
  API_END_RETURN()
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToMap(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = reinterpret_cast<torch::IValue*>(jhandle);
  torch::Dict<torch::IValue, torch::IValue> dict = ivalue_ptr->toGenericDict();
  size_t len = dict.size() * 2;
  jlongArray jarray = env->NewLongArray(len);
  std::vector<jlong> jptrs;
  jptrs.resize(len);
  size_t array_iter = 0;
  for (auto it = dict.begin(); it != dict.end(); ++it) {
    const auto* key_ptr = new torch::IValue(it->key());
    jptrs[array_iter++] = reinterpret_cast<uintptr_t>(key_ptr);
    const auto* value_ptr = new torch::IValue(it->value());
    jptrs[array_iter++] = reinterpret_cast<uintptr_t>(value_ptr);
  }
  env->SetLongArrayRegion(jarray, 0, len, jptrs.data());
  return jarray;
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsTensor(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  return reinterpret_cast<torch::IValue*>(jhandle)->isTensor();
  API_END_RETURN()
}

JNIEXPORT jstring JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueGetType(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  std::string type = reinterpret_cast<torch::IValue*>(jhandle)->type()->str();
  return env->NewStringUTF(type.c_str());
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsBool(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  return reinterpret_cast<torch::IValue*>(jhandle)->isBool();
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsLong(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  return reinterpret_cast<torch::IValue*>(jhandle)->isInt();
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsDouble(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  return reinterpret_cast<torch::IValue*>(jhandle)->isDouble();
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsString(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  return reinterpret_cast<torch::IValue*>(jhandle)->isString();
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsBoolList(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  return reinterpret_cast<torch::IValue*>(jhandle)->isBoolList();
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsLongList(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  return reinterpret_cast<torch::IValue*>(jhandle)->isIntList();
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsDoubleList(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  return reinterpret_cast<torch::IValue*>(jhandle)->isDoubleList();
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsTensorList(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  return reinterpret_cast<torch::IValue*>(jhandle)->isTensorList();
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsList(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  return reinterpret_cast<torch::IValue*>(jhandle)->isList();
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsMap(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  return reinterpret_cast<torch::IValue*>(jhandle)->isGenericDict();
  API_END_RETURN()
}

JNIEXPORT jboolean JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueIsTuple(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  return reinterpret_cast<torch::IValue*>(jhandle)->isTuple();
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteIValue(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* ivalue_ptr = reinterpret_cast<torch::IValue*>(jhandle);
  delete ivalue_ptr;
  API_END()
}
