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
#include "ai_djl_dlr_jni_DlrLibrary.h"

#include <jni.h>

#include "dlr.h"

JNIEXPORT jint JNICALL Java_ai_djl_dlr_jni_DlrLibrary_getDlrNumInputs(JNIEnv* env, jobject jthis, jlong jhandle) {
  int num;
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  if (GetDLRNumInputs(handle, &num)) {
    return -1;
  }
  return num;
}

JNIEXPORT jint JNICALL Java_ai_djl_dlr_jni_DlrLibrary_getDlrNumWeights(JNIEnv* env, jobject jthis, jlong jhandle) {
  int num;
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  if (GetDLRNumWeights(handle, &num)) {
    return -1;
  }
  return num;
}

JNIEXPORT jstring JNICALL Java_ai_djl_dlr_jni_DlrLibrary_getDlrInputName(
    JNIEnv* env, jobject jthis, jlong jhandle, jint jindex) {
  const char* name;
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  if (GetDLRInputName(handle, jindex, &name)) {
    return nullptr;
  }
  return env->NewStringUTF(name);
}

JNIEXPORT jstring JNICALL Java_ai_djl_dlr_jni_DlrLibrary_getDlrWeightName(
    JNIEnv* env, jobject jthis, jlong jhandle, jint jindex) {
  const char* name;
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  if (GetDLRWeightName(handle, jindex, &name)) {
    return nullptr;
  }
  return env->NewStringUTF(name);
}

JNIEXPORT jint JNICALL Java_ai_djl_dlr_jni_DlrLibrary_setDlrInput(
    JNIEnv* env, jobject jthis, jlong jhandle, jstring jname, jlongArray shape, jfloatArray input, jint dim) {
  jfloat* input_body = env->GetFloatArrayElements(input, JNI_FALSE);
  jlong* shape_body = env->GetLongArrayElements(shape, JNI_FALSE);
  const char* name = env->GetStringUTFChars(jname, JNI_FALSE);
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  int res = SetDLRInput(handle, name, shape_body, input_body, dim);
  env->ReleaseLongArrayElements(shape, shape_body, 0);
  return res;
}

JNIEXPORT jint JNICALL Java_ai_djl_dlr_jni_DlrLibrary_getDlrInput(
    JNIEnv* env, jobject jthis, jlong jhandle, jstring jname, jfloatArray jinput) {
  jfloat* arr_body = env->GetFloatArrayElements(jinput, JNI_FALSE);
  const char* name = env->GetStringUTFChars(jname, JNI_FALSE);
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  int res = GetDLRInput(handle, name, arr_body);
  env->ReleaseFloatArrayElements(jinput, arr_body, 0);
  return res;
}

JNIEXPORT jint JNICALL Java_ai_djl_dlr_jni_DlrLibrary_getDlrOutputShape(
    JNIEnv* env, jobject jthis, jlong jhandle, jint jindex, jlongArray jshape) {
  jboolean isCopy = JNI_FALSE;
  jlong* arr_body = env->GetLongArrayElements(jshape, &isCopy);
  DLRModelHandle* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  int res = GetDLROutputShape(handle, jindex, arr_body);
  env->ReleaseLongArrayElements(jshape, arr_body, 0);
  return res;
}

JNIEXPORT jint JNICALL Java_ai_djl_dlr_jni_DlrLibrary_getDlrOutput(
    JNIEnv* env, jobject jthis, jlong jhandle, jint jindex, jfloatArray joutput) {
  jfloat* arr_body = env->GetFloatArrayElements(joutput, JNI_FALSE);
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  int res = GetDLROutput(handle, jindex, arr_body);
  env->ReleaseFloatArrayElements(joutput, arr_body, 0);
  return res;
}

JNIEXPORT jint JNICALL Java_ai_djl_dlr_jni_DlrLibrary_getDlrOutputDim(
    JNIEnv* env, jobject jthis, jlong jhandle, jint jindex) {
  int64_t out_size;
  int out_dim;
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  if (GetDLROutputSizeDim(handle, jindex, &out_size, &out_dim)) {
    return -1;
  }
  return out_dim;
}

JNIEXPORT jlong JNICALL Java_ai_djl_dlr_jni_DlrLibrary_getDlrOutputSize(
    JNIEnv* env, jobject jthis, jlong jhandle, jint jindex) {
  int64_t out_size;
  int out_dim;
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  if (GetDLROutputSizeDim(handle, jindex, &out_size, &out_dim)) {
    return -1;
  }
  return out_size;
}

JNIEXPORT jint JNICALL Java_ai_djl_dlr_jni_DlrLibrary_getDlrNumOutputs(JNIEnv* env, jobject jthis, jlong jhandle) {
  int num;
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  if (GetDLRNumOutputs(handle, &num)) {
    return -1;
  }
  return num;
}

JNIEXPORT jlong JNICALL Java_ai_djl_dlr_jni_DlrLibrary_createDlrModel(
    JNIEnv* env, jobject jthis, jstring jmodel_path, jint jdev_type, jint jdev_id) {
  const char* model_path = env->GetStringUTFChars(jmodel_path, JNI_FALSE);
  auto* handle = new DLRModelHandle();
  if (CreateDLRModel(handle, model_path, jdev_type, jdev_id)) {
    // FAIL
    return 0;
  }
  // Return handle as jlong
  uintptr_t jhandle = reinterpret_cast<uintptr_t>(handle);
  return jhandle;
}

JNIEXPORT jint JNICALL Java_ai_djl_dlr_jni_DlrLibrary_deleteDlrModel(JNIEnv* env, jobject jthis, jlong jhandle) {
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  return DeleteDLRModel(handle);
}

JNIEXPORT jint JNICALL Java_ai_djl_dlr_jni_DlrLibrary_runDlrModel(JNIEnv* env, jobject jthis, jlong jhandle) {
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  return RunDLRModel(handle);
}

JNIEXPORT jstring JNICALL Java_ai_djl_dlr_jni_DlrLibrary_dlrGetLastError(JNIEnv* env, jobject jthis) {
  const char* err = DLRGetLastError();
  return env->NewStringUTF(err);
}

JNIEXPORT jstring JNICALL Java_ai_djl_dlr_jni_DlrLibrary_getDlrBackend(JNIEnv* env, jobject jthis, jlong jhandle) {
  const char* name;
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  if (GetDLRBackend(handle, &name)) {
    return nullptr;
  }
  return env->NewStringUTF(name);
}

JNIEXPORT jint JNICALL Java_ai_djl_dlr_jni_DlrLibrary_setDlrNumThreads(
    JNIEnv* env, jobject jthis, jlong jhandle, jint jthreads) {
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  return SetDLRNumThreads(handle, jthreads);
}

JNIEXPORT jint JNICALL Java_ai_djl_dlr_jni_DlrLibrary_useDlrCPUAffinity(
    JNIEnv* env, jobject jthis, jlong jhandle, jboolean juse) {
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  return UseDLRCPUAffinity(handle, juse);
}
