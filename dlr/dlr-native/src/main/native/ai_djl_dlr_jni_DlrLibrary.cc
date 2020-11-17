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

#include <vector>

#include "dlr.h"

inline void CheckStatus(JNIEnv* env, int status) {
  if (status) {
    jclass jexception = env->FindClass("ai/djl/engine/EngineException");
    const char* err = DLRGetLastError();
    env->ThrowNew(jexception, err);
  }
}

JNIEXPORT jint JNICALL Java_ai_djl_dlr_jni_DlrLibrary_getDlrNumInputs(JNIEnv* env, jobject jthis, jlong jhandle) {
  int num;
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  CheckStatus(env, GetDLRNumInputs(handle, &num));
  return num;
}

JNIEXPORT jint JNICALL Java_ai_djl_dlr_jni_DlrLibrary_getDlrNumWeights(JNIEnv* env, jobject jthis, jlong jhandle) {
  int num;
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  CheckStatus(env, GetDLRNumWeights(handle, &num));
  return num;
}

JNIEXPORT jstring JNICALL Java_ai_djl_dlr_jni_DlrLibrary_getDlrInputName(
    JNIEnv* env, jobject jthis, jlong jhandle, jint jindex) {
  const char* name;
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  CheckStatus(env, GetDLRInputName(handle, jindex, &name));
  return env->NewStringUTF(name);
}

JNIEXPORT jstring JNICALL Java_ai_djl_dlr_jni_DlrLibrary_getDlrWeightName(
    JNIEnv* env, jobject jthis, jlong jhandle, jint jindex) {
  const char* name;
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  CheckStatus(env, GetDLRWeightName(handle, jindex, &name));
  return env->NewStringUTF(name);
}

JNIEXPORT void JNICALL Java_ai_djl_dlr_jni_DlrLibrary_setDLRInput(
    JNIEnv* env, jobject jthis, jlong jhandle, jstring jname, jlongArray jshape, jfloatArray jinput, jint jdim) {
  jfloat* input_body = env->GetFloatArrayElements(jinput, JNI_FALSE);
  jlong* shape_body = env->GetLongArrayElements(jshape, JNI_FALSE);
  const char* name = env->GetStringUTFChars(jname, JNI_FALSE);
  ;
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  CheckStatus(env, SetDLRInput(handle, name, reinterpret_cast<int64_t*>(shape_body), input_body, jdim));
  env->ReleaseFloatArrayElements(jinput, input_body, JNI_ABORT);
  env->ReleaseLongArrayElements(jshape, shape_body, JNI_ABORT);
  env->ReleaseStringUTFChars(jname, name);
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_dlr_jni_DlrLibrary_getDlrOutputShape(
    JNIEnv* env, jobject jthis, jlong jhandle, jint jindex) {
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  int64_t size;
  int dim;
  CheckStatus(env, GetDLROutputSizeDim(handle, jindex, &size, &dim));
  jlong shape[dim];
  CheckStatus(env, GetDLROutputShape(handle, jindex, reinterpret_cast<int64_t*>(shape)));
  jlongArray res = env->NewLongArray(dim);
  env->SetLongArrayRegion(res, 0, dim, shape);
  return res;
}

JNIEXPORT jfloatArray JNICALL Java_ai_djl_dlr_jni_DlrLibrary_getDlrOutput(
    JNIEnv* env, jobject jthis, jlong jhandle, jint jindex) {
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  int64_t size;
  int dim;
  CheckStatus(env, GetDLROutputSizeDim(handle, jindex, &size, &dim));
  float data[size];
  CheckStatus(env, GetDLROutput(handle, jindex, data));
  jfloatArray res = env->NewFloatArray(size);
  env->SetFloatArrayRegion(res, 0, size, data);
  return res;
}

JNIEXPORT jint JNICALL Java_ai_djl_dlr_jni_DlrLibrary_getDlrNumOutputs(JNIEnv* env, jobject jthis, jlong jhandle) {
  int num;
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  CheckStatus(env, GetDLRNumOutputs(handle, &num));
  return num;
}

JNIEXPORT jlong JNICALL Java_ai_djl_dlr_jni_DlrLibrary_createDlrModel(
    JNIEnv* env, jobject jthis, jstring jmodel_path, jint jdev_type, jint jdev_id) {
  const char* model_path = env->GetStringUTFChars(jmodel_path, JNI_FALSE);
  auto* handle = new DLRModelHandle();
  CheckStatus(env, CreateDLRModel(handle, model_path, jdev_type, jdev_id));
  auto jhandle = reinterpret_cast<uintptr_t>(handle);
  return jhandle;
}

JNIEXPORT void JNICALL Java_ai_djl_dlr_jni_DlrLibrary_deleteDlrModel(JNIEnv* env, jobject jthis, jlong jhandle) {
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  CheckStatus(env, DeleteDLRModel(handle));
}

JNIEXPORT void JNICALL Java_ai_djl_dlr_jni_DlrLibrary_runDlrModel(JNIEnv* env, jobject jthis, jlong jhandle) {
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  CheckStatus(env, RunDLRModel(handle));
}

JNIEXPORT jstring JNICALL Java_ai_djl_dlr_jni_DlrLibrary_getDlrBackend(JNIEnv* env, jobject jthis, jlong jhandle) {
  const char* name;
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  CheckStatus(env, GetDLRBackend(handle, &name));
  return env->NewStringUTF(name);
}

JNIEXPORT jstring JNICALL Java_ai_djl_dlr_jni_DlrLibrary_getDlrVersion(JNIEnv* env, jobject jthis) {
  const char* version;
  CheckStatus(env, GetDLRVersion(&version));
  return env->NewStringUTF(version);
}

JNIEXPORT void JNICALL Java_ai_djl_dlr_jni_DlrLibrary_setDlrNumThreads(
    JNIEnv* env, jobject jthis, jlong jhandle, jint jthreads) {
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  CheckStatus(env, SetDLRNumThreads(handle, jthreads));
}

JNIEXPORT void JNICALL Java_ai_djl_dlr_jni_DlrLibrary_useDlrCPUAffinity(
    JNIEnv* env, jobject jthis, jlong jhandle, jboolean juse) {
  auto* handle = reinterpret_cast<DLRModelHandle*>(jhandle);
  CheckStatus(env, UseDLRCPUAffinity(handle, juse));
}
