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

#include "ai_djl_tensorrt_jni_TrtLibrary.h"

#include <djl/utils.h>
#include <jni.h>

#include <cctype>
#include <iterator>
#include <string>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "ai_djl_tensorrt_jni_common.h"
#include "ai_djl_tensorrt_jni_exception.h"
#include "ai_djl_tensorrt_jni_log.h"
#include "ai_djl_tensorrt_jni_model.h"

using namespace nvinfer1;
using namespace djl_trt;

JNIEXPORT void JNICALL Java_ai_djl_tensorrt_jni_TrtLibrary_initPlugins(
    JNIEnv *env, jobject jthis, jstring jnamespace, jint jseverity) {
  API_BEGIN()
  const char *lib_namespace = env->GetStringUTFChars(jnamespace, JNI_FALSE);
  initLibNvInferPlugins(&gLogger.getTrtLogger(), lib_namespace);
  gLogger.setSeverity(static_cast<Severity>(jseverity));
  API_END()
}

JNIEXPORT jlong JNICALL Java_ai_djl_tensorrt_jni_TrtLibrary_loadTrtModel(JNIEnv *env, jobject jthis, jint jmodel_type,
    jstring jmodel_path, jint jdev_id, jobjectArray jkeys, jobjectArray jvalues) {
  API_BEGIN()
  djl_trt::ModelParams params;
  params.modelType = jmodel_type;
  params.modelPath = env->GetStringUTFChars(jmodel_path, JNI_FALSE);
  params.device = jdev_id;

  CHECK(cudaSetDevice(jdev_id));

  auto len = static_cast<size_t>(env->GetArrayLength(jkeys));
  for (int i = 0; i < len; ++i) {
    auto jkey = (jstring) env->GetObjectArrayElement(jkeys, i);
    std::string name = djl::utils::jni::GetStringFromJString(env, jkey);
    auto jvalue = (jstring) env->GetObjectArrayElement(jvalues, i);

    if (name == "int8") {
      params.int8 = true;
    } else if (name == "fp16") {
      params.fp16 = true;
    } else if (name == "dlaCore") {
      params.dlaCore = stoi(djl::utils::jni::GetStringFromJString(env, jvalue));
    } else if (jmodel_type == 1 && name == "maxBatchSize") {
      params.maxBatchSize = stoi(djl::utils::jni::GetStringFromJString(env, jvalue));
    } else if (jmodel_type == 1 && name == "uffNHWC") {
      params.uffNHWC = true;
    } else if (jmodel_type == 1 && name == "uffInputs") {
      std::string value = djl::utils::jni::GetStringFromJString(env, jvalue);
      params.uffInputs = parseUffInputs(value);
    } else if (jmodel_type == 1 && name == "uffOutputs") {
      std::string value = djl::utils::jni::GetStringFromJString(env, jvalue);
      params.uffOutputs = splitString(value);
    }
  }
  if (jmodel_type == 1 && params.maxBatchSize == 0) {
    params.maxBatchSize = 1;
  }

  auto *model = new TrtModel(params);
  try {
    model->buildModel();
  } catch (const std::exception &e_) {
    delete model;
    throw;
  }

  auto jhandle = reinterpret_cast<uintptr_t>(model);
  return jhandle;
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_tensorrt_jni_TrtLibrary_deleteTrtModel(JNIEnv *env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto *model = reinterpret_cast<TrtModel *>(jhandle);
  delete model;
  API_END()
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_tensorrt_jni_TrtLibrary_getInputNames(
    JNIEnv *env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto *model = reinterpret_cast<TrtModel *>(jhandle);
  std::vector<std::string> names = model->getInputNames();
  return djl::utils::jni::GetStringArrayFromVec(env, names);
  API_END_RETURN()
}

JNIEXPORT jintArray JNICALL Java_ai_djl_tensorrt_jni_TrtLibrary_getInputDataTypes(
    JNIEnv *env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto *model = reinterpret_cast<TrtModel *>(jhandle);
  std::vector<nvinfer1::DataType> data_types = model->getInputTypes();
  int size = data_types.size();
  jintArray jarray = env->NewIntArray(size);
  jint elements[size];
  for (int i = 0; i < size; i++) {
    elements[i] = static_cast<int>(data_types[i]);
  }
  env->SetIntArrayRegion(jarray, 0, size, elements);
  return jarray;
  API_END_RETURN()
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_tensorrt_jni_TrtLibrary_getOutputNames(
    JNIEnv *env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto *model = reinterpret_cast<TrtModel *>(jhandle);
  std::vector<std::string> names = model->getOutputNames();
  return djl::utils::jni::GetStringArrayFromVec(env, names);
  API_END_RETURN()
}

JNIEXPORT jintArray JNICALL Java_ai_djl_tensorrt_jni_TrtLibrary_getOutputDataTypes(
    JNIEnv *env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto *model = reinterpret_cast<TrtModel *>(jhandle);
  std::vector<nvinfer1::DataType> data_types = model->getOutputTypes();
  int size = data_types.size();
  jintArray jarray = env->NewIntArray(size);
  jint elements[size];
  for (int i = 0; i < size; i++) {
    elements[i] = static_cast<int>(data_types[i]);
  }
  env->SetIntArrayRegion(jarray, 0, size, elements);
  return jarray;
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_tensorrt_jni_TrtLibrary_createSession(JNIEnv *env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto *model = reinterpret_cast<TrtModel *>(jhandle);
  auto *session = model->createSession();
  auto jsession = reinterpret_cast<uintptr_t>(session);
  return jsession;
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_tensorrt_jni_TrtLibrary_deleteSession(JNIEnv *env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto *session = reinterpret_cast<TrtSession *>(jhandle);
  delete session;
  API_END()
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_tensorrt_jni_TrtLibrary_getShape(
    JNIEnv *env, jobject jthis, jlong jhandle, jstring jname) {
  API_BEGIN()
  auto *session = reinterpret_cast<TrtSession *>(jhandle);
  const char *name = env->GetStringUTFChars(jname, JNI_FALSE);
  Dims dims = session->getShape(name);
  jlongArray jarray = env->NewLongArray(dims.nbDims);
  jlong elements[dims.nbDims];
  for (int i = 0; i < dims.nbDims; i++) {
    elements[i] = dims.d[i];
  }
  env->SetLongArrayRegion(jarray, 0, dims.nbDims, elements);
  return jarray;
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_tensorrt_jni_TrtLibrary_bind(
    JNIEnv *env, jobject jthis, jlong jhandle, jstring jname, jobject jbuffer) {
  API_BEGIN()
  auto *session = reinterpret_cast<TrtSession *>(jhandle);
  void *buffer = (void *) env->GetDirectBufferAddress(jbuffer);
  size_t size = env->GetDirectBufferCapacity(jbuffer);
  const char *input_name = env->GetStringUTFChars(jname, JNI_FALSE);
  session->bind(input_name, buffer, size);
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_tensorrt_jni_TrtLibrary_runTrtModel(JNIEnv *env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto *session = reinterpret_cast<TrtSession *>(jhandle);
  session->predict();
  API_END()
}

JNIEXPORT jint JNICALL Java_ai_djl_tensorrt_jni_TrtLibrary_getTrtVersion(JNIEnv *env, jobject jthis) {
  API_BEGIN()
  return getInferLibVersion();
  API_END_RETURN()
}
