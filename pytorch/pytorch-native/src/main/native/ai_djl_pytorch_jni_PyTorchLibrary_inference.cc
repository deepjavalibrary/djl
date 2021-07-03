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
#include <torch/csrc/jit/python/update_graph_executor_opt.h>
#include <torch/script.h>

#include "ai_djl_pytorch_jni_PyTorchLibrary.h"
#include "djl_pytorch_jni_exception.h"
#include "djl_pytorch_utils.h"

// The file is the implementation for PyTorch inference operations

struct JITCallGuard {
  torch::autograd::AutoGradMode no_autograd_guard{false};
  torch::NoGradGuard no_grad;
};

JNIEXPORT jlong JNICALL
Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleLoad__Ljava_lang_String_2_3I_3Ljava_lang_String_2_3Ljava_lang_String_2(
    JNIEnv* env, jobject jthis, jstring jpath, jintArray jarray, jobjectArray jefnames, jobjectArray jefvalues) {
  API_BEGIN()
  const std::string path = djl::utils::jni::GetStringFromJString(env, jpath);
  const torch::Device device = utils::GetDeviceFromJDevice(env, jarray);
  std::unordered_map<std::string, std::string> map;
  size_t len = static_cast<size_t>(env->GetArrayLength(jefnames));
  for (size_t i = 0; i < len; ++i) {
    auto jname = (jstring) env->GetObjectArrayElement(jefnames, i);
    auto name = djl::utils::jni::GetStringFromJString(env, jname);
    map[name] = "";
  }
  const torch::jit::script::Module module = torch::jit::load(path, device, map);
  const auto* module_ptr = new torch::jit::script::Module(module);
  for (size_t i = 0; i < len; ++i) {
    auto jname = (jstring) env->GetObjectArrayElement(jefnames, i);
    auto name = djl::utils::jni::GetStringFromJString(env, jname);
    env->SetObjectArrayElement(jefvalues, i, env->NewStringUTF(map[name].c_str()));
  }
  return reinterpret_cast<uintptr_t>(module_ptr);
  API_END_RETURN()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleLoad__Ljava_io_InputStream_2_3I_3BJ(
    JNIEnv* env, jobject jthis, jobject jis, jintArray jarray, jbyteArray arr, jlong size) {
  API_BEGIN()
  jclass is_class = env->GetObjectClass(jis);
  if (is_class == nullptr) {
    env->ThrowNew(NULL_PTR_EXCEPTION_CLASS, "Java inputStream class is not found");
    return -1;
  }
  jmethodID method_id = env->GetMethodID(is_class, "read", "([BII)I");
  if (method_id == nullptr) {
    env->ThrowNew(ENGINE_EXCEPTION_CLASS, "The read method in InputStream is not found");
    return -1;
  }
  std::ostringstream os;
  int len = env->GetArrayLength(arr);
  jbyte* data;
  if (size != -1) {
    for (; size > 0; size -= len) {
      if (size < len) {
        len = size;
      }
      env->CallIntMethod(jis, method_id, arr, 0, len);
      data = env->GetByteArrayElements(arr, JNI_FALSE);
      os.write(reinterpret_cast<char*>(data), len);
      env->ReleaseByteArrayElements(arr, data, JNI_ABORT);
    }
  } else {
    int available = 0;
    while (available != -1) {
      available = env->CallIntMethod(jis, method_id, arr, 0, len);
      if (available != -1) {
        data = env->GetByteArrayElements(arr, JNI_FALSE);
        os.write(reinterpret_cast<char*>(data), available);
        env->ReleaseByteArrayElements(arr, data, JNI_ABORT);
      }
    }
  }

  std::istringstream in(os.str());
  const torch::Device device = utils::GetDeviceFromJDevice(env, jarray);
  const torch::jit::script::Module module = torch::jit::load(in, device);
  const auto* module_ptr = new torch::jit::script::Module(module);
  return reinterpret_cast<uintptr_t>(module_ptr);
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleWrite(
    JNIEnv* env, jobject jthis, jlong module_handle, jobject jos, jbyteArray arr, jboolean jwrite_size) {
  API_BEGIN()
  auto* module_ptr = reinterpret_cast<torch::jit::script::Module*>(module_handle);
#if defined(__ANDROID__)
  env->ThrowNew(ENGINE_EXCEPTION_CLASS, "This kind of mode is not supported on Android");
  return;
#endif
  std::ostringstream stream;
  module_ptr->save(stream);
  auto str = stream.str();
  jclass os_class = env->GetObjectClass(jos);
  if (os_class == nullptr) {
    env->ThrowNew(NULL_PTR_EXCEPTION_CLASS, "Java OutputStream class is not found");
    return;
  }
  jmethodID method_id = env->GetMethodID(os_class, "write", "([BII)V");
  if (method_id == nullptr) {
    env->ThrowNew(ENGINE_EXCEPTION_CLASS, "The write method in OutputStream is not found");
    return;
  }
  if (jwrite_size) {
    auto jbytes = env->NewByteArray(8);
    int64_t length = str.size();
    char bytes[8];
    for (int i = 0; i < 8; i++) {
      bytes[i] = static_cast<int>(length >> (56 - 8 * i) & 0XFF);
    }
    env->SetByteArrayRegion(jbytes, 0, 8, (jbyte*) bytes);
    env->CallVoidMethod(jos, method_id, jbytes, 0, 8);
  }
  int len = env->GetArrayLength(arr);
  int i = 0;
  for (; i + len < str.length(); i += len) {
    auto substr = str.substr(i, i + len);
    env->SetByteArrayRegion(arr, 0, len, (jbyte*) substr.c_str());
    env->CallVoidMethod(jos, method_id, arr, 0, len);
  }
  auto last_len = str.length() - i;
  if (last_len > 0) {
    auto substr = str.substr(i, last_len);
    env->SetByteArrayRegion(arr, 0, last_len, (jbyte*) substr.c_str());
    env->CallVoidMethod(jos, method_id, arr, 0, last_len);
  }
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_setGraphExecutorOptimize(
    JNIEnv* env, jobject jthis, jboolean jenabled) {
  API_BEGIN()
  torch::jit::setGraphExecutorOptimize(jenabled);
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleEval(
    JNIEnv* env, jobject jthis, jlong module_handle) {
  API_BEGIN()
  auto* module_ptr = reinterpret_cast<torch::jit::script::Module*>(module_handle);
  module_ptr->eval();
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleTrain(
    JNIEnv* env, jobject jthis, jlong module_handle) {
  API_BEGIN()
  auto* module_ptr = reinterpret_cast<torch::jit::script::Module*>(module_handle);
  module_ptr->train(true);
  API_END()
}

JNIEXPORT jlong JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleForward(
    JNIEnv* env, jobject jthis, jlong module_handle, jlongArray jivalue_ptrs, jboolean jis_train) {
  API_BEGIN()
  auto* module_ptr = reinterpret_cast<torch::jit::script::Module*>(module_handle);
  size_t len = env->GetArrayLength(jivalue_ptrs);
  jlong* jptrs = env->GetLongArrayElements(jivalue_ptrs, JNI_FALSE);
  std::vector<torch::IValue> inputs;
  inputs.reserve(len);
  for (auto i = 0; i < len; ++i) {
    inputs.emplace_back(*reinterpret_cast<torch::IValue*>(jptrs[i]));
  }
  torch::IValue output = [&]() {
    if (jis_train) {
      return module_ptr->forward(inputs);
    }
    // disable autograd
    JITCallGuard guard;
    return module_ptr->forward(inputs);
  }();
  // release resource
  // each IValue is created by new, free the memory after the inference
  for (auto i = 0; i < len; ++i) {
    delete reinterpret_cast<torch::IValue*>(jptrs[i]);
  }
  env->ReleaseLongArrayElements(jivalue_ptrs, jptrs, djl::utils::jni::RELEASE_MODE);
  const auto* result_ptr = new torch::IValue(output);
  return reinterpret_cast<uintptr_t>(result_ptr);
  API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_torchDeleteModule(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* module_ptr = reinterpret_cast<torch::jit::script::Module*>(jhandle);
  delete module_ptr;
  API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleSave(
    JNIEnv* env, jobject jthis, jlong jhandle, jstring jpath) {
  API_BEGIN()
  auto* module_ptr = reinterpret_cast<torch::jit::script::Module*>(jhandle);
#if defined(__ANDROID__)
  env->ThrowNew(ENGINE_EXCEPTION_CLASS, "This kind of mode is not supported on Android");
  return;
#endif
  module_ptr->save(djl::utils::jni::GetStringFromJString(env, jpath));
  API_END()
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleGetParams(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* module_ptr = reinterpret_cast<torch::jit::script::Module*>(jhandle);
  std::vector<jlong> jptrs;
  for (const auto& tensor : module_ptr->parameters()) {
    jptrs.push_back(reinterpret_cast<uintptr_t>(new torch::Tensor(tensor)));
  }
  size_t len = jptrs.size();
  jlongArray jarray = env->NewLongArray(len);
  env->SetLongArrayRegion(jarray, 0, len, jptrs.data());
  return jarray;
  API_END_RETURN()
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleGetParamNames(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  API_BEGIN()
  auto* module_ptr = reinterpret_cast<torch::jit::script::Module*>(jhandle);
  std::vector<std::string> jptrs;
  for (const auto& named_tensor : module_ptr->named_parameters()) {
    jptrs.push_back(named_tensor.name);
  }
  return djl::utils::jni::GetStringArrayFromVec(env, jptrs);
  API_END_RETURN()
}
