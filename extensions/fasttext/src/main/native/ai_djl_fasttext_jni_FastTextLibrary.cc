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
#include "ai_djl_fasttext_jni_FastTextLibrary.h"

#include <djl/utils.h>

#include <numeric>

#include "args.h"
#include "dictionary.h"
#include "fasttext.cc"
#include "main.cc"
#include "matrix.cc"
#include "matrix.h"
#include "model.cc"
#include "model.h"
#include "productquantizer.cc"
#include "quantmatrix.cc"
#include "utils.cc"
#include "vector.cc"

struct FastTextPrivateMembers {
  std::shared_ptr<fasttext::Args> args_;
  std::shared_ptr<fasttext::Dictionary> dict_;
  std::shared_ptr<fasttext::Matrix> input_;
  std::shared_ptr<fasttext::Matrix> output_;
  std::shared_ptr<fasttext::Model> model_;
};

JNIEXPORT jlong JNICALL Java_ai_djl_fasttext_jni_FastTextLibrary_createFastText(JNIEnv* env, jobject jthis) {
  auto* fasttext_ptr = new fasttext::FastText();
  return reinterpret_cast<uintptr_t>(fasttext_ptr);
}

JNIEXPORT void JNICALL Java_ai_djl_fasttext_jni_FastTextLibrary_freeFastText(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  auto* fasttext_ptr = reinterpret_cast<fasttext::FastText*>(jhandle);
  delete fasttext_ptr;
}

JNIEXPORT void JNICALL Java_ai_djl_fasttext_jni_FastTextLibrary_loadModel(
    JNIEnv* env, jobject jthis, jlong jhandle, jstring jpath) {
  auto* fasttext_ptr = reinterpret_cast<fasttext::FastText*>(jhandle);
  const std::string path_string = djl::utils::jni::GetStringFromJString(env, jpath);
  try {
    fasttext_ptr->loadModel(path_string);
  } catch (const std::invalid_argument& e) {
    jclass jexception = env->FindClass("ai/djl/engine/EngineException");
    env->ThrowNew(jexception, e.what());
  }
}

JNIEXPORT jboolean JNICALL Java_ai_djl_fasttext_jni_FastTextLibrary_checkModel(
    JNIEnv* env, jobject jthis, jstring jpath) {
  const std::string filename = djl::utils::jni::GetStringFromJString(env, jpath);
  std::ifstream in(filename, std::ifstream::binary);
  int32_t magic;
  int32_t version;
  in.read((char*) &(magic), sizeof(int32_t));
  if (magic != fasttext::FASTTEXT_FILEFORMAT_MAGIC_INT32) {
    return false;
  }
  in.read((char*) &(version), sizeof(int32_t));
  if (version != fasttext::FASTTEXT_VERSION) {
    return false;
  }
  return true;
}

JNIEXPORT void JNICALL Java_ai_djl_fasttext_jni_FastTextLibrary_unloadModel(JNIEnv* env, jobject jthis, jlong jhandle) {
  auto* fasttext_ptr = reinterpret_cast<fasttext::FastText*>(jhandle);

  FastTextPrivateMembers* privateMembers = (FastTextPrivateMembers*) fasttext_ptr;
  privateMembers->args_.reset();
  privateMembers->dict_.reset();
  privateMembers->input_.reset();
  privateMembers->output_.reset();
  privateMembers->model_.reset();
}

JNIEXPORT jstring JNICALL Java_ai_djl_fasttext_jni_FastTextLibrary_getModelType(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  auto* fasttext_ptr = reinterpret_cast<fasttext::FastText*>(jhandle);

  auto* privateMembers = (FastTextPrivateMembers*) fasttext_ptr;
  model_name modelName = privateMembers->args_->model;
  if (modelName == model_name::cbow) {
    return env->NewStringUTF("cbow");
  } else if (modelName == model_name::sg) {
    return env->NewStringUTF("sg");
  } else if (modelName == model_name::sup) {
    return env->NewStringUTF("sup");
  } else {
    jclass jexception = env->FindClass("ai/djl/engine/EngineException");
    env->ThrowNew(jexception, "Unrecognized model type");
    return nullptr;
  }
}

JNIEXPORT jint JNICALL Java_ai_djl_fasttext_jni_FastTextLibrary_predictProba(
    JNIEnv* env, jobject jthis, jlong jhandle, jstring jtext, jint top_k, jobject jclasses, jobject jprob) {
  auto* fasttext_ptr = reinterpret_cast<fasttext::FastText*>(jhandle);
  std::string text = djl::utils::jni::GetStringFromJString(env, jtext);
  std::istringstream in(text);
  std::vector<std::pair<real, std::string>> predictions;
  fasttext_ptr->predictLine(in, predictions, top_k, 0.0);

  int size = predictions.size();
  jclass java_lang_Float = static_cast<jclass>(env->NewGlobalRef(env->FindClass("java/lang/Float")));
  jmethodID java_lang_Float_ = env->GetMethodID(java_lang_Float, "<init>", "(F)V");
  jclass java_util_ArrayList = static_cast<jclass>(env->NewGlobalRef(env->FindClass("java/util/ArrayList")));
  jmethodID java_util_ArrayList_add = env->GetMethodID(java_util_ArrayList, "add", "(Ljava/lang/Object;)Z");
  for (int i = 0; i < size; ++i) {
    std::pair<real, std::string> pair = predictions[i];
    env->CallBooleanMethod(jclasses, java_util_ArrayList_add, env->NewStringUTF(pair.second.c_str()));
    env->CallBooleanMethod(
        jprob, java_util_ArrayList_add, env->NewObject(java_lang_Float, java_lang_Float_, pair.first));
  }

  return size;
}

JNIEXPORT jfloatArray JNICALL Java_ai_djl_fasttext_jni_FastTextLibrary_getWordVector(
    JNIEnv* env, jobject jthis, jlong jhandle, jstring word) {
  std::string word_str = djl::utils::jni::GetStringFromJString(env, word);
  auto* fasttext_ptr = reinterpret_cast<fasttext::FastText*>(jhandle);
  auto* privateMembers = (FastTextPrivateMembers*) fasttext_ptr;

  Vector vec(privateMembers->args_->dim);
  fasttext_ptr->getWordVector(vec, word_str);

  jfloatArray array = env->NewFloatArray(vec.size());
  env->SetFloatArrayRegion(array, 0, vec.size(), vec.data());
  return array;
}

JNIEXPORT int JNICALL Java_ai_djl_fasttext_jni_FastTextLibrary_runCmd(JNIEnv* env, jobject jthis, jobjectArray args) {
  std::vector<std::string> vec = djl::utils::jni::GetVecFromJStringArray(env, args);
  if (vec.size() < 2) {
    printUsage();
    return -1;
  }
  std::string command(vec[1]);
  if (command == "skipgram" || command == "cbow" || command == "supervised") {
    train(vec);
  } else if (command == "test" || command == "test-label") {
    test(vec);
  } else if (command == "quantize") {
    quantize(vec);
  } else if (command == "print-word-vectors") {
    printWordVectors(vec);
  } else if (command == "print-sentence-vectors") {
    printSentenceVectors(vec);
  } else if (command == "print-ngrams") {
    printNgrams(vec);
  } else if (command == "nn") {
    nn(vec);
  } else if (command == "analogies") {
    analogies(vec);
  } else if (command == "predict" || command == "predict-prob") {
    predict(vec);
  } else if (command == "dump") {
    dump(vec);
  } else {
    printUsage();
    return -1;
  }
  return 0;
}
