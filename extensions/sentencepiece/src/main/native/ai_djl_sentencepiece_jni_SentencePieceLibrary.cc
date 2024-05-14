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

#include "ai_djl_sentencepiece_jni_SentencePieceLibrary.h"

#include <djl/utils.h>

#include "sentencepiece_processor.h"

inline void CheckStatus(JNIEnv* env, const sentencepiece::util::Status& status) {
  jclass jexception = env->FindClass("ai/djl/engine/EngineException");
  if (!status.ok()) {
    env->ThrowNew(jexception, status.ToString().c_str());
  }
}

JNIEXPORT jlong JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_createSentencePieceProcessor(
    JNIEnv* env, jobject jthis) {
  auto* processor_ptr = new sentencepiece::SentencePieceProcessor();
  return reinterpret_cast<uintptr_t>(processor_ptr);
}

JNIEXPORT void JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_loadModel(
    JNIEnv* env, jobject jthis, jlong jhandle, jstring jpath) {
  auto* processor_ptr = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(jhandle);
  const std::string path_string = djl::utils::jni::GetStringFromJString(env, jpath);
  CheckStatus(env, processor_ptr->Load(path_string));
}

JNIEXPORT void JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_loadModelFromBytes(
    JNIEnv* env, jobject jthis, jlong jhandle, jbyteArray jserialized) {
  auto* processor_ptr = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(jhandle);
  int length = env->GetArrayLength(jserialized);
  std::vector<char> buff(length, 0);
  env->GetByteArrayRegion(jserialized, 0, length, reinterpret_cast<jbyte*>(buff.data()));
  std::string serialized(buff.data(), buff.size());
  CheckStatus(env, processor_ptr->LoadFromSerializedProto(serialized));
}

JNIEXPORT void JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_deleteSentencePieceProcessor(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  auto* processor_ptr = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(jhandle);
  delete processor_ptr;
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_tokenize(
    JNIEnv* env, jobject jthis, jlong jhandle, jstring jtext) {
  auto* processor_ptr = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(jhandle);
  const std::string text = djl::utils::jni::GetStringFromJString(env, jtext);
  std::vector<std::string> pieces;
  CheckStatus(env, processor_ptr->Encode(text, &pieces));
  return djl::utils::jni::GetStringArrayFromVec(env, pieces);
}

JNIEXPORT jintArray JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_encode(
    JNIEnv* env, jobject jthis, jlong jhandle, jstring jtext) {
  auto* processor_ptr = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(jhandle);
  const std::string text = djl::utils::jni::GetStringFromJString(env, jtext);
  std::vector<int> ids;
  CheckStatus(env, processor_ptr->Encode(text, &ids));
  return djl::utils::jni::GetIntArrayFromVec(env, ids);
}

JNIEXPORT jstring JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_detokenize(
    JNIEnv* env, jobject jthis, jlong jhandle, jobjectArray jtokens) {
  auto* processor_ptr = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(jhandle);
  std::vector<std::string> pieces = djl::utils::jni::GetVecFromJStringArray(env, jtokens);
  std::string detokenized;
  CheckStatus(env, processor_ptr->Decode(pieces, &detokenized));
  return env->NewStringUTF(detokenized.c_str());
}

JNIEXPORT jstring JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_decode(
    JNIEnv* env, jobject jthis, jlong jhandle, jintArray jids) {
  jclass jexception = env->FindClass("ai/djl/engine/EngineException");
  auto* processor_ptr = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(jhandle);
  std::vector<int> ids = djl::utils::jni::GetVecFromJIntArray(env, jids);
  std::string detokenized;
  CheckStatus(env, processor_ptr->Decode(ids, &detokenized));
  return env->NewStringUTF(detokenized.c_str());
}

JNIEXPORT jstring JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_idToPiece(
    JNIEnv* env, jobject jthis, jlong jhandle, jint jid) {
  auto* processor_ptr = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(jhandle);
  return env->NewStringUTF(processor_ptr->IdToPiece(jid).c_str());
}

JNIEXPORT jint JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_pieceToId(
    JNIEnv* env, jobject jthis, jlong jhandle, jstring jpiece) {
  auto* processor_ptr = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(jhandle);
  return processor_ptr->PieceToId(djl::utils::jni::GetStringFromJString(env, jpiece));
}
