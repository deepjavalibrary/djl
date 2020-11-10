#include "ai_djl_sentencepiece_jni_SentencePieceLibrary.h"

#include "sentencepiece_processor.h"

inline void CheckStatus(JNIEnv* env, const sentencepiece::util::Status& status) {
  jclass jexception = env->FindClass("ai/djl/engine/EngineException");
  if (!status.ok()) {
    env->ThrowNew(jexception, status.ToString().c_str());
  }
}

inline std::string GetStringFromJString(JNIEnv* env, jstring jstr) {
  if (jstr == nullptr) {
    return std::string();
  }
  const char* c_str = env->GetStringUTFChars(jstr, JNI_FALSE);
  std::string str = std::string(c_str);
  env->ReleaseStringUTFChars(jstr, c_str);
  return str;
}

// String[]
inline jobjectArray GetStringArrayFromVector(JNIEnv* env, const std::vector<std::string>& vec) {
  jobjectArray array = env->NewObjectArray(vec.size(), env->FindClass("Ljava/lang/String;"), nullptr);
  for (int i = 0; i < vec.size(); ++i) {
    env->SetObjectArrayElement(array, i, env->NewStringUTF(vec[i].c_str()));
  }
  return array;
}

// String[][]
inline jobjectArray Get2DStringArrayFrom2DVector(JNIEnv* env, const std::vector<std::vector<std::string>>& vec) {
  jobjectArray array = env->NewObjectArray(vec.size(), env->FindClass("[Ljava/lang/String;"), nullptr);
  for (int i = 0; i < vec.size(); ++i) {
    env->SetObjectArrayElement(array, i, GetStringArrayFromVector(env, vec[i]));
  }
  return array;
}

inline std::vector<std::string> GetVectorFromStringArray(JNIEnv* env, jobjectArray array) {
  std::vector<std::string> vec;
  jsize len = env->GetArrayLength(array);
  vec.reserve(len);
  for (int i = 0; i < len; ++i) {
    std::string str = GetStringFromJString(env, (jstring)env->GetObjectArrayElement(array, i));
    vec.emplace_back(str);
  }
  return std::move(vec);
}

inline jintArray GetIntArrayFromVector(JNIEnv* env, const std::vector<int>& vec) {
  jintArray array = env->NewIntArray(vec.size());
  env->SetIntArrayRegion(array, 0, vec.size(), vec.data());
  return array;
}

inline jobjectArray Get2DIntArrayFrom2DVector(JNIEnv* env, const std::vector<std::vector<int>>& vec) {
  jobjectArray array = env->NewObjectArray(vec.size(), env->FindClass("[I"), nullptr);
  for (int i = 0; i < vec.size(); ++i) {
    env->SetObjectArrayElement(array, i, GetIntArrayFromVector(env, vec[i]));
  }
  return array;
}

inline std::vector<int> GetVectorFromIntArray(JNIEnv* env, jintArray array) {
  jsize len = env->GetArrayLength(array);
  void* data = env->GetPrimitiveArrayCritical(array, JNI_FALSE);
  std::vector<int> vec((int*)data, ((int*)data) + len);
  env->ReleasePrimitiveArrayCritical(array, data, JNI_ABORT);
  return vec;
}

JNIEXPORT jlong JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_createSentencePieceProcessor(
    JNIEnv* env, jobject jthis) {
  auto* processor_ptr = new sentencepiece::SentencePieceProcessor();
  return reinterpret_cast<uintptr_t>(processor_ptr);
}

JNIEXPORT void JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_loadModel(
    JNIEnv* env, jobject jthis, jlong jhandle, jstring jpath) {
  auto* processor_ptr = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(jhandle);
  const std::string path_string = GetStringFromJString(env, jpath);
  CheckStatus(env, processor_ptr->Load(path_string));
}

JNIEXPORT void JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_deleteSentencePieceProcessor(
    JNIEnv* env, jobject jthis, jlong jhandle) {
  auto* processor_ptr = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(jhandle);
  delete processor_ptr;
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_tokenize(
    JNIEnv* env, jobject jthis, jlong jhandle, jstring jtext) {
  auto* processor_ptr = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(jhandle);
  const std::string text = GetStringFromJString(env, jtext);
  std::vector<std::string> pieces;
  CheckStatus(env, processor_ptr->Encode(text, &pieces));
  return GetStringArrayFromVector(env, pieces);
}

JNIEXPORT jintArray JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_encode(
    JNIEnv* env, jobject jthis, jlong jhandle, jstring jtext) {
  auto* processor_ptr = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(jhandle);
  const std::string text = GetStringFromJString(env, jtext);
  std::vector<int> ids;
  CheckStatus(env, processor_ptr->Encode(text, &ids));
  return GetIntArrayFromVector(env, ids);
}

JNIEXPORT jstring JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_detokenize(
    JNIEnv* env, jobject jthis, jlong jhandle, jobjectArray jtokens) {
  auto* processor_ptr = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(jhandle);
  std::vector<std::string> pieces = GetVectorFromStringArray(env, jtokens);
  std::string detokenized;
  CheckStatus(env, processor_ptr->Decode(pieces, &detokenized));
  return env->NewStringUTF(detokenized.c_str());
}

JNIEXPORT jstring JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_decode(
    JNIEnv* env, jobject jthis, jlong jhandle, jintArray jids) {
  jclass jexception = env->FindClass("ai/djl/engine/EngineException");
  auto* processor_ptr = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(jhandle);
  std::vector<int> ids = GetVectorFromIntArray(env, jids);
  std::string detokenized;
  CheckStatus(env, processor_ptr->Decode(ids, &detokenized));
  return env->NewStringUTF(detokenized.c_str());
}

JNIEXPORT jstring JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_idToPiece(
    JNIEnv* env, jobject jthis, jlong jhandle, jint jid) {
  auto* processor_ptr = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(jhandle);
  return env->NewStringUTF(processor_ptr->IdToPiece(jid).c_str());
}

JNIEXPORT int JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_pieceToId(
    JNIEnv* env, jobject jthis, jlong jhandle, jstring jpiece) {
  auto* processor_ptr = reinterpret_cast<sentencepiece::SentencePieceProcessor*>(jhandle);
  return processor_ptr->PieceToId(GetStringFromJString(env, jpiece));
}
