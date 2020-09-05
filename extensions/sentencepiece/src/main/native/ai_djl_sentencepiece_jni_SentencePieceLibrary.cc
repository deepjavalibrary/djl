#include "ai_djl_sentencepiece_jni_SentencePieceLibrary.h"

#include "sentencepiece_processor.h"

static constexpr const char* const POINTER_CLASS = "ai/djl/sentencepiece/jni/Pointer";

template <typename T>
inline T* GetPointerFromJHandle(JNIEnv* env, jobject jhandle) {
  jclass jexception = env->FindClass("java/lang/NullPointerException");
  jclass cls = env->FindClass(POINTER_CLASS);
  jmethodID get_value = env->GetMethodID(cls, "getValue", "()J");
  if (get_value == nullptr) {
    env->ThrowNew(jexception, "getValue method not found!");
  }
  jlong ptr = env->CallLongMethod(jhandle, get_value);
  return reinterpret_cast<T*>(ptr);
}

template <typename T>
inline jobject CreatePointer(JNIEnv* env, const T* ptr) {
  jclass jexception = env->FindClass("java/lang/NullPointerException");
  jclass cls = env->FindClass(POINTER_CLASS);
  if (cls == nullptr) {
    env->ThrowNew(jexception, "Pointer class not found!");
  }
  jmethodID init = env->GetMethodID(cls, "<init>", "(J)V");
  jobject new_obj = env->NewObject(cls, init, ptr);
  if (new_obj == nullptr) {
    env->ThrowNew(jexception, "object created failed");
  }
  env->DeleteLocalRef(jexception);
  env->DeleteLocalRef(cls);
  return new_obj;
}

inline std::string jstringToString(JNIEnv* env, jstring array) {
  jsize len = env->GetStringUTFLength(array);

  const char* str = env->GetStringUTFChars(array, nullptr);
  std::string s(str, len);
  env->ReleaseStringUTFChars(array, str);

  return s;
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

inline void GetVectorFromStringArray(JNIEnv* env, jobjectArray array, std::vector<std::string>* vec) {
  jsize len = env->GetArrayLength(array);
  vec->resize(len);
  for (int i = 0; i < len; ++i) {
    std::string stdStr = jstringToString(env, (jstring)env->GetObjectArrayElement(array, i));
    (*vec)[i] = stdStr;
  }
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

  void* data = env->GetPrimitiveArrayCritical(array, nullptr);
  std::vector<int> vec((int*)data, ((int*)data) + len);
  env->ReleasePrimitiveArrayCritical(array, data, JNI_ABORT);

  return vec;
}

JNIEXPORT jobject JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_createSentencePieceProcessor(
    JNIEnv* env, jobject jthis) {
  auto* processor_ptr = new sentencepiece::SentencePieceProcessor();
  return CreatePointer<sentencepiece::SentencePieceProcessor>(env, processor_ptr);
}

JNIEXPORT void JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_loadModel(
    JNIEnv* env, jobject jthis, jobject jhandle, jstring jpath) {
  jclass jexception = env->FindClass("ai/djl/engine/EngineException");
  auto* processor_ptr = GetPointerFromJHandle<sentencepiece::SentencePieceProcessor>(env, jhandle);
  const std::string path_string = jstringToString(env, jpath);
  const auto status = processor_ptr->Load(path_string);
  if (!status.ok()) {
    env->ThrowNew(jexception, status.ToString().c_str());
  }
}

JNIEXPORT void JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_deleteSentencePieceProcessor(
    JNIEnv* env, jobject jthis, jobject jhandle) {
  auto* processor_ptr = GetPointerFromJHandle<sentencepiece::SentencePieceProcessor>(env, jhandle);
  delete processor_ptr;
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_tokenize(
    JNIEnv* env, jobject jthis, jobject jhandle, jstring jtext) {
  jclass jexception = env->FindClass("ai/djl/engine/EngineException");
  auto* processor_ptr = GetPointerFromJHandle<sentencepiece::SentencePieceProcessor>(env, jhandle);
  const std::string text = jstringToString(env, jtext);
  std::vector<std::string> pieces;
  const auto status = processor_ptr->Encode(text, &pieces);
  if (!status.ok()) {
    env->ThrowNew(jexception, status.ToString().c_str());
  }
  return GetStringArrayFromVector(env, pieces);
}

JNIEXPORT jintArray JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_encode(
    JNIEnv* env, jobject jthis, jobject jhandle, jstring jtext) {
  jclass jexception = env->FindClass("ai/djl/engine/EngineException");
  auto* processor_ptr = GetPointerFromJHandle<sentencepiece::SentencePieceProcessor>(env, jhandle);
  const std::string text = jstringToString(env, jtext);
  std::vector<int> ids;
  const auto status = processor_ptr->Encode(text, &ids);
  if (!status.ok()) {
    env->ThrowNew(jexception, status.ToString().c_str());
  }
  return GetIntArrayFromVector(env, ids);
}

JNIEXPORT jstring JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_detokenize(
    JNIEnv* env, jobject jthis, jobject jhandle, jobjectArray jtokens) {
  jclass jexception = env->FindClass("ai/djl/engine/EngineException");
  auto* processor_ptr = GetPointerFromJHandle<sentencepiece::SentencePieceProcessor>(env, jhandle);
  std::vector<std::string> pieces;
  GetVectorFromStringArray(env, jtokens, &pieces);
  std::string detokenized;
  const auto status = processor_ptr->Decode(pieces, &detokenized);
  if (!status.ok()) {
    env->ThrowNew(jexception, status.ToString().c_str());
  }
  return env->NewStringUTF(detokenized.c_str());
}

JNIEXPORT jstring JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_decode(
    JNIEnv* env, jobject jthis, jobject jhandle, jintArray jids) {
  jclass jexception = env->FindClass("ai/djl/engine/EngineException");
  auto* processor_ptr = GetPointerFromJHandle<sentencepiece::SentencePieceProcessor>(env, jhandle);
  std::vector<int> ids = GetVectorFromIntArray(env, jids);
  std::string detokenized;
  const auto status = processor_ptr->Decode(ids, &detokenized);
  if (!status.ok()) {
    env->ThrowNew(jexception, status.ToString().c_str());
  }
  return env->NewStringUTF(detokenized.c_str());
}

JNIEXPORT jstring JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_idToPiece(
    JNIEnv* env, jobject jthis, jobject jhandle, jint jid) {
  auto* processor_ptr = GetPointerFromJHandle<sentencepiece::SentencePieceProcessor>(env, jhandle);
  return env->NewStringUTF(processor_ptr->IdToPiece(jid).c_str());
}

JNIEXPORT int JNICALL Java_ai_djl_sentencepiece_jni_SentencePieceLibrary_pieceToId(
    JNIEnv* env, jobject jthis, jobject jhandle, jstring jpiece) {
  auto* processor_ptr = GetPointerFromJHandle<sentencepiece::SentencePieceProcessor>(env, jhandle);
  return processor_ptr->PieceToId(jstringToString(env, jpiece));
}
