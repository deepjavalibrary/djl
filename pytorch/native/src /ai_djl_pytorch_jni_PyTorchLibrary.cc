#include "../build/include/ai_djl_pytorch_jni_PyTorchLibrary.h"
#include <torch/torch.h>
#include <torch/script.h>

static constexpr const char *const POINTER_CLASS = "ai/djl/pytorch/jni/Pointer";

template<typename T>
inline T* GetPointerFromHandle(JNIEnv* env, jobject jhandle) {
  jclass cls = env->GetObjectClass(jhandle);
  jmethodID get_value = env->GetMethodID(cls, "getValue", "()J");
  if (nullptr == get_value) {
    std::cout << "getValue method not found!" << std::endl;
  }
  jlong ptr = env->CallLongMethod(jhandle, get_value);
  return reinterpret_cast<T*>(ptr);
}

inline jobject CreatePointer(JNIEnv* env, const void* ptr) {
  jclass cls = env->FindClass(POINTER_CLASS);
  if (nullptr == cls) {
    std::cout << "Pointer class not found!" << std::endl;
    return nullptr;
  }
  jmethodID init = env->GetMethodID(cls, "<init>", "(J)V");
  jobject new_obj = env->NewObject(cls, init, ptr);
  if (nullptr == new_obj) {
    std::cout << "object created failed" << std::endl;
    return nullptr;
  }
  return new_obj;
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_atOnes
  (JNIEnv* env, jobject this_object, jlongArray jshape) {
  jlong* shape = env->GetLongArrayElements(jshape, JNI_FALSE);
  jsize length = env->GetArrayLength(jshape);

  std::vector<int64_t> shape_vec(shape, shape + length);
  const void* tensor_ptr = new torch::Tensor(torch::ones(shape_vec));
  return CreatePointer(env, tensor_ptr);
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_atSizes
  (JNIEnv* env, jobject this_object, jobject jhandle) {
  const auto* tensor_ptr = GetPointerFromHandle<torch::Tensor>(env, jhandle);
  jlongArray size = env->NewLongArray(tensor_ptr->dim());
  env->SetLongArrayRegion(size, 0, tensor_ptr->dim(),
    reinterpret_cast<const jlong *>(tensor_ptr->sizes().data()));
  return size;
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_atDataPtr
  (JNIEnv* env, jobject this_object, jobject jhandle) {
  const auto* tensor_ptr = GetPointerFromHandle<torch::Tensor>(env, jhandle);
  jobject buf = env->NewDirectByteBuffer(tensor_ptr->data_ptr(), tensor_ptr->nbytes());
  return buf;
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleLoad
  (JNIEnv* env, jobject this_object, jstring path) {
  std::string path_string((env)->GetStringUTFChars(path, JNI_FALSE));
  torch::jit::script::Module module = torch::jit::load(path_string);
  const void* module_handle = new torch::jit::script::Module(module);
  return CreatePointer(env, module_handle);
}

JNIEXPORT void JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleEval
  (JNIEnv* env, jobject this_object, jobject module_handle) {
  auto* module = GetPointerFromHandle<torch::jit::script::Module>(env, module_handle);
  module->eval();
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_moduleForward
  (JNIEnv* env, jobject this_object, jobject module_handle, jobjectArray ivalue_handle_array) {
  auto ivalue_array = std::vector<c10::IValue>();
  for (int i = 0; i < env->GetArrayLength(ivalue_handle_array); ++i) {
    auto ivalue = GetPointerFromHandle<c10::IValue>(env, env->GetObjectArrayElement(ivalue_handle_array, i));
    ivalue_array.emplace_back(*ivalue);
  }
  auto* module = GetPointerFromHandle<torch::jit::script::Module>(env, module_handle);
  void* result_handle = new c10::IValue(module->forward(ivalue_array));
  return CreatePointer(env, result_handle);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueCreateFromTensor
  (JNIEnv* env, jobject this_object, jobject tensor_handle) {
  void* ivalue_handle = new at::IValue(
    *GetPointerFromHandle<torch::Tensor>(env, tensor_handle)));
  return CreatePointer(env, ivalue_handle);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_iValueToTensor
  (JNIEnv* env, jobject this_object, jobject ivalue_handle) {
  void* tensor_handle = new torch::Tensor(
    GetPointerFromHandle<c10::IValue>(env, ivalue_handle)->toTensor());
  return CreatePointer(env, tensor_handle);
}
