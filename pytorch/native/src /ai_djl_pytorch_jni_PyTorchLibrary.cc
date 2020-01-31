#include "../build/include/ai_djl_pytorch_jni_PyTorchLibrary.h"
#include <torch/torch.h>

static constexpr const char* const POINTER_CLASS = "ai/djl/pytorch/jni/Pointer";

inline const at::Tensor* GetTensorFromHandle(JNIEnv *env, jobject jhandle) {
  jclass cls = env->GetObjectClass(jhandle);
  jmethodID get_value = env->GetMethodID(cls, "getValue", "()J");
  if (nullptr == get_value) {
    std::cout << "getValue method not found!" << std::endl;
    return nullptr;
  }
  jlong peer = env->CallLongMethod(jhandle, get_value);
  return reinterpret_cast<const at::Tensor*>(peer);
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_atOnes
  (JNIEnv *env, jobject this_object, jlongArray jshape) {
  jclass cls = env->FindClass(POINTER_CLASS);
  if (nullptr == cls) {
    std::cout << "class not found!" << std::endl;
    return nullptr;
  }
  jmethodID init = env->GetMethodID(cls, "<init>", "(J)V");
  if (nullptr == init) {
    std::cout << "method not found!" << std::endl;
    return nullptr;
  }
  jlong *shape = env->GetLongArrayElements(jshape, JNI_FALSE);
  jsize length = env->GetArrayLength(jshape);

  std::vector<int64_t> shape_vec(shape, shape + length);
  const at::Tensor* tensor_ptr = new at::Tensor(at::ones(shape_vec));
  jobject new_obj = env->NewObject(cls, init, tensor_ptr);
  if (nullptr == new_obj) {
    std::cout << "object created failed" << std::endl;
    return nullptr;
  }
  return new_obj;
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_atSizes
  (JNIEnv *env, jobject this_object, jobject jhandle) {
  const at::Tensor* tensor_ptr = GetTensorFromHandle(env, jhandle);
  jlongArray size = env->NewLongArray(tensor_ptr->dim());
  env->SetLongArrayRegion(size, 0, tensor_ptr->dim(), reinterpret_cast<const jlong*>(tensor_ptr->sizes().data()));
  return size;
}

JNIEXPORT jobject JNICALL Java_ai_djl_pytorch_jni_PyTorchLibrary_atDataPtr
(JNIEnv *env, jobject this_object, jobject jhandle) {
  const at::Tensor* tensor_ptr = GetTensorFromHandle(env, jhandle);
  jobject buf = env->NewDirectByteBuffer(tensor_ptr->data_ptr(), tensor_ptr->nbytes());
  return buf;
}