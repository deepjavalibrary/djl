#include "c_api.h"

using namespace at;

int map_to_int_dtype(caffe2::TypeMeta dtype) {
  if (dtype == torch::kFloat) {
    return 0;
  } else if (dtype == torch::kFloat32) {
    return 0;
  } else if (dtype == torch::kFloat64) {
    return 1;
  } else if (dtype == torch::kFloat16) {
    return 2;
  } else if (dtype == torch::kUInt8) {
    return 3;
  } else if (dtype == torch::kInt32) {
    return 4;
  } else if (dtype == torch::kInt8) {
    return 5;
  } else if (dtype == torch::kInt64) {
    return 6;
  } else if (dtype == torch::kBool) {
    return 7;
  }
  return -1;
}

c10::ScalarType map_to_torch_dtype(int dtype) {
  switch (dtype) {
    case 0:
      return torch::kFloat32;
    case 1:
      return torch::kFloat64;
    case 2:
      return torch::kFloat16;
    case 3:
      return torch::kUInt8;
    case 4:
      return torch::kInt32;
    case 5:
      return torch::kInt8;
    case 6:
      return torch::kInt64;
    case 7:
      return torch::kBool;
    default:
      throw;
  }
}

int ones(const int64_t *shape, const int dim, TensorHandle* output) {
  const std::vector<int64_t> shape_vec(shape, shape + dim);
  *output = new Tensor(torch::ones(shape_vec));
  return 0;
}

int TensorToFloat(TensorHandle handle, const void **data, size_t* size) {
  const auto tensor_ptr = static_cast<at::Tensor*>(handle);
  *size = tensor_ptr->numel();
  *data = tensor_ptr->data_ptr<float>();
  return 0;
}

int TensorGetDType(TensorHandle handle, int* size) {
  const auto tensor_ptr = static_cast<at::Tensor*>(handle);
  *size = map_to_int_dtype(tensor_ptr->dtype());
  return 0;
}

int TensorCreate(const int64_t *shape, const int dim, int dtype, int layout, int device, TensorHandle *out_handle) {
  auto options = torch::TensorOptions().dtype(map_to_torch_dtype(dtype)).requires_grad(false);
  const std::vector<int64_t> shape_vec(shape, shape + dim);
  *out_handle = new Tensor(torch::empty(shape_vec, options));
  return 0;
}

int TensorGetShape(TensorHandle handle, int *out_dim, const int64_t **out_data) {
    const auto tensor_ptr = static_cast<at::Tensor *>(handle);
    *out_dim = tensor_ptr->dim();
    auto *shape = new std::vector<int64_t>;
    for (auto i = 0; i < *out_dim; ++i) {
        shape->emplace_back(tensor_ptr->size(i));
    }
    *out_data = shape->data();
    return 0;
}

int ModuleLoad(const char* path, ModuleHandle* moduleHandle) {
    //TODO: add device support
    std::string pathString(path);
    torch::jit::script::Module module = torch::jit::load(pathString);
    *moduleHandle = new torch::jit::script::Module(module);
    return 0;
}

/**
 * enable evaluation mode, Module enable training mode by default.
 * @param module the Module handle
 * @return status code
 */
int ModuleEval(ModuleHandle moduleHandle) {
    static_cast<torch::jit::script::Module*>(moduleHandle)->eval();
    return 0;
}

int ModuleForward(ModuleHandle moduleHandle, IValueArrayHandle iValueArrayHandle, int length, IValueHandle* resultHandle) {
    //TODO: This haven't been test the case with multiple inputs
    auto module = static_cast<torch::jit::script::Module*>(moduleHandle);
    auto iValueArray = std::vector<c10::IValue>();
    for (int i = 0; i < length; i++) {
        iValueArray.emplace_back(*static_cast<c10::IValue*>(*(iValueArrayHandle + i)));
    }
    *resultHandle = new c10::IValue(module->forward(iValueArray));
    return 0;
}

int IValueCreateFromTensor(TensorHandle tensorHandle, IValueHandle* iValueHandle) {
    *iValueHandle = new at::IValue(*static_cast<torch::Tensor*>(tensorHandle));
    return 0;
}

int IValueToTensor(IValueHandle iValueHandle, TensorHandle* tensorHandle) {
    *tensorHandle = new torch::Tensor(static_cast<c10::IValue*>(iValueHandle)->toTensor());
    return 0;
}
