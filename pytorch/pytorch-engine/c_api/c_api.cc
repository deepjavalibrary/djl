#include "c_api.h"

using namespace at;

int ones(TensorHandle* output) {
  *output = new Tensor(torch::ones({1, 3, 224, 224}));
  return 0;
}

int TensorToFloat(TensorHandle input, const void **data, size_t* size) {
  const auto tensor_ptr = static_cast<at::Tensor*>(input);
  *size = tensor_ptr->numel();
  *data = tensor_ptr->data_ptr<float>();
  return 0;
}

int TensorGetShape(TensorHandle input, int *out_dim, const int64_t **out_data) {
    const auto tensor_ptr = static_cast<at::Tensor *>(input);
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
