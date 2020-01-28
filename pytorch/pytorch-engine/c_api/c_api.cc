#include "c_api.h"

using namespace at;

int ones(NDArrayHandle* output) {
  *output = new Tensor(torch::ones({2, 3}));
  return 0;
}

int TensorToFloat(NDArrayHandle input, const void **data, size_t* size) {
  const auto tensor_ptr = static_cast<at::Tensor*>(input);
  *size = tensor_ptr->numel();
  *data = tensor_ptr->data_ptr<float>();
  return 0;
}

int TensorGetShape(NDArrayHandle input, int *out_dim, const int64_t **out_data) {
  const auto tensor_ptr = static_cast<at::Tensor*>(input);
  *out_dim = tensor_ptr->dim();
  auto* shape = new std::vector<int64_t>;
  for (auto i = 0; i < *out_dim; ++i) {
    shape->emplace_back(tensor_ptr->size(i));
  }
  *out_data = shape->data();
  return 0;
}
