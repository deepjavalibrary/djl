#include <torch/torch.h>

extern "C" {
  int randomTensor(void* output) {
    const at::Tensor tensor = torch::ones({2, 3});
    output = (void *)(&tensor);
    return 1;
  }

  int getShape(void* input, int *dim, const int64_t **out_data) {
//    const at::Tensor* tensor_ptr = static_cast<at::Tensor*>(input);
    const at::Tensor tensor = torch::ones({1, 2, 3});
    *dim = tensor.dim();
    std::unique_ptr<std::vector<int64_t>> shape {new std::vector<int64_t>};
//    for (auto i = 0; i < tensor.dim(); ++i) {
//      shape[i] = tensor.size(i);
//    }
//    shape[0] = 1;
//    shape[1] = 2;
//    shape[2] = 3;
    shape->emplace_back(1);
    shape->emplace_back(2);
    shape->emplace_back(3);
    *out_data = shape->data();
    return 1;
  }
}
