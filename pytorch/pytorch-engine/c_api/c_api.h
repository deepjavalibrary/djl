/**
 * pytorch c_api
 */
#ifndef TORCH_WRAPPER_C_API_H
#define TORCH_WRAPPER_C_API_H

#include <torch/torch.h>

typedef void* NDArrayHandle;
extern "C" {
int ones(NDArrayHandle *output);
int TensorToFloat(NDArrayHandle input, const void **data, size_t* size);
int TensorGetShape(NDArrayHandle input, int *dim, const int64_t **out_data);
}



#endif //TORCH_WRAPPER_C_API_H
