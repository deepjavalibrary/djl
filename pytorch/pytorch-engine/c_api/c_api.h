/**
 * pytorch c_api
 */
#ifndef TORCH_WRAPPER_C_API_H
#define TORCH_WRAPPER_C_API_H

#include <torch/torch.h>
#include <torch/script.h>

typedef void* ModuleHandle;
typedef void* IValueHandle;
typedef void* TensorHandle;
typedef void** IValueArrayHandle;

extern "C" {
    int ones(TensorHandle *output);
    int TensorToFloat(TensorHandle input, const void **data, size_t* size);
    int TensorGetShape(TensorHandle input, int *dim, const int64_t **out_data);
    int ModuleLoad(const char* path, ModuleHandle* moduleHandle);
    int ModuleEval(ModuleHandle moduleHandle);
    int ModuleForward(ModuleHandle moduleHandle, IValueArrayHandle iValueArrayHandle, int length, IValueHandle* resultHandle);
    int IValueCreateFromTensor(TensorHandle tensorHandle, IValueHandle* iValueHandle);
    int IValueToTensor(IValueHandle iValueHandle, TensorHandle* tensorHandle);
}



#endif //TORCH_WRAPPER_C_API_H
