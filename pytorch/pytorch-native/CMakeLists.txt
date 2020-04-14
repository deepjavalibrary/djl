cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(djl_torch)

find_package(Torch REQUIRED)
find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(SOURCE_FILES
    build/include/ai_djl_pytorch_jni_PyTorchLibrary.h
    "src/djl_pytorch_jni_utils.h"
    "src/ai_djl_pytorch_jni_PyTorchLibrary_system.cc"
    "src/ai_djl_pytorch_jni_PyTorchLibrary_nn_functional.cc"
    "src/ai_djl_pytorch_jni_PyTorchLibrary_inference.cc"
    "src/ai_djl_pytorch_jni_PyTorchLibrary_torch_comparison.cc"
    "src/ai_djl_pytorch_jni_PyTorchLibrary_tensor.cc"
    "src/ai_djl_pytorch_jni_PyTorchLibrary_torch_creation.cc"
    "src/ai_djl_pytorch_jni_PyTorchLibrary_torch_isjm.cc"
    "src/ai_djl_pytorch_jni_PyTorchLibrary_torch_pointwise.cc"
    "src/ai_djl_pytorch_jni_PyTorchLibrary_torch_reduction.cc"
    "src/ai_djl_pytorch_jni_PyTorchLibrary_torch_random_sampling.cc"
    "src/ai_djl_pytorch_jni_PyTorchLibrary_torch_other.cc"
    "src/ai_djl_pytorch_jni_PyTorchLibrary_torch_nn.cc"
    "src/djl_pytorch_jni_log.cc"
    "src/djl_pytorch_jni_log.h"
    "src/djl_pytorch_jni_error.h")
add_library(djl_torch SHARED ${SOURCE_FILES})
target_link_libraries(djl_torch "${TORCH_LIBRARIES}")
set_property(TARGET djl_torch PROPERTY CXX_STANDARD 14)
# We have to kill the default rpath and use current dir
set(CMAKE_SKIP_RPATH TRUE)
if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set_target_properties(djl_torch PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN")
endif ()

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET djl_torch
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:djl_torch>)
endif (MSVC)
