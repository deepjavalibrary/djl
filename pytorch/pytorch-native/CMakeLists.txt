cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(djl_torch)

option(BUILD_ANDROID "Build the static djl_torch jni linked with libtorch statically" OFF)

if(BUILD_ANDROID AND NOT DEFINED ANDROID_ABI)
    message(FATAL_ERROR "Please specify ANDROID_ABI!")
endif()

if(NOT BUILD_ANDROID)
    message("Building djl_torch with the host...")
    find_package(Torch REQUIRED)
    find_package(JNI REQUIRED)

    include_directories(${JNI_INCLUDE_DIRS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
else()
    message("Building djl_torch with ${ANDROID_ABI}...")
    # The dependency of the cpuinfo clong.c
    find_library(log-lib log)
endif()

set(SOURCE_FILES
    "src/djl_pytorch_jni_utils.h"
    "src/ai_djl_pytorch_jni_PyTorchLibrary_system.cc"
    "src/ai_djl_pytorch_jni_PyTorchLibrary_nn_functional.cc"
    "src/ai_djl_pytorch_jni_PyTorchLibrary_inference.cc"
    "src/ai_djl_pytorch_jni_PyTorchLibrary_training.cc"
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
# build host
if(NOT BUILD_ANDROID)
    target_link_libraries(djl_torch "${TORCH_LIBRARIES}")
    target_include_directories(djl_torch PUBLIC build/include)
    set_property(TARGET djl_torch PROPERTY CXX_STANDARD 14)
    # We have to kill the default rpath and use current dir
    set(CMAKE_SKIP_RPATH TRUE)
    if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        set_target_properties(djl_torch PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN")
    endif()
    # The following code block is suggested to be used on Windows.
    # According to https://github.com/pytorch/pytorch/issues/25457,
    # the DLLs need to be copied to avoid memory errors.
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET djl_torch
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:djl_torch>)
    endif(MSVC)
else()
    # build android
    # function to static link the libtorch to djl_pytorch
    function(import_static_lib name)
        add_library(${name} STATIC IMPORTED)
        set_property(
            TARGET ${name}
            PROPERTY IMPORTED_LOCATION
            ${CMAKE_CURRENT_SOURCE_DIR}/libtorch_android/${ANDROID_ABI}/lib/${name}.a)
    endfunction(import_static_lib)
    list(APPEND LIBS libtorch libtorch_cpu libc10 libnnpack libXNNPACK libpytorch_qnnpack libeigen_blas libcpuinfo libclog)
    foreach(lib IN LISTS LIBS)
        import_static_lib(${lib})
    endforeach()

    target_include_directories(djl_torch PUBLIC build/include libtorch_android/${ANDROID_ABI}/include libtorch_android/${ANDROID_ABI}/include/torch/csrc/api/include)
    target_link_libraries(djl_torch
        libtorch
        -Wl,--whole-archive
        libtorch_cpu
        -Wl,--no-whole-archive
        libc10
        libnnpack
        libXNNPACK
        libpytorch_qnnpack
        libeigen_blas
        libcpuinfo
        libclog
        ${log-lib}
        )
endif()
