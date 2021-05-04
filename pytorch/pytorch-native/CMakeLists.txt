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

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
else()
    message("Building djl_torch with ${ANDROID_ABI}...")
    # The dependency of the cpuinfo clong.c
    find_library(log-lib log)
    # It is required to set those when using library elsewhere than Android SDK dir during the cross-compiling
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
endif()

# find utils header
find_path(UTILS_INCLUDE_DIR NAMES djl/utils.h PATHS ${PROJECT_SOURCE_DIR}/../../api/src/main/native REQUIRED)

set(SOURCE_FILES
    "src/main/native/djl_pytorch_utils.h"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_system.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_nn_functional.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_inference.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_ivalue.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_optim.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_torch_comparison.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_tensor.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_torch_creation.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_torch_isjm.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_torch_pointwise.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_torch_reduction.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_torch_random_sampling.cc"
    "src/main/native/ai_djl_pytorch_jni_PyTorchLibrary_torch_other.cc"
    "src/main/native/djl_pytorch_jni_log.cc"
    "src/main/native/djl_pytorch_jni_log.h"
    "src/main/native/djl_pytorch_jni_exception.h"
    "src/main/native/ai_djl_pytorch_jni_cache.h"
    "src/main/native/ai_djl_pytorch_jni_cache.cc")

add_library(djl_torch SHARED ${SOURCE_FILES})
# build host
if(NOT BUILD_ANDROID)
    target_link_libraries(djl_torch "${TORCH_LIBRARIES}")
    target_include_directories(djl_torch PUBLIC build/include ${JNI_INCLUDE_DIRS} ${UTILS_INCLUDE_DIR})
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
    list(APPEND LIBS libtorch libtorch_cpu libc10 libnnpack libXNNPACK libpytorch_qnnpack libpthreadpool libeigen_blas libcpuinfo libclog libVulkanWrapper)
    foreach(lib IN LISTS LIBS)
        import_static_lib(${lib})
    endforeach()

    target_include_directories(djl_torch PUBLIC build/include libtorch_android/${ANDROID_ABI}/include libtorch_android/${ANDROID_ABI}/include/torch/csrc/api/include ${UTILS_INCLUDE_DIR})
    target_link_libraries(djl_torch
        -Wl,--gc-sections
        -Wl,--whole-archive
        libtorch
        libtorch_cpu
        -Wl,--no-whole-archive
        libc10
        libnnpack
        libXNNPACK
        libpytorch_qnnpack
        libpthreadpool
        libeigen_blas
        libcpuinfo
        libclog
        libVulkanWrapper
        ${log-lib}
    )
endif()
