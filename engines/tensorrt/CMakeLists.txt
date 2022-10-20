cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
project(djl_trt)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(JAVA_AWT_LIBRARY NotNeeded)
set(JAVA_AWT_INCLUDE_PATH NotNeeded)
find_package(JNI REQUIRED)
#find_library(TRT_ONNX_PARSER
#        NAMES nvonnxparser
#        PATH_SUFFIXES lib
#        REQUIRED)

find_path(UTILS_INCLUDE_DIR
        NAMES djl/utils.h
        PATHS ${PROJECT_SOURCE_DIR}/../../api/src/main/native
        REQUIRED)

set(SOURCE_FILES
        "src/main/native/ai_djl_tensorrt_jni_TrtLibrary.cc"
        "src/main/native/ai_djl_tensorrt_jni_cache.cc"
        "src/main/native/ai_djl_tensorrt_jni_cache.h"
        "src/main/native/ai_djl_tensorrt_jni_common.h"
        "src/main/native/ai_djl_tensorrt_jni_exception.h"
        "src/main/native/ai_djl_tensorrt_jni_log.cc"
        "src/main/native/ai_djl_tensorrt_jni_log.h"
        "src/main/native/ai_djl_tensorrt_jni_model.cc"
        "src/main/native/ai_djl_tensorrt_jni_model.h"
        )

add_library(djl_trt SHARED ${SOURCE_FILES})
target_include_directories(djl_trt PUBLIC
        ${JNI_INCLUDE_DIRS}
        ${UTILS_INCLUDE_DIR}
        /usr/local/cuda/include
        main/native
        trt/include
        build/include)
target_link_libraries(djl_trt nvonnxparser nvparsers)
