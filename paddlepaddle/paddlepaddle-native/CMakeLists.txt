cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(djl_paddle)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(JNI REQUIRED)

set(SOURCE_FILES
        "src/main/native/djl_paddle_jni_utils.h"
        "src/main/native/ai_djl_paddlepaddle_jni_PaddleLibrary_tensor.cc"
        "src/main/native/ai_djl_paddlepaddle_jni_PaddleLibrary_inference.cc"
        )

link_directories(paddle/paddle/lib)
add_library(djl_paddle SHARED ${SOURCE_FILES})
target_include_directories(djl_paddle PUBLIC
    ${JNI_INCLUDE_DIRS}
    paddle/paddle/include
    build/include)
target_link_libraries(djl_paddle paddle_fluid)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif(MSVC)