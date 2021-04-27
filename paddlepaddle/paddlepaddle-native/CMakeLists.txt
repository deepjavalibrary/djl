cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(djl_paddle)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(JNI REQUIRED)
find_library(PADDLE_LIBRARY
        NAMES paddle_inference
        PATHS paddle/paddle
        PATH_SUFFIXES lib
        DOC "The paddlepaddle library")
find_path(UTILS_INCLUDE_DIR NAMES djl/utils.h PATHS ${PROJECT_SOURCE_DIR}/../../api/src/main/native REQUIRED)


set(SOURCE_FILES
        "src/main/native/djl_paddle_utils.h"
        "src/main/native/ai_djl_paddlepaddle_jni_PaddleLibrary_tensor.cc"
        "src/main/native/ai_djl_paddlepaddle_jni_PaddleLibrary_inference.cc"
        )

add_library(djl_paddle SHARED ${SOURCE_FILES})
target_include_directories(djl_paddle PUBLIC
    ${JNI_INCLUDE_DIRS}
    ${UTILS_INCLUDE_DIR}
    paddle/paddle/include
    build/include)
target_link_libraries(djl_paddle ${PADDLE_LIBRARY})

# We have to kill the default rpath and use current dir
set(CMAKE_SKIP_RPATH TRUE)
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set_target_properties(djl_paddle PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN")
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif(MSVC)