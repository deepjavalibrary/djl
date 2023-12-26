cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(djl_llama CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS ON)

set(JAVA_AWT_LIBRARY NotNeeded)
set(JAVA_AWT_INCLUDE_PATH NotNeeded)
find_package(JNI REQUIRED)

add_subdirectory(llama.cpp)
include(build-args.cmake)
add_library(djl_llama SHARED src/main/native/ai_djl_llama.cpp)

target_include_directories(djl_llama PRIVATE
    ${JNI_INCLUDE_DIRS}
    src/main/native
    llama.cpp
    llama.cpp/common
    build/include)
target_link_libraries(djl_llama PRIVATE common llama ${LLAMA_EXTRA_LIBS})
target_compile_features(djl_llama PRIVATE cxx_std_11)
