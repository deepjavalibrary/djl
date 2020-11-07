cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(jni_fasttext)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(JNI REQUIRED)

add_subdirectory(fastText)

add_library(jni_fasttext SHARED src/main/native/ai_djl_fasttext_jni_FastTextLibrary.cc)
target_include_directories(jni_fasttext PUBLIC
    ${JNI_INCLUDE_DIRS}
    fastText/src
    build/include)
target_link_libraries(jni_fasttext fasttext-static_pic)
