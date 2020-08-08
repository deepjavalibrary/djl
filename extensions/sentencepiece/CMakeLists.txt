cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(sentencepiece-native)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(JNI REQUIRED)

set(sentencepiece-native_STATIC_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/sentencepiece/src/libsentencepiece.a)

add_library(sentencepiece_native SHARED src/main/native/ai_djl_sentencepiece_jni_SentencePieceLibrary.cc)
target_include_directories(sentencepiece_native PUBLIC
    ${JNI_INCLUDE_DIRS}
    download/src
    build/include)
target_link_libraries(sentencepiece_native ${sentencepiece-native_STATIC_LIBRARIES})
