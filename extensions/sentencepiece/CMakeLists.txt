cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
project(sentencepiece-native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# disable tcmalloc to avoid extra dependency
# TODO enable it when they fix static link tcmalloc option
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "Build sentencepiece static library")
# disable shared library as we only link static library
set(SPM_ENABLE_SHARED OFF CACHE BOOL "Build sentencepiece static library")

if (MSVC)
  string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_DEBUG          ${CMAKE_CXX_FLAGS_DEBUG})
  string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_MINSIZEREL     ${CMAKE_CXX_FLAGS_MINSIZEREL})
  string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE        ${CMAKE_CXX_FLAGS_RELEASE})
  string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
  add_definitions("/wd4267 /wd4244 /wd4305 /Zc:strictStrings /utf-8")
endif()

set(JAVA_AWT_LIBRARY NotNeeded)
set(JAVA_AWT_INCLUDE_PATH NotNeeded)
find_package(JNI REQUIRED)
find_path(UTILS_INCLUDE_DIR NAMES djl/utils.h PATHS ${PROJECT_SOURCE_DIR}/../../api/src/main/native REQUIRED)

add_subdirectory(sentencepiece)

add_library(sentencepiece_native SHARED src/main/native/ai_djl_sentencepiece_jni_SentencePieceLibrary.cc)
target_include_directories(sentencepiece_native PUBLIC
    ${JNI_INCLUDE_DIRS}
    ${UTILS_INCLUDE_DIR}
    sentencepiece/src
    build/include)
target_link_libraries(sentencepiece_native sentencepiece-static)
