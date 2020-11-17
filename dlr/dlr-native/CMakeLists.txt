cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
project(djl_dlr)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(JNI REQUIRED)

add_subdirectory(neo-ai-dlr)

add_library(djl_dlr SHARED src/main/native/ai_djl_dlr_jni_DlrLibrary.cc)
target_include_directories(djl_dlr PUBLIC
    ${JNI_INCLUDE_DIRS}
    neo-ai-dlr/include
    build/include)
target_link_libraries(djl_dlr dlr)
