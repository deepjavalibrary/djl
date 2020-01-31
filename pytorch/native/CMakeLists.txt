cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(djl_torch)

find_package(Torch REQUIRED)
find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(SOURCE_FILES build/include/ai_djl_pytorch_jni_PyTorchLibrary.h "src /ai_djl_pytorch_jni_PyTorchLibrary.cc")
add_library(djl_torch SHARED ${SOURCE_FILES})
target_link_libraries(djl_torch "${TORCH_LIBRARIES}")
set_property(TARGET djl_torch PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET djl_torch
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:torch-wrapper>)
endif (MSVC)
