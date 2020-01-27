cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torch-wrapper)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(torch-wrapper SHARED torch-wrapper.cpp)
#add_executable(torch-wrapper test.cpp)
target_link_libraries(torch-wrapper "${TORCH_LIBRARIES}")
set_property(TARGET torch-wrapper PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET torch-wrapper
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:torch-wrapper>)
endif (MSVC)
