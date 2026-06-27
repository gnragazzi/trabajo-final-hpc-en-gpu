cmake_minimum_required(VERSION 4.1)
project(01__Secuencial LANGUAGES C CUDA)

set(CMAKE_CUDA_STANDARD 17)

include_directories(lib)
add_executable(01__Secuencial main.cu)
target_link_libraries(01__Secuencial -lm)



