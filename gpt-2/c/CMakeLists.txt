cmake_minimum_required(VERSION 3.21)
project(gpt2_c LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_executable(gpt2_c src/main.c src/model_config.c)
target_include_directories(gpt2_c PRIVATE include)
if(MSVC)
    target_compile_options(gpt2_c PRIVATE /W4)
else()
    target_compile_options(gpt2_c PRIVATE -Wall -Wextra -Wpedantic)
endif()
