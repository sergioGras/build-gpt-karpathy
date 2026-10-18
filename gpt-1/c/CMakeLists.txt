cmake_minimum_required(VERSION 3.20)

project(gpt1_c LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_executable(gpt1_c src/main.c src/dataset.c)
target_include_directories(gpt1_c PRIVATE include)

add_executable(gpt1_tokenization src/tokenization_main.c src/dataset.c src/vocabulary.c)
target_include_directories(gpt1_tokenization PRIVATE include)
