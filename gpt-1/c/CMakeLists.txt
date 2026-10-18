cmake_minimum_required(VERSION 3.20)

project(gpt1_c LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_executable(gpt1_c src/main.c src/dataset.c)
target_include_directories(gpt1_c PRIVATE include)

add_executable(gpt1_tokenization src/tokenization_main.c src/dataset.c src/vocabulary.c)
target_include_directories(gpt1_tokenization PRIVATE include)

add_executable(gpt1_batches src/batches_main.c src/dataset.c src/vocabulary.c)
target_include_directories(gpt1_batches PRIVATE include)

add_executable(gpt1_bigram src/bigram_main.c src/bigram_model.c src/dataset.c src/vocabulary.c)
target_include_directories(gpt1_bigram PRIVATE include)
add_executable(gpt1_bigram_training src/bigram_training_main.c src/bigram_model.c src/dataset.c src/vocabulary.c)
target_include_directories(gpt1_bigram_training PRIVATE include)
add_executable(gpt1_attention_math src/attention_math_main.c)
if (NOT WIN32)
    target_link_libraries(gpt1_bigram PRIVATE m)
    target_link_libraries(gpt1_bigram_training PRIVATE m)
    target_link_libraries(gpt1_attention_math PRIVATE m)
endif()
