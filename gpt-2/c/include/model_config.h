#ifndef GPT2_MODEL_CONFIG_H
#define GPT2_MODEL_CONFIG_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    int vocab_size;
    int context_length;
    int n_layer;
    int n_head;
    int n_embd;
} ModelConfig;

ModelConfig model_config_tiny(void);
ModelConfig model_config_tutorial1_tiny(void);
ModelConfig model_config_gpt2(void);
bool model_config_valid(const ModelConfig *config);
uint64_t gpt2_parameter_count(const ModelConfig *config);

#endif
