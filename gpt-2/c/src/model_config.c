#include "model_config.h"

ModelConfig model_config_tiny(void) {
    return (ModelConfig){256, 32, 2, 4, 32};
}

ModelConfig model_config_gpt2(void) {
    return (ModelConfig){50257, 1024, 12, 12, 768};
}

bool model_config_valid(const ModelConfig *config) {
    return config && config->vocab_size > 0 && config->context_length > 0
        && config->n_layer > 0 && config->n_head > 0 && config->n_embd > 0
        && config->n_embd % config->n_head == 0;
}
