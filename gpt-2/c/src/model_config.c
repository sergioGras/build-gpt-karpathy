#include "model_config.h"

ModelConfig model_config_tiny(void) {
    return (ModelConfig){256, 32, 2, 4, 32};
}

ModelConfig model_config_tutorial1_tiny(void) {
    return (ModelConfig){65, 16, 2, 2, 16};
}

ModelConfig model_config_gpt2(void) {
    return (ModelConfig){50257, 1024, 12, 12, 768};
}

uint64_t gpt2_parameter_count(const ModelConfig *config) {
    const uint64_t vocab = (uint64_t)config->vocab_size;
    const uint64_t context = (uint64_t)config->context_length;
    const uint64_t width = (uint64_t)config->n_embd;
    const uint64_t layers = (uint64_t)config->n_layer;
    return vocab * width + context * width
        + layers * (12 * width * width + 13 * width) + 2 * width;
}

bool model_config_valid(const ModelConfig *config) {
    return config && config->vocab_size > 0 && config->context_length > 0
        && config->n_layer > 0 && config->n_head > 0 && config->n_embd > 0
        && config->n_embd % config->n_head == 0;
}
