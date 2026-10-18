#include "model_config.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *preset = argc == 2 ? argv[1] : "tiny";
    if (argc > 2 || (strcmp(preset, "tiny") && strcmp(preset, "gpt2"))) {
        fprintf(stderr, "Usage: gpt2_c [tiny|gpt2]\n");
        return 1;
    }
    const ModelConfig config = strcmp(preset, "gpt2") == 0
        ? model_config_gpt2() : model_config_tiny();
    if (!model_config_valid(&config)) {
        fprintf(stderr, "Invalid model dimensions\n");
        return 1;
    }
    printf("GPT-2 C scaffold (%s)\n", preset);
    printf("vocab=%d context=%d layers=%d heads=%d embedding=%d head_size=%d\n",
        config.vocab_size, config.context_length, config.n_layer,
        config.n_head, config.n_embd, config.n_embd / config.n_head);
    puts("Configuration inspection only; no model weights are allocated.");
    return 0;
}
