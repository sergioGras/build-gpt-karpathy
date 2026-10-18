#include "model_config.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static void show_dimensions(void) {
    const ModelConfig tutorial1 = model_config_tutorial1_tiny();
    const ModelConfig gpt2 = model_config_gpt2();
    puts("Step 1: compare model dimensions (V=vocabulary, T=context, C=width)");
    printf("Tutorial 1 tiny model    V=%d T=%d L=%d H=%d C=%d\n",
        tutorial1.vocab_size, tutorial1.context_length, tutorial1.n_layer,
        tutorial1.n_head, tutorial1.n_embd);
    printf("GPT-2 124M              V=%d T=%d L=%d H=%d C=%d\n",
        gpt2.vocab_size, gpt2.context_length, gpt2.n_layer,
        gpt2.n_head, gpt2.n_embd);
}

static void show_attention_width(void) {
    const ModelConfig config = model_config_gpt2();
    puts("Step 2: split the residual width evenly across heads");
    printf("head size = C / H = %d / %d = %d channels per head\n",
        config.n_embd, config.n_head, config.n_embd / config.n_head);
    puts("At each sequence position: hidden state (768) -> Q, K, V (768 each).");
    puts("Each is then viewed as 12 heads x 64 channels.");
}

static void show_parameter_count(void) {
    const ModelConfig config = model_config_gpt2();
    ModelConfig padded_config = config;
    padded_config.vocab_size = 50304;
    const uint64_t total = gpt2_parameter_count(&config);
    const uint64_t padded_total = gpt2_parameter_count(&padded_config);
    puts("Step 3: estimate GPT-2 parameters without allocating tensors");
    puts("V*C + T*C + L*(12*C*C + 13*C) + 2*C");
    printf("= %" PRIu64 " parameters (%.3f million)\n",
        total, (double)total / 1000000.0);
    puts("The output projection shares its weights with token embeddings.");
    printf("Training pads output rows to 50,304: %" PRIu64 " (+%" PRIu64 ")\n",
        padded_total, padded_total - total);
}

static int show_steps(const char *part) {
    const bool all = strcmp(part, "all") == 0;
    const bool dimensions = all || strcmp(part, "1") == 0
        || strcmp(part, "1-2") == 0 || strcmp(part, "1-3") == 0;
    const bool attention = all || strcmp(part, "2") == 0
        || strcmp(part, "1-2") == 0 || strcmp(part, "2-3") == 0
        || strcmp(part, "1-3") == 0;
    const bool parameter_count = all || strcmp(part, "3") == 0
        || strcmp(part, "2-3") == 0 || strcmp(part, "1-3") == 0;

    if (dimensions) show_dimensions();
    if (attention) show_attention_width();
    if (parameter_count) show_parameter_count();
    return 0;
}

int main(int argc, char **argv) {
    const char *part = argc == 2 ? argv[1] : "all";
        if (argc > 2 || (strcmp(part, "all") && strcmp(part, "1")
                && strcmp(part, "2") && strcmp(part, "3")
                && strcmp(part, "1-2") && strcmp(part, "2-3")
                && strcmp(part, "1-3"))) {
        fprintf(stderr, "Usage: architecture_lesson [all|1|2|3|1-2|2-3|1-3]\n");
        return 1;
    }
    return show_steps(part);
}
