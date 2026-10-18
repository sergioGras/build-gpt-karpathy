#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    BATCH_SIZE = 2,
    BLOCK_SIZE = 4,
    VOCAB_SIZE = 5,
    CHANNELS = 4,
    HEADS = 2,
    HEAD_SIZE = CHANNELS / HEADS,
    FEED_FORWARD_SIZE = 4 * CHANNELS,
    LAYERS = 2,
    GENERATED_TOKENS = 5
};

typedef double Tensor[BATCH_SIZE][BLOCK_SIZE][CHANNELS];
typedef double Logits[BATCH_SIZE][BLOCK_SIZE][VOCAB_SIZE];

static const char *const VOCABULARY[VOCAB_SIZE] = { "a", "b", "c", "d", "e" };
static const int TOKEN_IDS[BATCH_SIZE][BLOCK_SIZE] = {
    { 0, 1, 0, 2 },
    { 1, 2, 3, 0 }
};
static const int TARGET_IDS[BATCH_SIZE][BLOCK_SIZE] = {
    { 1, 0, 2, 3 },
    { 2, 3, 0, 1 }
};
static const double SAMPLE_THRESHOLDS[GENERATED_TOKENS] = {
    0.19, 0.47, 0.81, 0.33, 0.68
};
static const double EPSILON = 1e-5;

static int parse_part(const char *part, int *first_step, int *last_step) {
    *first_step = 1;
    *last_step = 4;
    if (strcmp(part, "1") == 0) *last_step = 1;
    else if (strcmp(part, "2") == 0) *first_step = *last_step = 2;
    else if (strcmp(part, "3") == 0) *first_step = *last_step = 3;
    else if (strcmp(part, "4") == 0) *first_step = *last_step = 4;
    else if (strcmp(part, "1-2") == 0) *last_step = 2;
    else if (strcmp(part, "1-3") == 0) *last_step = 3;
    else if (strcmp(part, "2-4") == 0) *first_step = 2;
    else if (strcmp(part, "3-4") == 0) *first_step = 3;
    else if (strcmp(part, "1-4") != 0 && strcmp(part, "all") != 0) return 0;
    return 1;
}

static double deterministic_weight(int seed, int output_channel, int input_channel,
                                   int modulus, double scale) {
    int pattern = (seed * 5 + (output_channel + 1) * (input_channel + 2)) % modulus;
    return (pattern - modulus / 2) * scale;
}

static double deterministic_bias(int seed, int channel) {
    return ((seed + channel * 2) % 5 - 2) * 0.01;
}

static void layer_norm(const Tensor input, Tensor output, int time_steps) {
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int time = 0; time < time_steps; time++) {
            double mean = 0.0;
            for (int channel = 0; channel < CHANNELS; channel++) {
                mean += input[batch][time][channel];
            }
            mean /= CHANNELS;

            double variance = 0.0;
            for (int channel = 0; channel < CHANNELS; channel++) {
                double centered = input[batch][time][channel] - mean;
                variance += centered * centered;
            }
            variance /= CHANNELS;

            for (int channel = 0; channel < CHANNELS; channel++) {
                output[batch][time][channel] =
                    (input[batch][time][channel] - mean) / sqrt(variance + EPSILON);
            }
        }
    }
}

static void add_residual(const Tensor input, const Tensor branch, Tensor output,
                         int time_steps) {
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int time = 0; time < time_steps; time++) {
            for (int channel = 0; channel < CHANNELS; channel++) {
                output[batch][time][channel] =
                    input[batch][time][channel] + branch[batch][time][channel];
            }
        }
    }
}

static void multi_head_attention(const Tensor input, Tensor output,
                                 int layer, int time_steps) {
    double concatenated[BATCH_SIZE][BLOCK_SIZE][CHANNELS] = { 0 };
    double queries[BATCH_SIZE][BLOCK_SIZE][HEAD_SIZE];
    double keys[BATCH_SIZE][BLOCK_SIZE][HEAD_SIZE];
    double values[BATCH_SIZE][BLOCK_SIZE][HEAD_SIZE];
    double weights[BLOCK_SIZE][BLOCK_SIZE];

    for (int head = 0; head < HEADS; head++) {
        int query_seed = layer * 10 + head * 3 + 1;
        int key_seed = layer * 10 + head * 3 + 2;
        int value_seed = layer * 10 + head * 3 + 3;

        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < time_steps; time++) {
                for (int channel = 0; channel < HEAD_SIZE; channel++) {
                    queries[batch][time][channel] = 0.0;
                    keys[batch][time][channel] = 0.0;
                    values[batch][time][channel] = 0.0;
                    for (int input_channel = 0; input_channel < CHANNELS; input_channel++) {
                        queries[batch][time][channel] += input[batch][time][input_channel]
                            * deterministic_weight(query_seed, channel, input_channel, 11, 0.08);
                        keys[batch][time][channel] += input[batch][time][input_channel]
                            * deterministic_weight(key_seed, channel, input_channel, 11, 0.08);
                        values[batch][time][channel] += input[batch][time][input_channel]
                            * deterministic_weight(value_seed, channel, input_channel, 11, 0.08);
                    }
                }
            }
        }

        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            for (int time = 0; time < time_steps; time++) {
                double maximum = -INFINITY;
                for (int key_time = 0; key_time < time_steps; key_time++) {
                    if (key_time > time) {
                        weights[time][key_time] = -INFINITY;
                        continue;
                    }
                    double score = 0.0;
                    for (int channel = 0; channel < HEAD_SIZE; channel++) {
                        score += queries[batch][time][channel] * keys[batch][key_time][channel];
                    }
                    weights[time][key_time] = score / sqrt((double)HEAD_SIZE);
                    if (weights[time][key_time] > maximum) maximum = weights[time][key_time];
                }

                double total = 0.0;
                for (int key_time = 0; key_time < time_steps; key_time++) {
                    if (key_time > time) {
                        weights[time][key_time] = 0.0;
                    } else {
                        weights[time][key_time] = exp(weights[time][key_time] - maximum);
                        total += weights[time][key_time];
                    }
                }
                for (int key_time = 0; key_time <= time; key_time++) {
                    weights[time][key_time] /= total;
                }

                for (int channel = 0; channel < HEAD_SIZE; channel++) {
                    double value = 0.0;
                    for (int key_time = 0; key_time <= time; key_time++) {
                        value += weights[time][key_time] * values[batch][key_time][channel];
                    }
                    concatenated[batch][time][head * HEAD_SIZE + channel] = value;
                }
            }
        }
    }

    int projection_seed = layer * 10 + 7;
    int bias_seed = layer * 10 + 8;
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int time = 0; time < time_steps; time++) {
            for (int output_channel = 0; output_channel < CHANNELS; output_channel++) {
                double value = deterministic_bias(bias_seed, output_channel);
                for (int input_channel = 0; input_channel < CHANNELS; input_channel++) {
                    value += concatenated[batch][time][input_channel]
                        * deterministic_weight(projection_seed, output_channel,
                                               input_channel, 11, 0.05);
                }
                output[batch][time][output_channel] = value;
            }
        }
    }
}

static void feed_forward(const Tensor input, Tensor output,
                         int layer, int time_steps) {
    double hidden[BATCH_SIZE][BLOCK_SIZE][FEED_FORWARD_SIZE];
    int first_seed = layer * 13 + 9;
    int first_bias_seed = layer * 13 + 10;
    int second_seed = layer * 13 + 11;
    int second_bias_seed = layer * 13 + 12;

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int time = 0; time < time_steps; time++) {
            for (int hidden_channel = 0; hidden_channel < FEED_FORWARD_SIZE; hidden_channel++) {
                double value = deterministic_bias(first_bias_seed, hidden_channel);
                for (int input_channel = 0; input_channel < CHANNELS; input_channel++) {
                    value += input[batch][time][input_channel]
                        * deterministic_weight(first_seed, hidden_channel, input_channel, 11, 0.04);
                }
                hidden[batch][time][hidden_channel] = value > 0.0 ? value : 0.0;
            }
            for (int output_channel = 0; output_channel < CHANNELS; output_channel++) {
                double value = deterministic_bias(second_bias_seed, output_channel);
                for (int hidden_channel = 0; hidden_channel < FEED_FORWARD_SIZE; hidden_channel++) {
                    value += hidden[batch][time][hidden_channel]
                        * deterministic_weight(second_seed, output_channel,
                                               hidden_channel, 11, 0.03);
                }
                output[batch][time][output_channel] = value;
            }
        }
    }
}

static void initialize_embeddings(double token_table[VOCAB_SIZE][CHANNELS],
                                  double position_table[BLOCK_SIZE][CHANNELS]) {
    for (int token = 0; token < VOCAB_SIZE; token++) {
        for (int channel = 0; channel < CHANNELS; channel++) {
            int pattern = (token * 5 + channel * 3 + 1) % 11 - 5;
            token_table[token][channel] = pattern * 0.1;
        }
    }
    for (int position = 0; position < BLOCK_SIZE; position++) {
        for (int channel = 0; channel < CHANNELS; channel++) {
            int pattern = (position * 7 + channel * 2 + 2) % 9 - 4;
            position_table[position][channel] = pattern * 0.1;
        }
    }
}

static void model_forward(const int token_ids[BATCH_SIZE][BLOCK_SIZE],
                          int time_steps,
                          Tensor embedding_output,
                          Tensor block_outputs[LAYERS],
                          Logits logits) {
    double token_table[VOCAB_SIZE][CHANNELS];
    double position_table[BLOCK_SIZE][CHANNELS];
    initialize_embeddings(token_table, position_table);

    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int time = 0; time < time_steps; time++) {
            int token_id = token_ids[batch][time];
            for (int channel = 0; channel < CHANNELS; channel++) {
                embedding_output[batch][time][channel] =
                    token_table[token_id][channel] + position_table[time][channel];
            }
        }
    }

    Tensor current;
    memcpy(current, embedding_output, sizeof(current));
    for (int layer = 0; layer < LAYERS; layer++) {
        Tensor normalized;
        Tensor attention_output;
        Tensor after_attention;
        Tensor normalized_again;
        Tensor feed_forward_output;
        layer_norm(current, normalized, time_steps);
        multi_head_attention(normalized, attention_output, layer, time_steps);
        add_residual(current, attention_output, after_attention, time_steps);
        layer_norm(after_attention, normalized_again, time_steps);
        feed_forward(normalized_again, feed_forward_output, layer, time_steps);
        add_residual(after_attention, feed_forward_output,
                     block_outputs[layer], time_steps);
        memcpy(current, block_outputs[layer], sizeof(current));
    }

    Tensor normalized_final;
    layer_norm(current, normalized_final, time_steps);
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int time = 0; time < time_steps; time++) {
            for (int vocabulary_id = 0; vocabulary_id < VOCAB_SIZE; vocabulary_id++) {
                double value = deterministic_bias(42, vocabulary_id);
                for (int channel = 0; channel < CHANNELS; channel++) {
                    value += normalized_final[batch][time][channel]
                        * deterministic_weight(41, vocabulary_id, channel, 11, 0.05);
                }
                logits[batch][time][vocabulary_id] = value;
            }
        }
    }
}

static double cross_entropy(const Logits logits, int time_steps) {
    double total_loss = 0.0;
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        for (int time = 0; time < time_steps; time++) {
            double maximum = logits[batch][time][0];
            for (int token = 1; token < VOCAB_SIZE; token++) {
                if (logits[batch][time][token] > maximum) maximum = logits[batch][time][token];
            }
            double exponential_sum = 0.0;
            for (int token = 0; token < VOCAB_SIZE; token++) {
                exponential_sum += exp(logits[batch][time][token] - maximum);
            }
            int target = TARGET_IDS[batch][time];
            total_loss += maximum + log(exponential_sum) - logits[batch][time][target];
        }
    }
    return total_loss / (BATCH_SIZE * time_steps);
}

static int sample_token(const double logits[VOCAB_SIZE], double threshold) {
    double maximum = logits[0];
    for (int token = 1; token < VOCAB_SIZE; token++) {
        if (logits[token] > maximum) maximum = logits[token];
    }
    double probabilities[VOCAB_SIZE];
    double total = 0.0;
    for (int token = 0; token < VOCAB_SIZE; token++) {
        probabilities[token] = exp(logits[token] - maximum);
        total += probabilities[token];
    }

    double cumulative = 0.0;
    for (int token = 0; token < VOCAB_SIZE; token++) {
        cumulative += probabilities[token] / total;
        if (threshold < cumulative) return token;
    }
    return VOCAB_SIZE - 1;
}

static void generate(int generated[2 + GENERATED_TOKENS]) {
    int generated_count = 2;
    generated[0] = 0;
    generated[1] = 1;

    for (int step = 0; step < GENERATED_TOKENS; step++) {
        int context_length = generated_count < BLOCK_SIZE ? generated_count : BLOCK_SIZE;
        int start = generated_count - context_length;
        int context[BATCH_SIZE][BLOCK_SIZE] = { 0 };
        for (int time = 0; time < context_length; time++) {
            context[0][time] = generated[start + time];
        }

        Tensor embedding_output;
        Tensor block_outputs[LAYERS];
        Logits logits;
        model_forward(context, context_length, embedding_output, block_outputs, logits);
        int last_time = context_length - 1;
        generated[generated_count++] =
            sample_token(logits[0][last_time], SAMPLE_THRESHOLDS[step]);
    }
}

static void print_tensor(const char *label, const Tensor values, int time_steps) {
    printf("%s, shape [batch, time, channels] = [%d, %d, %d]:\n",
           label, BATCH_SIZE, time_steps, CHANNELS);
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        printf("  batch %d\n", batch);
        for (int time = 0; time < time_steps; time++) {
            printf("    t=%d [", time);
            for (int channel = 0; channel < CHANNELS; channel++) {
                printf("%s%.4f", channel == 0 ? "" : ", ", values[batch][time][channel]);
            }
            puts("]");
        }
    }
}

static void print_logits(const Logits logits, int time_steps) {
    printf("Logits, shape [batch, time, vocab] = [%d, %d, %d] (batch 0):\n",
           BATCH_SIZE, time_steps, VOCAB_SIZE);
    for (int time = 0; time < time_steps; time++) {
        printf("  t=%d [", time);
        for (int token = 0; token < VOCAB_SIZE; token++) {
            printf("%s%.4f", token == 0 ? "" : ", ", logits[0][time][token]);
        }
        puts("]");
    }
}

static void print_generated(const int generated[2 + GENERATED_TOKENS]) {
    printf("Prompt: %s%s\nGenerated sequence: ",
           VOCABULARY[generated[0]], VOCABULARY[generated[1]]);
    for (int index = 0; index < 2 + GENERATED_TOKENS; index++) {
        printf("%s", VOCABULARY[generated[index]]);
    }
    puts("");
}

int main(int argc, char *argv[]) {
    const char *part = argc == 1 ? "all" : argv[1];
    int first_step;
    int last_step;
    if (argc > 2 || !parse_part(part, &first_step, &last_step)) {
        fprintf(stderr, "Usage: gpt1_decoder_model [all|1|2|3|4|1-2|1-3|1-4|2-4|3-4]\n");
        return EXIT_FAILURE;
    }

    Tensor embedding_output;
    Tensor block_outputs[LAYERS];
    Logits logits;
    model_forward(TOKEN_IDS, BLOCK_SIZE, embedding_output, block_outputs, logits);

    if (first_step <= 1) {
        puts("Step 1 - Embed token IDs and add position vectors");
        puts("Vocabulary IDs: a=0, b=1, c=2, d=3, e=4");
        printf("Tiny model: vocabulary=%d, model width=%d, context=%d, layers=%d, heads=%d\n",
               VOCAB_SIZE, CHANNELS, BLOCK_SIZE, LAYERS, HEADS);
        print_tensor("Input to the Transformer blocks", embedding_output, BLOCK_SIZE);
    }

    if (first_step <= 2 && last_step >= 2) {
        puts("Step 2 - Pass representations through two pre-LayerNorm blocks");
        puts("Each block adds causal multi-head attention, then a position-wise feed-forward network.");
        for (int layer = 0; layer < LAYERS; layer++) {
            char label[48];
            snprintf(label, sizeof(label), "Output after block %d", layer + 1);
            print_tensor(label, block_outputs[layer], BLOCK_SIZE);
        }
    }

    if (first_step <= 3 && last_step >= 3) {
        puts("Step 3 - Project each final vector to next-token logits");
        puts("Next-token target IDs, shape [batch, time] = [2, 4]:");
        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            printf("  batch %d [", batch);
            for (int time = 0; time < BLOCK_SIZE; time++) {
                printf("%s%d", time == 0 ? "" : ", ", TARGET_IDS[batch][time]);
            }
            puts("]");
        }
        print_logits(logits, BLOCK_SIZE);
        printf("Mean next-token cross-entropy on the tiny target batch: %.6f\n",
               cross_entropy(logits, BLOCK_SIZE));
        puts("The output width is the vocabulary size; these fixed weights have not been trained.");
    }

    if (last_step >= 4) {
        int generated[2 + GENERATED_TOKENS];
        generate(generated);
        puts("Step 4 - Generate one token at a time");
        puts("Each step uses the last position's logits; the context is cropped to the four-position limit.");
        print_generated(generated);
        puts("Fixed cumulative-probability thresholds make this untrained demo repeatable across languages.");
    }
    return EXIT_SUCCESS;
}
