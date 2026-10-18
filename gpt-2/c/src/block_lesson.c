#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SEQUENCE_LENGTH = 3, EMBEDDING_SIZE = 2, MLP_WIDTH = 4 * EMBEDDING_SIZE };
static const double LAYER_NORM_EPSILON = 1e-5;
static const double SQRT_TWO_OVER_PI = 0.7978845608028654;

/* Reuse the first residual-stream sequence from Lesson 4. */
static double input[SEQUENCE_LENGTH][EMBEDDING_SIZE] = {
    {1.1, 0.0}, {0.0, 1.2}, {1.3, 1.3}
};

/* Identity projections keep Lesson 6 focused on the decoder-block structure. */
static double query_weights[EMBEDDING_SIZE][EMBEDDING_SIZE] = {
    {1.0, 0.0}, {0.0, 1.0}
};
static double key_weights[EMBEDDING_SIZE][EMBEDDING_SIZE] = {
    {1.0, 0.0}, {0.0, 1.0}
};
static double value_weights[EMBEDDING_SIZE][EMBEDDING_SIZE] = {
    {1.0, 0.0}, {0.0, 1.0}
};
static double attention_output_weights[EMBEDDING_SIZE][EMBEDDING_SIZE] = {
    {1.0, 0.0}, {0.0, 1.0}
};

/* LayerNorm starts with gamma=1 and beta=0; training can learn both. */
static double norm1_gamma[EMBEDDING_SIZE] = {1.0, 1.0};
static double norm1_beta[EMBEDDING_SIZE] = {0.0, 0.0};
static double norm2_gamma[EMBEDDING_SIZE] = {1.0, 1.0};
static double norm2_beta[EMBEDDING_SIZE] = {0.0, 0.0};

/* Linear-layer weights use (output channel, input channel) order, as in nn.Linear. */
static double mlp_input_weights[MLP_WIDTH][EMBEDDING_SIZE] = {
    {0.20, -0.10}, {-0.10, 0.25}, {0.30, 0.05}, {0.00, 0.20},
    {-0.20, 0.10}, {0.10, -0.15}, {0.15, 0.00}, {-0.05, 0.30}
};
static double mlp_input_bias[MLP_WIDTH] = {0.0};
static double mlp_output_weights[EMBEDDING_SIZE][MLP_WIDTH] = {
    {0.10, -0.10, 0.20, 0.00, 0.10, -0.15, 0.05, 0.10},
    {-0.20, 0.15, 0.05, 0.20, 0.10, 0.05, -0.05, 0.20}
};
static double mlp_output_bias[EMBEDDING_SIZE] = {0.0};

typedef struct {
    double normalized_input[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double attention_weights[SEQUENCE_LENGTH][SEQUENCE_LENGTH];
    double attention_branch[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double after_attention[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double normalized_for_mlp[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double mlp_pre_activation[SEQUENCE_LENGTH][MLP_WIDTH];
    double mlp_activated[SEQUENCE_LENGTH][MLP_WIDTH];
    double mlp_branch[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double output[SEQUENCE_LENGTH][EMBEDDING_SIZE];
} BlockTrace;

static void layer_norm(double values[SEQUENCE_LENGTH][EMBEDDING_SIZE],
        double gamma[EMBEDDING_SIZE], double beta[EMBEDDING_SIZE],
        double normalized[SEQUENCE_LENGTH][EMBEDDING_SIZE]) {
    for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
        double mean = 0.0;
        for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
            mean += values[position][channel];
        }
        mean /= EMBEDDING_SIZE;

        double variance = 0.0;
        for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
            const double difference = values[position][channel] - mean;
            variance += difference * difference;
        }
        variance /= EMBEDDING_SIZE;
        const double inverse_standard_deviation = 1.0 / sqrt(variance + LAYER_NORM_EPSILON);

        for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
            normalized[position][channel] = (values[position][channel] - mean)
                * inverse_standard_deviation * gamma[channel] + beta[channel];
        }
    }
}

static void project_embedding(double values[SEQUENCE_LENGTH][EMBEDDING_SIZE],
        double weights[EMBEDDING_SIZE][EMBEDDING_SIZE],
        double projected[SEQUENCE_LENGTH][EMBEDDING_SIZE]) {
    for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
        for (size_t output_channel = 0; output_channel < EMBEDDING_SIZE; ++output_channel) {
            projected[position][output_channel] = 0.0;
            for (size_t input_channel = 0; input_channel < EMBEDDING_SIZE; ++input_channel) {
                projected[position][output_channel] += values[position][input_channel]
                    * weights[output_channel][input_channel];
            }
        }
    }
}

static void causal_attention(double query[SEQUENCE_LENGTH][EMBEDDING_SIZE],
        double key[SEQUENCE_LENGTH][EMBEDDING_SIZE],
        double value[SEQUENCE_LENGTH][EMBEDDING_SIZE],
        double weights[SEQUENCE_LENGTH][SEQUENCE_LENGTH],
        double context[SEQUENCE_LENGTH][EMBEDDING_SIZE]) {
    double logits[SEQUENCE_LENGTH][SEQUENCE_LENGTH];
    const double scale = sqrt((double) EMBEDDING_SIZE);
    for (size_t query_position = 0; query_position < SEQUENCE_LENGTH; ++query_position) {
        for (size_t key_position = 0; key_position < SEQUENCE_LENGTH; ++key_position) {
            double dot_product = 0.0;
            for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
                dot_product += query[query_position][channel] * key[key_position][channel];
            }
            logits[query_position][key_position] = key_position > query_position
                ? -INFINITY : dot_product / scale;
        }

        double maximum = -INFINITY;
        for (size_t key_position = 0; key_position < SEQUENCE_LENGTH; ++key_position) {
            if (logits[query_position][key_position] > maximum) {
                maximum = logits[query_position][key_position];
            }
        }
        double denominator = 0.0;
        for (size_t key_position = 0; key_position < SEQUENCE_LENGTH; ++key_position) {
            weights[query_position][key_position] = exp(logits[query_position][key_position] - maximum);
            denominator += weights[query_position][key_position];
        }
        for (size_t key_position = 0; key_position < SEQUENCE_LENGTH; ++key_position) {
            weights[query_position][key_position] /= denominator;
        }
    }

    for (size_t query_position = 0; query_position < SEQUENCE_LENGTH; ++query_position) {
        for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
            context[query_position][channel] = 0.0;
            for (size_t key_position = 0; key_position < SEQUENCE_LENGTH; ++key_position) {
                context[query_position][channel] += weights[query_position][key_position]
                    * value[key_position][channel];
            }
        }
    }
}

static void mlp_linear_in(double values[SEQUENCE_LENGTH][EMBEDDING_SIZE],
        double projected[SEQUENCE_LENGTH][MLP_WIDTH]) {
    for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
        for (size_t neuron = 0; neuron < MLP_WIDTH; ++neuron) {
            projected[position][neuron] = mlp_input_bias[neuron];
            for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
                projected[position][neuron] += values[position][channel]
                    * mlp_input_weights[neuron][channel];
            }
        }
    }
}

static void gelu_tanh(double values[SEQUENCE_LENGTH][MLP_WIDTH],
        double activated[SEQUENCE_LENGTH][MLP_WIDTH]) {
    for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
        for (size_t neuron = 0; neuron < MLP_WIDTH; ++neuron) {
            const double value = values[position][neuron];
            const double cubic_term = 0.044715 * value * value * value;
            activated[position][neuron] = 0.5 * value
                * (1.0 + tanh(SQRT_TWO_OVER_PI * (value + cubic_term)));
        }
    }
}

static void mlp_linear_out(double values[SEQUENCE_LENGTH][MLP_WIDTH],
        double projected[SEQUENCE_LENGTH][EMBEDDING_SIZE]) {
    for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
        for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
            projected[position][channel] = mlp_output_bias[channel];
            for (size_t neuron = 0; neuron < MLP_WIDTH; ++neuron) {
                projected[position][channel] += values[position][neuron]
                    * mlp_output_weights[channel][neuron];
            }
        }
    }
}

static void build_block(BlockTrace *trace) {
    double query[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double key[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double value[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double attention_context[SEQUENCE_LENGTH][EMBEDDING_SIZE];

    layer_norm(input, norm1_gamma, norm1_beta, trace->normalized_input);
    project_embedding(trace->normalized_input, query_weights, query);
    project_embedding(trace->normalized_input, key_weights, key);
    project_embedding(trace->normalized_input, value_weights, value);
    causal_attention(query, key, value, trace->attention_weights, attention_context);
    project_embedding(attention_context, attention_output_weights, trace->attention_branch);

    for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
        for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
            trace->after_attention[position][channel] = input[position][channel]
                + trace->attention_branch[position][channel];
        }
    }

    layer_norm(trace->after_attention, norm2_gamma, norm2_beta, trace->normalized_for_mlp);
    mlp_linear_in(trace->normalized_for_mlp, trace->mlp_pre_activation);
    gelu_tanh(trace->mlp_pre_activation, trace->mlp_activated);
    mlp_linear_out(trace->mlp_activated, trace->mlp_branch);
    for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
        for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
            trace->output[position][channel] = trace->after_attention[position][channel]
                + trace->mlp_branch[position][channel];
        }
    }
}

static void print_embedding_matrix(const char *label,
        double values[SEQUENCE_LENGTH][EMBEDDING_SIZE]) {
    printf("%s (%d x %d):\n", label, SEQUENCE_LENGTH, EMBEDDING_SIZE);
    for (size_t row = 0; row < SEQUENCE_LENGTH; ++row) {
        printf("  [%0.4f, %0.4f]\n", values[row][0], values[row][1]);
    }
}

static void print_attention_matrix(const char *label,
        double values[SEQUENCE_LENGTH][SEQUENCE_LENGTH]) {
    printf("%s (%d x %d):\n", label, SEQUENCE_LENGTH, SEQUENCE_LENGTH);
    for (size_t row = 0; row < SEQUENCE_LENGTH; ++row) {
        printf("  [%0.4f, %0.4f, %0.4f]\n", values[row][0], values[row][1], values[row][2]);
    }
}

static void print_hidden_matrix(const char *label,
        double values[SEQUENCE_LENGTH][MLP_WIDTH]) {
    printf("%s (%d x %d):\n", label, SEQUENCE_LENGTH, MLP_WIDTH);
    for (size_t row = 0; row < SEQUENCE_LENGTH; ++row) {
        printf("  [");
        for (size_t column = 0; column < MLP_WIDTH; ++column) {
            printf("%s%0.4f", column == 0 ? "" : ", ", values[row][column]);
        }
        puts("]");
    }
}

static bool close_enough(double left, double right) {
    return fabs(left - right) < 1e-12;
}

static bool residual_equations_hold(const BlockTrace *trace) {
    for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
        for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
            if (!close_enough(trace->after_attention[position][channel],
                    input[position][channel] + trace->attention_branch[position][channel])
                    || !close_enough(trace->output[position][channel],
                    trace->after_attention[position][channel] + trace->mlp_branch[position][channel])
                    || !isfinite(trace->output[position][channel])) {
                return false;
            }
        }
    }
    return true;
}

static bool selected(const char *part, const char *step) {
    return strcmp(part, "all") == 0 || strcmp(part, step) == 0;
}

int main(int argc, char **argv) {
    const char *part = argc == 2 ? argv[1] : "all";
    if (argc > 2 || (strcmp(part, "all") != 0 && strcmp(part, "1") != 0
            && strcmp(part, "2") != 0 && strcmp(part, "3") != 0)) {
        fprintf(stderr, "Usage: block_lesson [all|1|2|3]\n");
        return EXIT_FAILURE;
    }

    BlockTrace trace = {0};
    build_block(&trace);
    if (selected(part, "1")) {
        puts("Step 1: normalize, attend causally, and add the attention residual");
        print_embedding_matrix("LayerNorm 1 output", trace.normalized_input);
        print_attention_matrix("Causal attention weights", trace.attention_weights);
        print_embedding_matrix("Attention branch output", trace.attention_branch);
        print_embedding_matrix("After x = x + attention(LayerNorm 1(x))", trace.after_attention);
        puts("The skip path preserves the incoming representation while attention adds context.\n");
    }
    if (selected(part, "2")) {
        puts("Step 2: normalize again, expand through the MLP, and add its residual");
        print_embedding_matrix("LayerNorm 2 output", trace.normalized_for_mlp);
        print_hidden_matrix("MLP linear layer before GELU (T, 4C)", trace.mlp_pre_activation);
        print_hidden_matrix("MLP after GELU", trace.mlp_activated);
        print_embedding_matrix("MLP branch output", trace.mlp_branch);
        puts("The MLP changes each position independently; it does not mix information across tokens.\n");
    }
    if (selected(part, "3")) {
        puts("Step 3: inspect the complete decoder-block output");
        print_embedding_matrix("Block output after the second residual", trace.output);
        const bool valid = residual_equations_hold(&trace);
        printf("Both residual equations hold and all outputs are finite: %s\n", valid ? "yes" : "no");
        if (!valid) {
            return EXIT_FAILURE;
        }
        puts("GPT-2 stacks 12 such blocks; the final normalization and vocabulary projection come afterward.\n");
    }
    return EXIT_SUCCESS;
}
