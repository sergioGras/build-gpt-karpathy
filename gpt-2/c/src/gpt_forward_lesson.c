#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SEQUENCE_LENGTH = 3, EMBEDDING_SIZE = 2, MLP_WIDTH = 4 * EMBEDDING_SIZE, VOCAB_SIZE = 4 };
static const double EPSILON = 1e-5;
static const double SQRT_TWO_OVER_PI = 0.7978845608028654;

/* The singleton batch axis is omitted from these C arrays: IDs still have shape (1, T). */
static const int token_ids[SEQUENCE_LENGTH] = {1, 2, 3};
static const double token_embedding_weight[VOCAB_SIZE][EMBEDDING_SIZE] = {
    {0.10, 0.20}, {0.30, -0.10}, {-0.20, 0.40}, {0.05, 0.25}
};
static const double position_embedding_weight[SEQUENCE_LENGTH][EMBEDDING_SIZE] = {
    {0.01, 0.00}, {0.00, 0.02}, {0.03, -0.01}
};

/* The toy block uses identity Q/K/V and attention-output projections, as in Lesson 6. */
static const double identity[EMBEDDING_SIZE][EMBEDDING_SIZE] = {
    {1.0, 0.0}, {0.0, 1.0}
};
static const double mlp_input_weights[MLP_WIDTH][EMBEDDING_SIZE] = {
    {0.20, -0.10}, {-0.10, 0.25}, {0.30, 0.05}, {0.00, 0.20},
    {-0.20, 0.10}, {0.10, -0.15}, {0.15, 0.00}, {-0.05, 0.30}
};
static const double mlp_output_weights[EMBEDDING_SIZE][MLP_WIDTH] = {
    {0.10, -0.10, 0.20, 0.00, 0.10, -0.15, 0.05, 0.10},
    {-0.20, 0.15, 0.05, 0.20, 0.10, 0.05, -0.05, 0.20}
};
static const double norm_gamma[EMBEDDING_SIZE] = {1.0, 1.0};
static const double norm_beta[EMBEDDING_SIZE] = {0.0, 0.0};

typedef struct {
    double token_vectors[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double position_vectors[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double input[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double normalized_input[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double attention_weights[SEQUENCE_LENGTH][SEQUENCE_LENGTH];
    double after_attention[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double mlp_hidden[SEQUENCE_LENGTH][MLP_WIDTH];
    double block_output[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double final_normalized[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double logits[SEQUENCE_LENGTH][VOCAB_SIZE];
} ForwardTrace;

static void layer_norm(double values[SEQUENCE_LENGTH][EMBEDDING_SIZE],
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
        const double inverse_stddev = 1.0 / sqrt(variance + EPSILON);
        for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
            normalized[position][channel] = (values[position][channel] - mean)
                * inverse_stddev * norm_gamma[channel] + norm_beta[channel];
        }
    }
}

static void project(double values[SEQUENCE_LENGTH][EMBEDDING_SIZE],
        const double weights[EMBEDDING_SIZE][EMBEDDING_SIZE],
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
    const double scale = sqrt((double) EMBEDDING_SIZE);
    for (size_t query_position = 0; query_position < SEQUENCE_LENGTH; ++query_position) {
        double logits[SEQUENCE_LENGTH];
        double maximum = -INFINITY;
        for (size_t key_position = 0; key_position < SEQUENCE_LENGTH; ++key_position) {
            double dot_product = 0.0;
            for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
                dot_product += query[query_position][channel] * key[key_position][channel];
            }
            logits[key_position] = key_position > query_position
                ? -INFINITY : dot_product / scale;
            if (logits[key_position] > maximum) maximum = logits[key_position];
        }

        double denominator = 0.0;
        for (size_t key_position = 0; key_position < SEQUENCE_LENGTH; ++key_position) {
            weights[query_position][key_position] = exp(logits[key_position] - maximum);
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

static double gelu_tanh(double value) {
    const double cubic_term = 0.044715 * value * value * value;
    return 0.5 * value * (1.0 + tanh(SQRT_TWO_OVER_PI * (value + cubic_term)));
}

static void build_forward(ForwardTrace *trace) {
    double query[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double key[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double value[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double attention_context[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double normalized_for_mlp[SEQUENCE_LENGTH][EMBEDDING_SIZE];
    double mlp_pre_activation[SEQUENCE_LENGTH][MLP_WIDTH];
    double mlp_activated[SEQUENCE_LENGTH][MLP_WIDTH];
    double mlp_branch[SEQUENCE_LENGTH][EMBEDDING_SIZE];

    for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
        const int token_id = token_ids[position];
        for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
            trace->token_vectors[position][channel] = token_embedding_weight[token_id][channel];
            trace->position_vectors[position][channel] = position_embedding_weight[position][channel];
            trace->input[position][channel] = trace->token_vectors[position][channel]
                + trace->position_vectors[position][channel];
        }
    }

    layer_norm(trace->input, trace->normalized_input);
    project(trace->normalized_input, identity, query);
    project(trace->normalized_input, identity, key);
    project(trace->normalized_input, identity, value);
    causal_attention(query, key, value, trace->attention_weights, attention_context);
    for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
        for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
            trace->after_attention[position][channel] = trace->input[position][channel]
                + attention_context[position][channel];
        }
    }

    layer_norm(trace->after_attention, normalized_for_mlp);
    for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
        for (size_t neuron = 0; neuron < MLP_WIDTH; ++neuron) {
            mlp_pre_activation[position][neuron] = 0.0;
            for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
                mlp_pre_activation[position][neuron] += normalized_for_mlp[position][channel]
                    * mlp_input_weights[neuron][channel];
            }
            mlp_activated[position][neuron] = gelu_tanh(mlp_pre_activation[position][neuron]);
            trace->mlp_hidden[position][neuron] = mlp_activated[position][neuron];
        }
    }
    for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
        for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
            mlp_branch[position][channel] = 0.0;
            for (size_t neuron = 0; neuron < MLP_WIDTH; ++neuron) {
                mlp_branch[position][channel] += mlp_activated[position][neuron]
                    * mlp_output_weights[channel][neuron];
            }
            trace->block_output[position][channel] = trace->after_attention[position][channel]
                + mlp_branch[position][channel];
        }
    }

    layer_norm(trace->block_output, trace->final_normalized);
    /* Weight tying: the output projection reuses token_embedding_weight. */
    for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
        for (size_t token = 0; token < VOCAB_SIZE; ++token) {
            trace->logits[position][token] = 0.0;
            for (size_t channel = 0; channel < EMBEDDING_SIZE; ++channel) {
                trace->logits[position][token] += trace->final_normalized[position][channel]
                    * token_embedding_weight[token][channel];
            }
        }
    }
}

static void print_matrix(const char *label, double values[SEQUENCE_LENGTH][EMBEDDING_SIZE]) {
    printf("%s (1 x %d x %d):\n", label, SEQUENCE_LENGTH, EMBEDDING_SIZE);
    for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
        printf("  [%0.4f, %0.4f]\n", values[position][0], values[position][1]);
    }
}

static void print_hidden(const char *label, double values[SEQUENCE_LENGTH][MLP_WIDTH]) {
    printf("%s (1 x %d x %d):\n", label, SEQUENCE_LENGTH, MLP_WIDTH);
    for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
        printf("  [");
        for (size_t neuron = 0; neuron < MLP_WIDTH; ++neuron) {
            printf("%s%0.4f", neuron == 0 ? "" : ", ", values[position][neuron]);
        }
        puts("]");
    }
}

static void print_logits(double values[SEQUENCE_LENGTH][VOCAB_SIZE]) {
    printf("Vocabulary logits (1 x %d x %d):\n", SEQUENCE_LENGTH, VOCAB_SIZE);
    for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
        printf("  [");
        for (size_t token = 0; token < VOCAB_SIZE; ++token) {
            printf("%s%0.4f", token == 0 ? "" : ", ", values[position][token]);
        }
        puts("]");
    }
}

static bool valid_output(const ForwardTrace *trace, size_t *prediction) {
    *prediction = 0;
    for (size_t position = 0; position < SEQUENCE_LENGTH; ++position) {
        double row_sum = 0.0;
        for (size_t key_position = 0; key_position < SEQUENCE_LENGTH; ++key_position) {
            const double weight = trace->attention_weights[position][key_position];
            if (!isfinite(weight) || (key_position > position && fabs(weight) > 1e-12)) return false;
            row_sum += weight;
        }
        if (fabs(row_sum - 1.0) > 1e-12) return false;
        for (size_t token = 0; token < VOCAB_SIZE; ++token) {
            if (!isfinite(trace->logits[position][token])) return false;
        }
    }
    for (size_t token = 1; token < VOCAB_SIZE; ++token) {
        if (trace->logits[SEQUENCE_LENGTH - 1][token]
                > trace->logits[SEQUENCE_LENGTH - 1][*prediction]) {
            *prediction = token;
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
        fprintf(stderr, "Usage: gpt_forward_lesson [all|1|2|3]\n");
        return EXIT_FAILURE;
    }

    ForwardTrace trace = {0};
    build_forward(&trace);
    size_t prediction;
    if (!valid_output(&trace, &prediction)) {
        fputs("The forward-pass checks failed.\n", stderr);
        return EXIT_FAILURE;
    }

    if (selected(part, "1")) {
        puts("Step 1: map token IDs to vectors and add position vectors");
        printf("Token IDs (1 x %d): [%d, %d, %d]\n", SEQUENCE_LENGTH,
            token_ids[0], token_ids[1], token_ids[2]);
        print_matrix("Initial residual stream", trace.input);
        puts("The singleton batch axis is omitted in C storage; these vectors have shape (1, T, C).\n");
    }
    if (selected(part, "2")) {
        puts("Step 2: run one decoder block and preserve the sequence shape");
        print_hidden("MLP hidden state", trace.mlp_hidden);
        print_matrix("Decoder block output", trace.block_output);
        puts("Attention mixes positions; the MLP transforms each position independently.\n");
    }
    if (selected(part, "3")) {
        puts("Step 3: final LayerNorm, tied output projection, and next-token choice");
        print_matrix("Final normalized hidden states", trace.final_normalized);
        print_logits(trace.logits);
        printf("Toy next-token choice at the last position: token ID %zu\n", prediction);
        puts("Logits are unnormalized scores. These fixed toy weights are not a trained language model.\n");
    }
    return EXIT_SUCCESS;
}
