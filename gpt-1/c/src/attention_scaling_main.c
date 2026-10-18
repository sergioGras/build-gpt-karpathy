#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_HEAD_SIZE = 8, TOY_LOGIT_COUNT = 5 };

static int parse_part(const char *part, int *first_step, int *last_step) {
    *first_step = 1;
    *last_step = 3;
    if (strcmp(part, "1") == 0) *last_step = 1;
    else if (strcmp(part, "2") == 0) *first_step = *last_step = 2;
    else if (strcmp(part, "3") == 0) *first_step = *last_step = 3;
    else if (strcmp(part, "1-2") == 0) *last_step = 2;
    else if (strcmp(part, "1-3") != 0 && strcmp(part, "all") != 0) return 0;
    return 1;
}

static void softmax(const double *scores, size_t count, double *probabilities) {
    double maximum = scores[0];
    for (size_t index = 1; index < count; index++) {
        if (scores[index] > maximum) maximum = scores[index];
    }

    double total = 0.0;
    for (size_t index = 0; index < count; index++) {
        probabilities[index] = exp(scores[index] - maximum);
        total += probabilities[index];
    }
    for (size_t index = 0; index < count; index++) {
        probabilities[index] /= total;
    }
}

static double vector_entropy(const double *probabilities, size_t count) {
    double entropy = 0.0;
    for (size_t index = 0; index < count; index++) {
        if (probabilities[index] > 0.0) {
            entropy -= probabilities[index] * log(probabilities[index]);
        }
    }
    return entropy;
}

static void print_vector(const char *label, const double *values, size_t count) {
    printf("%s: [", label);
    for (size_t index = 0; index < count; index++) {
        printf("%s%.6g", index == 0 ? "" : ", ", values[index]);
    }
    puts("]");
}

static void print_key_vectors(const double keys[TOY_LOGIT_COUNT][MAX_HEAD_SIZE]) {
    puts("Five key vectors (each has eight components):");
    for (size_t key = 0; key < TOY_LOGIT_COUNT; key++) {
        printf("  key %zu: [", key);
        for (size_t channel = 0; channel < MAX_HEAD_SIZE; channel++) {
            printf("%s%.0f", channel == 0 ? "" : ", ", keys[key][channel]);
        }
        puts("]");
    }
}

int main(int argc, char *argv[]) {
    const char *part = argc == 1 ? "all" : argv[1];
    int first_step;
    int last_step;
    if (argc > 2 || !parse_part(part, &first_step, &last_step)) {
        fprintf(stderr, "Usage: gpt1_attention_scaling [all|1|2|3|1-2|1-3]\n");
        return EXIT_FAILURE;
    }

    if (first_step <= 1) {
        const size_t head_sizes[] = { 1, 2, 4, 8 };
        puts("Step 1 - See how query-key score variance grows with head size");
        puts("Fix q to all +1 values and enumerate every key made of -1/+1 values.");
        puts("This exactly samples independent, zero-mean, unit-variance key components.");
        puts("head size | raw score variance | variance after dividing by sqrt(head size)");

        for (size_t size_index = 0;
             size_index < sizeof(head_sizes) / sizeof(head_sizes[0]); size_index++) {
            size_t head_size = head_sizes[size_index];
            size_t key_count = (size_t)1 << head_size;
            double score_sum = 0.0;
            double squared_score_sum = 0.0;

            for (size_t pattern = 0; pattern < key_count; pattern++) {
                double score = 0.0;
                for (size_t channel = 0; channel < head_size; channel++) {
                    score += (pattern & ((size_t)1 << channel)) != 0 ? 1.0 : -1.0;
                }
                score_sum += score;
                squared_score_sum += score * score;
            }

            double mean = score_sum / (double)key_count;
            double variance = squared_score_sum / (double)key_count - mean * mean;
            double scaled_variance = variance / (double)head_size;
            printf("%9zu | %18.4f | %39.4f\n",
                   head_size, variance, scaled_variance);
        }
        puts("The dot product sums head_size terms, so its variance grows with head_size.");
        puts("Dividing by sqrt(head_size) divides that variance by head_size.");
    }

    if (first_step <= 2 && last_step >= 2) {
        const double base_logits[TOY_LOGIT_COUNT] = { 0.1, -0.2, 0.3, -0.2, 0.5 };
        double sharp_logits[TOY_LOGIT_COUNT];
        double base_probabilities[TOY_LOGIT_COUNT];
        double sharp_probabilities[TOY_LOGIT_COUNT];
        const double scale = 8.0; /* sqrt(64), used to make the effect easy to see. */

        for (size_t index = 0; index < TOY_LOGIT_COUNT; index++) {
            sharp_logits[index] = base_logits[index] * scale;
        }
        softmax(base_logits, TOY_LOGIT_COUNT, base_probabilities);
        softmax(sharp_logits, TOY_LOGIT_COUNT, sharp_probabilities);

        puts("Step 2 - See how large logits make softmax overly concentrated");
        print_vector("Example logits", base_logits, TOY_LOGIT_COUNT);
        print_vector("Softmax at the original scale", base_probabilities, TOY_LOGIT_COUNT);
        print_vector("Logits multiplied by 8 (sqrt of head size 64)",
                     sharp_logits, TOY_LOGIT_COUNT);
        print_vector("Softmax at the larger scale", sharp_probabilities, TOY_LOGIT_COUNT);
        printf("Largest probability: %.4f -> %.4f; entropy: %.4f -> %.4f nats\n",
               base_probabilities[4], sharp_probabilities[4],
               vector_entropy(base_probabilities, TOY_LOGIT_COUNT),
               vector_entropy(sharp_probabilities, TOY_LOGIT_COUNT));
        puts("Large score gaps concentrate probability; saturated softmax can produce small gradients.");
    }

    if (last_step >= 3) {
        double query[MAX_HEAD_SIZE];
        double keys[TOY_LOGIT_COUNT][MAX_HEAD_SIZE];
        double raw_scores[TOY_LOGIT_COUNT];
        double scaled_scores[TOY_LOGIT_COUNT];
        double raw_probabilities[TOY_LOGIT_COUNT];
        double scaled_probabilities[TOY_LOGIT_COUNT];

        for (size_t channel = 0; channel < MAX_HEAD_SIZE; channel++) query[channel] = 1.0;
        for (size_t key = 0; key < TOY_LOGIT_COUNT; key++) {
            for (size_t channel = 0; channel < MAX_HEAD_SIZE; channel++) {
                keys[key][channel] = channel < 2 * key ? -1.0 : 1.0;
            }
            raw_scores[key] = 0.0;
            for (size_t channel = 0; channel < MAX_HEAD_SIZE; channel++) {
                raw_scores[key] += query[channel] * keys[key][channel];
            }
            scaled_scores[key] = raw_scores[key] / sqrt((double)MAX_HEAD_SIZE);
        }
        softmax(raw_scores, TOY_LOGIT_COUNT, raw_probabilities);
        softmax(scaled_scores, TOY_LOGIT_COUNT, scaled_probabilities);

        puts("Step 3 - Scale actual query-key dot products before softmax");
        print_key_vectors(keys);
        print_vector("Raw dot-product scores", raw_scores, TOY_LOGIT_COUNT);
        print_vector("Raw softmax weights", raw_probabilities, TOY_LOGIT_COUNT);
        print_vector("Scores divided by sqrt(8)", scaled_scores, TOY_LOGIT_COUNT);
        print_vector("Scaled softmax weights", scaled_probabilities, TOY_LOGIT_COUNT);
        printf("Largest attention weight: %.4f -> %.4f\n",
               raw_probabilities[0], scaled_probabilities[0]);
        puts("Scaling controls the score range while preserving which keys score higher.");
    }

    return EXIT_SUCCESS;
}
