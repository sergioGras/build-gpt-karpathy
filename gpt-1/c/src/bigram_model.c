#include "bigram_model.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static uint32_t next_random_uint(BigramLanguageModel *model) {
    /* A small repeatable generator keeps this example dependency-free. */
    model->random_state = model->random_state * UINT32_C(1664525) + UINT32_C(1013904223);
    return model->random_state;
}

static double next_uniform(BigramLanguageModel *model) {
    return ((double)next_random_uint(model) + 0.5) / 4294967296.0;
}

static double next_normal(BigramLanguageModel *model) {
    /* Box-Muller transform: two uniform samples become a standard normal. */
    double first = next_uniform(model);
    double second = next_uniform(model);
    return sqrt(-2.0 * log(first)) * cos(6.283185307179586 * second);
}

int bigram_model_init(BigramLanguageModel *model, size_t vocab_size, uint32_t seed) {
    if (model == NULL || vocab_size == 0 || vocab_size > SIZE_MAX / vocab_size ||
        vocab_size * vocab_size > SIZE_MAX / sizeof(*model->token_embedding_table)) {
        return 0;
    }

    model->vocab_size = vocab_size;
    model->random_state = seed;
    model->optimizer_step = 0;
    model->token_embedding_table = malloc(vocab_size * vocab_size *
                                          sizeof(*model->token_embedding_table));
    model->first_moment = calloc(vocab_size * vocab_size, sizeof(*model->first_moment));
    model->second_moment = calloc(vocab_size * vocab_size, sizeof(*model->second_moment));
    if (model->token_embedding_table == NULL || model->first_moment == NULL ||
        model->second_moment == NULL) {
        bigram_model_destroy(model);
        model->vocab_size = 0;
        return 0;
    }

    for (size_t index = 0; index < vocab_size * vocab_size; index++) {
        /* Normal scores give the untrained model a varied starting distribution. */
        model->token_embedding_table[index] = (float)next_normal(model);
    }
    return 1;
}

void bigram_model_destroy(BigramLanguageModel *model) {
    if (model != NULL) {
        free(model->token_embedding_table);
        free(model->first_moment);
        free(model->second_moment);
        model->token_embedding_table = NULL;
        model->first_moment = NULL;
        model->second_moment = NULL;
        model->vocab_size = 0;
        model->optimizer_step = 0;
    }
}

int bigram_model_forward(const BigramLanguageModel *model, const int *input_ids,
                         size_t batch_size, size_t time_steps, float *logits) {
    if (model == NULL || input_ids == NULL || logits == NULL ||
        model->token_embedding_table == NULL || model->vocab_size == 0) {
        return 0;
    }

    for (size_t batch = 0; batch < batch_size; batch++) {
        for (size_t time = 0; time < time_steps; time++) {
            int input_id = input_ids[batch * time_steps + time];
            if (input_id < 0 || (size_t)input_id >= model->vocab_size) {
                return 0;
            }
            size_t output_offset = (batch * time_steps + time) * model->vocab_size;
            size_t table_offset = (size_t)input_id * model->vocab_size;
            for (size_t next_id = 0; next_id < model->vocab_size; next_id++) {
                logits[output_offset + next_id] =
                    model->token_embedding_table[table_offset + next_id];
            }
        }
    }
    return 1;
}

double bigram_model_cross_entropy(const BigramLanguageModel *model,
                                  const float *logits, const int *target_ids,
                                  size_t batch_size, size_t time_steps) {
    if (model == NULL || logits == NULL || target_ids == NULL ||
        model->vocab_size == 0 || batch_size == 0 || time_steps == 0) {
        return NAN;
    }

    double total_loss = 0.0;
    size_t position_count = batch_size * time_steps;
    for (size_t position = 0; position < position_count; position++) {
        int target_id = target_ids[position];
        if (target_id < 0 || (size_t)target_id >= model->vocab_size) {
            return NAN;
        }

        const float *row = logits + position * model->vocab_size;
        double maximum = row[0];
        for (size_t token_id = 1; token_id < model->vocab_size; token_id++) {
            if (row[token_id] > maximum) {
                maximum = row[token_id];
            }
        }

        /* log-sum-exp is stable even when logits have a large magnitude. */
        double exponential_sum = 0.0;
        for (size_t token_id = 0; token_id < model->vocab_size; token_id++) {
            exponential_sum += exp((double)row[token_id] - maximum);
        }
        total_loss += log(exponential_sum) + maximum - row[target_id];
    }

    return total_loss / (double)position_count;
}

int bigram_model_train_batch(BigramLanguageModel *model,
                             const int *input_ids, const int *target_ids,
                             size_t batch_size, size_t time_steps,
                             double learning_rate, double weight_decay) {
    if (model == NULL || model->token_embedding_table == NULL ||
        model->first_moment == NULL || model->second_moment == NULL ||
        input_ids == NULL || target_ids == NULL || batch_size == 0 ||
        time_steps == 0 || learning_rate <= 0.0 || weight_decay < 0.0) {
        return 0;
    }

    size_t position_count = batch_size * time_steps;
    size_t parameter_count = model->vocab_size * model->vocab_size;
    double *gradient = calloc(parameter_count, sizeof(*gradient));
    if (gradient == NULL) {
        return 0;
    }

    /*
     * For softmax cross-entropy, dLoss/dLogit = probability - one_hot(target).
     * Since an input ID selects one table row, accumulate into that row.
     */
    double gradient_scale = 1.0 / (double)position_count;
    for (size_t batch = 0; batch < batch_size; batch++) {
        for (size_t time = 0; time < time_steps; time++) {
            size_t position = batch * time_steps + time;
            int input_id = input_ids[position];
            int target_id = target_ids[position];
            if (input_id < 0 || target_id < 0 ||
                (size_t)input_id >= model->vocab_size ||
                (size_t)target_id >= model->vocab_size) {
                free(gradient);
                return 0;
            }

            size_t row_offset = (size_t)input_id * model->vocab_size;
            float *scores = model->token_embedding_table + row_offset;
            double maximum = scores[0];
            for (size_t token_id = 1; token_id < model->vocab_size; token_id++) {
                if (scores[token_id] > maximum) maximum = scores[token_id];
            }
            double exponential_sum = 0.0;
            for (size_t token_id = 0; token_id < model->vocab_size; token_id++) {
                exponential_sum += exp((double)scores[token_id] - maximum);
            }
            for (size_t token_id = 0; token_id < model->vocab_size; token_id++) {
                gradient[row_offset + token_id] +=
                    exp((double)scores[token_id] - maximum) /
                    exponential_sum * gradient_scale;
            }
            gradient[row_offset + (size_t)target_id] -= gradient_scale;
        }
    }

    model->optimizer_step++;
    const double beta1 = 0.9;
    const double beta2 = 0.999;
    const double epsilon = 1e-8;
    double beta1_correction = 1.0 - pow(beta1, (double)model->optimizer_step);
    double beta2_correction = 1.0 - pow(beta2, (double)model->optimizer_step);

    for (size_t index = 0; index < parameter_count; index++) {
        /* AdamW decouples weight decay from the gradient and momentum terms. */
        model->token_embedding_table[index] *= (float)(1.0 - learning_rate * weight_decay);
        model->first_moment[index] = (float)(
            beta1 * model->first_moment[index] + (1.0 - beta1) * gradient[index]);
        model->second_moment[index] = (float)(
            beta2 * model->second_moment[index] +
            (1.0 - beta2) * gradient[index] * gradient[index]);

        double corrected_first = model->first_moment[index] / beta1_correction;
        double corrected_second = model->second_moment[index] / beta2_correction;
        double update = learning_rate * corrected_first /
                        (sqrt(corrected_second) + epsilon);
        model->token_embedding_table[index] -= (float)update;
    }

    free(gradient);
    return 1;
}

void bigram_model_generate(BigramLanguageModel *model, int start_token,
                           size_t new_token_count, int *output_ids) {
    output_ids[0] = start_token;
    for (size_t generated = 0; generated < new_token_count; generated++) {
        int previous_id = output_ids[generated];
        const float *row = model->token_embedding_table +
                           (size_t)previous_id * model->vocab_size;
        double maximum = row[0];
        for (size_t token_id = 1; token_id < model->vocab_size; token_id++) {
            if (row[token_id] > maximum) {
                maximum = row[token_id];
            }
        }

        double probability_sum = 0.0;
        for (size_t token_id = 0; token_id < model->vocab_size; token_id++) {
            probability_sum += exp((double)row[token_id] - maximum);
        }

        double draw = next_uniform(model) * probability_sum;
        size_t chosen_id = model->vocab_size - 1;
        for (size_t token_id = 0; token_id < model->vocab_size; token_id++) {
            draw -= exp((double)row[token_id] - maximum);
            if (draw <= 0.0) {
                chosen_id = token_id;
                break;
            }
        }
        output_ids[generated + 1] = (int)chosen_id;
    }
}
