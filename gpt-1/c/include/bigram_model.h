#ifndef GPT1_BIGRAM_MODEL_H
#define GPT1_BIGRAM_MODEL_H

#include <stddef.h>
#include <stdint.h>

/*
 * A table with one row per input character and one column per possible next
 * character. The contiguous storage is row-major: weights[input_id * V + next_id].
 */
typedef struct {
    size_t vocab_size;
    float *token_embedding_table;
    float *first_moment;
    float *second_moment;
    size_t optimizer_step;
    uint32_t random_state;
} BigramLanguageModel;

int bigram_model_init(BigramLanguageModel *model, size_t vocab_size, uint32_t seed);
void bigram_model_destroy(BigramLanguageModel *model);

/* Inputs have shape [batch_size, time_steps], logits have [B, T, vocab_size]. */
int bigram_model_forward(const BigramLanguageModel *model, const int *input_ids,
                         size_t batch_size, size_t time_steps, float *logits);

/* Mean negative log-likelihood over every target position. */
double bigram_model_cross_entropy(const BigramLanguageModel *model,
                                  const float *logits, const int *target_ids,
                                  size_t batch_size, size_t time_steps);

/* One dense AdamW update; C computes the embedding-table gradient explicitly. */
int bigram_model_train_batch(BigramLanguageModel *model,
                             const int *input_ids, const int *target_ids,
                             size_t batch_size, size_t time_steps,
                             double learning_rate, double weight_decay);

/* output_ids must hold new_token_count + 1 values; the start token is included. */
void bigram_model_generate(BigramLanguageModel *model, int start_token,
                           size_t new_token_count, int *output_ids);

#endif
