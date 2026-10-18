#include "dataset.h"
#include "vocabulary.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_part(const char *part, int *first_step, int *last_step) {
    *first_step = 1;
    *last_step = 3;

    if (strcmp(part, "1") == 0) {
        *last_step = 1;
    } else if (strcmp(part, "2") == 0) {
        *first_step = *last_step = 2;
    } else if (strcmp(part, "3") == 0) {
        *first_step = *last_step = 3;
    } else if (strcmp(part, "1-2") == 0) {
        *last_step = 2;
    } else if (strcmp(part, "all") != 0 && strcmp(part, "1-3") != 0) {
        return 0;
    }
    return 1;
}

static void print_token_ids(const int *token_ids, size_t token_count) {
    for (size_t index = 0; index < token_count; index++) {
        printf("%s%d", index == 0 ? "" : " ", token_ids[index]);
    }
    putchar('\n');
}

int main(int argc, char *argv[]) {
    const char *part = argc == 1 ? "all" : argv[1];
    int first_step;
    int last_step;

    if (argc > 2 || !parse_part(part, &first_step, &last_step)) {
        fprintf(stderr, "Usage: gpt1_tokenization [all|1|2|3|1-2|1-3]\n");
        return EXIT_FAILURE;
    }

    const char *dataset_path = "../data/input.txt";
    size_t text_length = 0;
    char *text = dataset_read(dataset_path, &text_length);
    if (text == NULL) {
        fprintf(stderr, "Could not read %s\n", dataset_path);
        return EXIT_FAILURE;
    }

    CharacterVocabulary vocabulary;
    vocabulary_build(text, text_length, &vocabulary);

    if (first_step <= 1) {
        printf("Step 1 - Build the character vocabulary\n");
        printf("Vocabulary size: %zu\n", vocabulary.size);
        printf("Characters in ID order: \"");
        for (size_t id = 0; id < vocabulary.size; id++) {
            unsigned char byte = vocabulary.id_to_byte[id];
            if (byte == '\n') {
                fputs("\\n", stdout);
            } else {
                putchar(byte);
            }
        }
        puts("\"");
    }

    if (first_step <= 2 && last_step >= 2) {
        const char *sample = "hii there";
        size_t sample_length = strlen(sample);
        size_t sample_token_count = 0;
        int *sample_ids = vocabulary_encode(sample, sample_length, &vocabulary, &sample_token_count);
        if (sample_ids == NULL) {
            fprintf(stderr, "Could not encode the toy example.\n");
            free(text);
            return EXIT_FAILURE;
        }

        char *decoded_sample = vocabulary_decode(sample_ids, sample_token_count, &vocabulary);
        if (decoded_sample == NULL) {
            fprintf(stderr, "Could not decode the toy example.\n");
            free(sample_ids);
            free(text);
            return EXIT_FAILURE;
        }

        puts("Step 2 - Encode and decode a toy example");
        printf("Input text: %s\n", sample);
        printf("Token IDs: ");
        print_token_ids(sample_ids, sample_token_count);
        printf("Decoded text: %s\n", decoded_sample);
        printf("Round trip matches: %s\n", strcmp(sample, decoded_sample) == 0 ? "true" : "false");

        free(decoded_sample);
        free(sample_ids);
    }

    if (last_step >= 3) {
        size_t token_count = 0;
        int *token_ids = vocabulary_encode(text, text_length, &vocabulary, &token_count);
        if (token_ids == NULL) {
            fprintf(stderr, "Could not encode the full corpus.\n");
            free(text);
            return EXIT_FAILURE;
        }

        char *decoded_text = vocabulary_decode(token_ids, token_count, &vocabulary);
        if (decoded_text == NULL) {
            fprintf(stderr, "Could not decode the full corpus.\n");
            free(token_ids);
            free(text);
            return EXIT_FAILURE;
        }

        size_t preview_length = token_count < 100 ? token_count : 100;
        puts("Step 3 - Encode the full corpus");
        printf("Token count: %zu\n", token_count);
        printf("First %zu token IDs: ", preview_length);
        print_token_ids(token_ids, preview_length);
        printf("Full text round trip matches: %s\n",
               memcmp(text, decoded_text, text_length) == 0 ? "true" : "false");

        free(decoded_text);
        free(token_ids);
    }

    free(text);
    return EXIT_SUCCESS;
}
