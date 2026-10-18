#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    const char *part = argc == 1 ? "all" : argv[1];
    int first_step = 1;
    int last_step = 3;

    if (argc > 2) {
        fprintf(stderr, "Usage: gpt1_c [all|1|2|3|1-2|1-3]\n");
        return EXIT_FAILURE;
    }

    if (strcmp(part, "1") == 0) {
        last_step = 1;
    } else if (strcmp(part, "2") == 0) {
        first_step = last_step = 2;
    } else if (strcmp(part, "3") == 0) {
        first_step = last_step = 3;
    } else if (strcmp(part, "1-2") == 0) {
        last_step = 2;
    } else if (strcmp(part, "all") != 0 && strcmp(part, "1-3") != 0) {
        fprintf(stderr, "Usage: gpt1_c [all|1|2|3|1-2|1-3]\n");
        return EXIT_FAILURE;
    }

    const char *dataset_path = "../data/input.txt";
    FILE *file = fopen(dataset_path, "rb");
    if (file == NULL) {
        perror(dataset_path);
        return EXIT_FAILURE;
    }
    fclose(file);

    if (first_step <= 1) {
        puts("Step 1 - Find the shared corpus");
        printf("Dataset file: %s\n", dataset_path);
    }

    if (last_step == 1) {
        return EXIT_SUCCESS;
    }

    /* Later steps need the text even when their earlier output is hidden. */
    size_t byte_count = 0;
    char *text = dataset_read(dataset_path, &byte_count);

    if (text == NULL) {
        fprintf(stderr, "Could not read %s\n", dataset_path);
        return EXIT_FAILURE;
    }

    if (first_step <= 2) {
        puts("Step 2 - Read the corpus");
        puts("Corpus loaded into memory.");
    }

    if (last_step >= 3) {
        puts("Step 3 - Inspect the examples");
        printf("Dataset length in bytes: %zu\n", byte_count);
        puts("First 1000 bytes:");

        size_t preview_length = byte_count < 1000 ? byte_count : 1000;
        fwrite(text, sizeof(char), preview_length, stdout);
        putchar('\n');
    }

    free(text);
    return EXIT_SUCCESS;
}
