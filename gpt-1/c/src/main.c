#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const char *dataset_path = "../data/input.txt";
    size_t byte_count = 0;
    char *text = dataset_read(dataset_path, &byte_count);

    if (text == NULL) {
        fprintf(stderr, "Could not read %s\n", dataset_path);
        return EXIT_FAILURE;
    }

    printf("Dataset file: %s\n", dataset_path);
    printf("Dataset length in bytes: %zu\n", byte_count);
    puts("First 1000 bytes:");

    size_t preview_length = byte_count < 1000 ? byte_count : 1000;
    fwrite(text, sizeof(char), preview_length, stdout);
    putchar('\n');

    free(text);
    return EXIT_SUCCESS;
}
