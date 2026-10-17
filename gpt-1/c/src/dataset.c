#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>

char *dataset_read(const char *path, size_t *byte_count) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        perror(path);
        return NULL;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        perror("Could not seek to the end of the dataset");
        fclose(file);
        return NULL;
    }

    long file_size = ftell(file);
    if (file_size < 0 || fseek(file, 0, SEEK_SET) != 0) {
        perror("Could not determine the dataset size");
        fclose(file);
        return NULL;
    }

    char *text = malloc((size_t)file_size + 1);
    if (text == NULL) {
        perror("Could not allocate memory for the dataset");
        fclose(file);
        return NULL;
    }

    size_t bytes_read = fread(text, sizeof(char), (size_t)file_size, file);
    if (ferror(file)) {
        perror("Could not read the dataset");
        free(text);
        fclose(file);
        return NULL;
    }

    text[bytes_read] = '\0';
    if (byte_count != NULL) {
        *byte_count = bytes_read;
    }

    fclose(file);
    return text;
}
