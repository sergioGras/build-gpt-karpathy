#include "vocabulary.h"

#include <stdlib.h>

void vocabulary_build(const char *text, size_t text_length, CharacterVocabulary *vocabulary) {
    vocabulary->size = 0;

    for (size_t value = 0; value < BYTE_VALUE_COUNT; value++) {
        vocabulary->byte_to_id[value] = -1;
    }

    /* Mark bytes that occur, then assign IDs in ascending byte order. */
    for (size_t index = 0; index < text_length; index++) {
        unsigned char byte = (unsigned char)text[index];
        vocabulary->byte_to_id[byte] = -2;
    }

    for (size_t value = 0; value < BYTE_VALUE_COUNT; value++) {
        if (vocabulary->byte_to_id[value] == -2) {
            vocabulary->byte_to_id[value] = (int)vocabulary->size;
            vocabulary->id_to_byte[vocabulary->size] = (unsigned char)value;
            vocabulary->size++;
        }
    }
}

int *vocabulary_encode(const char *text, size_t text_length,
                       const CharacterVocabulary *vocabulary, size_t *token_count) {
    int *token_ids = malloc((text_length == 0 ? 1 : text_length) * sizeof(*token_ids));
    if (token_ids == NULL) {
        return NULL;
    }

    for (size_t index = 0; index < text_length; index++) {
        unsigned char byte = (unsigned char)text[index];
        int id = vocabulary->byte_to_id[byte];
        if (id < 0) {
            free(token_ids);
            return NULL;
        }
        token_ids[index] = id;
    }

    if (token_count != NULL) {
        *token_count = text_length;
    }
    return token_ids;
}

char *vocabulary_decode(const int *token_ids, size_t token_count,
                        const CharacterVocabulary *vocabulary) {
    char *text = malloc(token_count + 1);
    if (text == NULL) {
        return NULL;
    }

    for (size_t index = 0; index < token_count; index++) {
        int id = token_ids[index];
        if (id < 0 || (size_t)id >= vocabulary->size) {
            free(text);
            return NULL;
        }
        text[index] = (char)vocabulary->id_to_byte[id];
    }

    text[token_count] = '\0';
    return text;
}
