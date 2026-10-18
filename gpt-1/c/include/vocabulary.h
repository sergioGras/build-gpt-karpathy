#ifndef GPT1_VOCABULARY_H
#define GPT1_VOCABULARY_H

#include <stddef.h>

#define BYTE_VALUE_COUNT 256

typedef struct {
    int byte_to_id[BYTE_VALUE_COUNT];
    unsigned char id_to_byte[BYTE_VALUE_COUNT];
    size_t size;
} CharacterVocabulary;

/* Build a sorted byte vocabulary from the supplied UTF-8 text bytes. */
void vocabulary_build(const char *text, size_t text_length, CharacterVocabulary *vocabulary);

/* The caller owns the returned token array and must free it. */
int *vocabulary_encode(const char *text, size_t text_length,
                       const CharacterVocabulary *vocabulary, size_t *token_count);

/* The caller owns the returned null-terminated text and must free it. */
char *vocabulary_decode(const int *token_ids, size_t token_count,
                        const CharacterVocabulary *vocabulary);

#endif
