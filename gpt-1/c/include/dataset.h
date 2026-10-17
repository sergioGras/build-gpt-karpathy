#ifndef GPT1_DATASET_H
#define GPT1_DATASET_H

#include <stddef.h>

/* Read a whole file into a newly allocated, null-terminated buffer.
   The caller owns the returned buffer and must free it. */
char *dataset_read(const char *path, size_t *byte_count);

#endif
