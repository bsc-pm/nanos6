#ifndef NANOS6_UTILS_H
#define NANOS6_UTILS_H

enum nanos6_utils_api_t { nanos6_utils_api = 1 };

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif


//! \brief Fill up a buffer with zeros
void nanos6_bzero(void *buffer, size_t size);


#ifdef __cplusplus
}
#endif


#endif /* NANOS6_UTILS_H */
