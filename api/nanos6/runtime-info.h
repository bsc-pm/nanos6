/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_RUNTIME_INFO_H
#define NANOS6_RUNTIME_INFO_H

#include "major.h"


#include <stddef.h>


#pragma GCC visibility push(default)

#ifdef __cplusplus
extern "C" {
#endif


//! \brief type of contents on a nanos6_runtime_info_entry_t
typedef enum {
	nanos6_integer_runtime_info_entry,
	nanos6_real_runtime_info_entry,
	nanos6_text_runtime_info_entry
} nanos6_runtime_info_entry_type_t;


typedef struct {
	//! \brief name of the entry (single word)
	char const *name;
	
	//! \brief description of the entry (multiple words)
	char const *description;
	
	//! \brief type of the value
	nanos6_runtime_info_entry_type_t type;
	
	//! \brief the actual value
	union {
		long integer;
		double real;
		char const *text;
	};
	
	//! \brief units (if applicable)
	char const *units;
} nanos6_runtime_info_entry_t;


//! \brief obtain an iterator to the beginning of the runtime information entries
void *nanos6_runtime_info_begin(void);

//! \brief obtain an iterator past the end of the runtime information entries
void *nanos6_runtime_info_end(void);

//! \brief advance an iterator of the runtime information entries to the following element
void *nanos6_runtime_info_advance(void *runtimeInfoIterator);

//! \brief retrieve the runtime information entry pointed to by the iterator
void nanos6_runtime_info_get(void *runtimeInfoIterator, nanos6_runtime_info_entry_t *entry);


//! \brief format the value of a runtime info entry into a string
int nanos6_snprint_runtime_info_entry_value(char *str, size_t size, nanos6_runtime_info_entry_t const *entry);


#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif /* NANOS6_RUNTIME_INFO_H */
