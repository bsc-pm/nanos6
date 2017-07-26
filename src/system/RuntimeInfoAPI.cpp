/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include "RuntimeInfo.hpp"

#include "api/nanos6/runtime-info.h"


union index_or_pointer_t {
	size_t _index;
	void *_pointer;
};


void *nanos6_runtime_info_begin(void)
{
	index_or_pointer_t result;
	
	result._index = 0;
	
	return result._pointer;
}


void *nanos6_runtime_info_end(void)
{
	index_or_pointer_t result;
	
	result._index = RuntimeInfo::size();
	
	return result._pointer;
}


void *nanos6_runtime_info_advance(void *runtimeInfoIterator)
{
	index_or_pointer_t result;
	
	result._pointer = runtimeInfoIterator;
	result._index++;
	
	return result._pointer;
}


void nanos6_runtime_info_get(void *runtimeInfoIterator, nanos6_runtime_info_entry_t *entry)
{
	assert(entry != nullptr);
	
	index_or_pointer_t iterator;
	
	iterator._pointer = runtimeInfoIterator;
	RuntimeInfo::getEntryContents(iterator._index, entry);
}



int nanos6_snprint_runtime_info_entry_value(char *str, size_t size, nanos6_runtime_info_entry_t const *entry)
{
	assert(entry != nullptr);
	
	switch (entry->type) {
		case nanos6_integer_runtime_info_entry:
			return snprintf(str, size, "%li", entry->integer);
		case nanos6_real_runtime_info_entry:
			return snprintf(str, size, "%f", entry->real);
		case nanos6_text_runtime_info_entry:
			return snprintf(str, size, "%s", entry->text);
	}
	
	return 0;
}

