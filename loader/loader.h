/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_LOADER_LOADER_H
#define NANOS6_LOADER_LOADER_H


__attribute__ ((visibility ("hidden"))) void _nanos6_loader(void);

__attribute__ ((visibility ("hidden"))) extern void *_nanos6_lib_handle;
__attribute__ ((visibility ("hidden"))) extern char const *_nanos6_lib_filename;


#endif // NANOS6_LOADER_LOADER_H
