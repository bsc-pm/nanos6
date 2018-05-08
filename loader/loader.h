/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_LOADER_LOADER_H
#define NANOS6_LOADER_LOADER_H


#define ERROR_TEXT_SIZE 8192

__attribute__ ((visibility ("hidden"))) void _nanos6_loader(void);

__attribute__ ((visibility ("hidden"))) extern void *_nanos6_lib_handle;

__attribute__ ((visibility ("hidden"))) extern int _nanos6_has_started;
extern int _nanos6_exit_with_error;
extern char _nanos6_error_text[ERROR_TEXT_SIZE];


#endif // NANOS6_LOADER_LOADER_H
