/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_LOADER_CONFIG_PARSER_H
#define NANOS6_LOADER_CONFIG_PARSER_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <limits.h>

#define MAX_CONFIG_PATH PATH_MAX

int _nanos6_loader_parse_config(void);
void _nanos6_loader_free_config(void);

typedef struct {
	char *dependencies;
	char *variant;
	char *library_path;
	char *report_prefix;
	int verbose;
} _nanos6_loader_config_t;

extern _nanos6_loader_config_t _config;
extern char _nanos6_config_path[MAX_CONFIG_PATH];

#endif // NANOS6_LOADER_CONFIG_PARSER_H