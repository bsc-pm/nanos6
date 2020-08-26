/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "config-parser.h"
#include "loader.h"
#include "include/toml/toml.h"

#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

char _nanos6_config_path[MAX_CONFIG_PATH];
_nanos6_loader_config_t _config;

static void _nanos6_init_config()
{
	_config.dependencies = NULL;
	_config.library_path = NULL;
	_config.report_prefix = NULL;
	_config.variant = NULL;

	_config.verbose = 1;
}

void _nanos6_loader_free_config()
{
	if (_config.dependencies)
		free(_config.dependencies);

	if (_config.library_path)
		free(_config.library_path);

	if (_config.report_prefix)
		free(_config.report_prefix);

	if (_config.variant)
		free(_config.variant);
}

// The following rules are followed to find the config file:
//	 1. Check for NANOS6_CONFIG environment variable for the path. If there is a path but no file is found, stop with an error.
//   2. Look for a nanos6.toml in the current directory.
//   3. Look for a nanos6.toml in the installation path.
//	 4. If there still is no file, stop with an error.
static int _nanos6_find_config()
{
	char *config_path = getenv("NANOS6_CONFIG");

	// 1. NANOS6_CONFIG
	if (config_path != NULL) {
		// Can we access the file for reading?
		if (access(config_path, R_OK)) {
			// We cannot. Lets print the error by stderr and then die.
			snprintf(_nanos6_error_text, ERROR_TEXT_SIZE, "Failed to find the file specified in NANOS6_CONFIG: %s", sys_errlist[errno]);
			return -1;
		}

		// Greater or equal strict to account for the null character
		if (strlen(config_path) >= MAX_CONFIG_PATH) {
			snprintf(_nanos6_error_text, ERROR_TEXT_SIZE, "Path specified in NANOS6_CONFIG is too long");
			return -1;
		}

		strncpy(_nanos6_config_path, config_path, MAX_CONFIG_PATH);

		return 0;
	}

	// 2. Current directory
	if (getcwd(_nanos6_config_path, MAX_CONFIG_PATH) == NULL) {
		snprintf(_nanos6_error_text, ERROR_TEXT_SIZE, "Failed to get current working directory: %s", sys_errlist[errno]);
		return -1;
	}

	const char *current_path = strdup(_nanos6_config_path);
	snprintf(_nanos6_config_path, MAX_CONFIG_PATH, "%s/nanos6.toml", current_path);
	free(current_path);

	if (!access(_nanos6_config_path, R_OK)) {
		// Found file in current path.
		return 0;
	}

	// 3. Installation path. But what is the installation path?

	snprintf(_nanos6_error_text, ERROR_TEXT_SIZE, "Failed to find the Nanos6 config file.");
	return -1;
}

static int _toml_try_extract_string(toml_table_t *loader_section, char **output, const char *key)
{
	toml_raw_t raw;

	raw = toml_raw_in(loader_section, key);
	if (raw != NULL) {
		if (toml_rtos(raw, output)) {
			// Failure to cast.
			assert(*output == NULL);
			return -1;
		}

		// Empty strings count as null.
		if (strcmp(*output, "") == 0) {
			free(*output);
			*output = NULL;
		}
	}

	return 0;
}

static int _toml_try_extract_bool(toml_table_t *loader_section, int *output, const char *key)
{
	toml_raw_t raw;

	raw = toml_raw_in(loader_section, key);
	if (raw != NULL) {
		if (toml_rtob(raw, output)) {
			// Failure to cast.
			return -1;
		}
	}

	return 0;

}

static int _nanos6_parse_config_table(toml_table_t *loader_section)
{
	if (_toml_try_extract_string(loader_section, &_config.dependencies, "dependencies"))
		return -1;

	if (_toml_try_extract_string(loader_section, &_config.library_path, "library_path"))
		return -1;

	if (_toml_try_extract_string(loader_section, &_config.variant, "variant"))
		return -1;

	if (_toml_try_extract_string(loader_section, &_config.report_prefix, "report_prefix"))
		return -1;

	if (_toml_try_extract_bool(loader_section, &_config.verbose, "verbose"))
		return -1;

	return 0;
}

// Find and parse the Nanos6 configuration file
// The file used (path) is shared with the runtime. However, the runtime will parse independently the file.
// This is because we don't know here which are the expected data types of each variable.
// Additionally, every config option used by the loader is included in the [loader] section.
int _nanos6_loader_parse_config()
{
	FILE *f;
	toml_table_t *conf;
	char errbuf[200];

	_nanos6_init_config();

	if (_nanos6_find_config()) {
		return -1;
	}

	// Open found config file for reading
	f = fopen(_nanos6_config_path, "r");
	if (f == NULL) {
		snprintf(_nanos6_error_text, ERROR_TEXT_SIZE, "Failed to open config file for reading: %s", sys_errlist[errno]);
		return -1;
	}

	// Parse the file
	conf = toml_parse_file(f, errbuf, sizeof(errbuf));
	fclose(f);
	if (conf == NULL) {
		snprintf(_nanos6_error_text, ERROR_TEXT_SIZE, "Failed to parse config file: %s", errbuf);
		return -1;
	}

	// Find the [loader] section in the file
	toml_table_t *loader_section = toml_table_in(conf, "loader");
	if (loader_section) {
		if (_nanos6_parse_config_table(loader_section)) {
			_nanos6_loader_free_config();
			toml_free(conf);
			return -1;
		}
	}

	toml_free(conf);

	return 0;
}
