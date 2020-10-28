/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "config-parser.h"
#include "loader.h"
#include "support/toml/toml.h"

#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

char _nanos6_config_path[MAX_CONFIG_PATH];
_nanos6_loader_config_t _config;

#ifndef INSTALLED_CONFIG_DIR
	#error "INSTALLED_CONFIG_DIR should be defined at make time"
#endif

static void _nanos6_init_config(void)
{
	_config.dependencies = NULL;
	_config.library_path = NULL;
	_config.report_prefix = NULL;
	_config.variant = NULL;
	_config.verbose = 0;
}

void _nanos6_loader_free_config(void)
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
//   1. Check for NANOS6_CONFIG environment variable for the path. If there is a path but no file is found, stop with an error
//   2. Look for a nanos6.toml in the current directory
//   3. Look for a nanos6.toml in the installation path
//   4. If there still is no file, stop with an error
static int _nanos6_find_config(void)
{
	const char *config_path = getenv("NANOS6_CONFIG");

	// 1. NANOS6_CONFIG
	if (config_path != NULL) {
		// Can we access the file for reading?
		if (access(config_path, R_OK)) {
			// We cannot. Lets print the error by stderr and then die
			fprintf(stderr, "Error: Failed to find the file specified in NANOS6_CONFIG: %s\n", strerror(errno));
			return -1;
		}

		// Greater or equal strict to account for the null character
		if (strlen(config_path) >= MAX_CONFIG_PATH) {
			fprintf(stderr, "Error: Path specified in NANOS6_CONFIG is too long.\n");
			return -1;
		}

		strncpy(_nanos6_config_path, config_path, MAX_CONFIG_PATH);

		return 0;
	}

	// 2. Current directory
	if (getcwd(_nanos6_config_path, MAX_CONFIG_PATH) == NULL) {
		fprintf(stderr, "Error: Failed to get current working directory: %s\n", strerror(errno));
		return -1;
	}

	const char *current_path = strdup(_nanos6_config_path);
	snprintf(_nanos6_config_path, MAX_CONFIG_PATH, "%s/nanos6.toml", current_path);
	free((void *) current_path);

	if (!access(_nanos6_config_path, R_OK)) {
		// Found file in current path
		return 0;
	}

	// 3. Installation path
	snprintf(_nanos6_config_path, MAX_CONFIG_PATH, "%s/scripts/nanos6.toml", INSTALLED_CONFIG_DIR);
	if (!access(_nanos6_config_path, R_OK)) {
		return 0;
	}

	fprintf(stderr, "Error: Failed to find the Nanos6 config file.\n");
	return -1;
}

static int _toml_try_extract_string(toml_table_t *loader_section, char **output, const char *key)
{
	toml_raw_t raw;

	raw = toml_raw_in(loader_section, key);
	if (raw != NULL) {
		if (toml_rtos(raw, output)) {
			// Failure to cast
			assert(*output == NULL);
			return -1;
		}

		// Empty strings count as null
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
			// Failure to cast
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

static void _nanos6_config_parse_individual_override(const char *name, const char *value)
{
	if (strlen(name) == 0 || strlen(value) == 0)
		return;

	if (strcmp(name, "loader.dependencies") == 0) {
		if (_config.dependencies)
			free(_config.dependencies);
		_config.dependencies = strdup(value);
	} else if (strcmp(name, "loader.library_path") == 0) {
		if (_config.library_path)
			free(_config.library_path);
		_config.library_path = strdup(value);
	} else if (strcmp(name, "loader.variant") == 0) {
		if (_config.variant)
			free(_config.variant);
		_config.variant = strdup(value);
	} else if (strcmp(name, "loader.report_prefix") == 0) {
		if (_config.report_prefix)
			free(_config.report_prefix);
		_config.report_prefix = strdup(value);
	} else if (strcmp(name, "loader.verbose") == 0) {
		if (strcmp(value, "true") == 0)
			_config.verbose = 1;
		else if (strcmp(value, "false") == 0)
			_config.verbose = 0;
		else
			fprintf(stderr, "Bad value for loader.verbose override");
	}
}

static int _nanos6_config_parse_override(void)
{
	const char *config_override = getenv("NANOS6_CONFIG_OVERRIDE");

	if (config_override == NULL || strlen(config_override) == 0)
		return 0;

	char *nstring = strdup(config_override);
	char *current_option = strtok(nstring, ",");

	while (current_option) {
		// Let's extract the name and the value
		char *separator = strchr(current_option, '=');
		if (separator) {
			// We can "cheat" by creating two strings from the single place we have. As the separator is not important,
			// we just replace it by a null character
			*separator = '\0';
			_nanos6_config_parse_individual_override(current_option, separator + 1);
			*separator = '=';
		} // Otherwise maybe an invalid or empty variable, but more sanity checks are done on the runtime initialization

		current_option = strtok(NULL, ",");
	}

	free(nstring);
}

// Find and parse the Nanos6 configuration file
// The file used (path) is shared with the runtime. However, the runtime will parse independently the file
// This is because we don't know here which are the expected data types of each variable
// Additionally, every config option used by the loader is included in the [loader] section
int _nanos6_loader_parse_config(void)
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
		fprintf(stderr, "Error: Failed to open config file for reading: %s", strerror(errno));
		return -1;
	}

	// Parse the file
	conf = toml_parse_file(f, errbuf, sizeof(errbuf));
	fclose(f);
	if (conf == NULL) {
		fprintf(stderr, "Error: Failed to parse config file: %s", errbuf);
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

	// Now parse the configuration overrides
	if (_nanos6_config_parse_override()) {
		_nanos6_loader_free_config();
		return -1;
	}

	return 0;
}
