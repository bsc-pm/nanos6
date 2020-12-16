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

#ifndef INSTALLED_CONFIG_DIR
	#error "INSTALLED_CONFIG_DIR should be defined at make time"
#endif

_nanos6_loader_config_t _config;

char _nanos6_config_path[MAX_CONFIG_PATH];
char _nanos6_default_config_path[MAX_CONFIG_PATH];

extern char **environ;


static void _nanos6_init_config(void)
{
	_config.dependencies = NULL;
	_config.instrument = NULL;
	_config.library_path = NULL;
	_config.report_prefix = NULL;
	_config.debug = 0;
	_config.verbose = 0;
	_config.warn_envars = 1;
}

void _nanos6_loader_free_config(void)
{
	if (_config.dependencies)
		free(_config.dependencies);
	if (_config.instrument)
		free(_config.instrument);
	if (_config.library_path)
		free(_config.library_path);
	if (_config.report_prefix)
		free(_config.report_prefix);
}

// The following rules are followed to find the config file:
//   1. Check for NANOS6_CONFIG environment variable for the path. If there is a path but no file is found, stop with an error
//   2. Look for a nanos6.toml in the current directory
//   3. Look for a nanos6.toml in the installation path
//   4. If there still is no file, stop with an error
static int _nanos6_find_config(void)
{
	int cnt;
	const char *config_path = getenv("NANOS6_CONFIG");

	// Build the default config file path
	cnt = snprintf(_nanos6_default_config_path, MAX_CONFIG_PATH, "%s/scripts/nanos6.toml", INSTALLED_CONFIG_DIR);
	if (cnt >= MAX_CONFIG_PATH) {
		fprintf(stderr, "Error: The installation path for the default Nanos6 config file is too long.\n");
		return -1;
	}

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
	cnt = snprintf(_nanos6_config_path, MAX_CONFIG_PATH, "%s/nanos6.toml", current_path);
	free((void *) current_path);

	if (cnt >= MAX_CONFIG_PATH) {
		fprintf(stderr, "Warning: The current working path is too long, if there is a config file in the current directory it will not be used.\n");
	} else {
		if (!access(_nanos6_config_path, R_OK)) {
			// Found file in current path
			return 0;
		}
	}

	// 3. Default config path (installation)
	strncpy(_nanos6_config_path, _nanos6_default_config_path, MAX_CONFIG_PATH);

	if (!access(_nanos6_config_path, R_OK)) {
		return 0;
	}

	fprintf(stderr, "Error: Failed to find the Nanos6 config file.\n");
	return -1;
}

static int _toml_try_extract_string(toml_table_t *section, char **output, const char *key)
{
	toml_raw_t raw;

	raw = toml_raw_in(section, key);
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

static int _toml_try_extract_bool(toml_table_t *section, int *output, const char *key)
{
	toml_raw_t raw;

	raw = toml_raw_in(section, key);
	if (raw != NULL) {
		if (toml_rtob(raw, output)) {
			// Failure to cast
			return -1;
		}
	}

	return 0;
}

static int _nanos6_parse_config_tables(toml_table_t *loader_section, toml_table_t *version_section)
{
	if (loader_section) {
		if (_toml_try_extract_string(loader_section, &_config.library_path, "library_path"))
			return -1;
		if (_toml_try_extract_string(loader_section, &_config.report_prefix, "report_prefix"))
			return -1;
		if (_toml_try_extract_bool(loader_section, &_config.verbose, "verbose"))
			return -1;
		if (_toml_try_extract_bool(loader_section, &_config.warn_envars, "warn_envars"))
			return -1;
	}

	if (version_section) {
		if (_toml_try_extract_string(version_section, &_config.dependencies, "dependencies"))
			return -1;
		if (_toml_try_extract_string(version_section, &_config.instrument, "instrument"))
			return -1;
		if (_toml_try_extract_bool(version_section, &_config.debug, "debug"))
			return -1;
	}

	return 0;
}

static int _nanos6_config_parse_individual_override(const char *name, const char *value)
{
	if (strlen(name) == 0 || strlen(value) == 0)
		return 0;

	if (strcmp(name, "version.dependencies") == 0) {
		if (_config.dependencies)
			free(_config.dependencies);
		_config.dependencies = strdup(value);
	} else if (strcmp(name, "version.instrument") == 0) {
		if (_config.instrument)
			free(_config.instrument);
		_config.instrument = strdup(value);
	} else if (strcmp(name, "loader.library_path") == 0) {
		if (_config.library_path)
			free(_config.library_path);
		_config.library_path = strdup(value);
	} else if (strcmp(name, "loader.report_prefix") == 0) {
		if (_config.report_prefix)
			free(_config.report_prefix);
		_config.report_prefix = strdup(value);
	} else if (strcmp(name, "version.debug") == 0) {
		if (strcmp(value, "true") == 0) {
			_config.debug = 1;
		} else if (strcmp(value, "false") == 0) {
			_config.debug = 0;
		} else {
			fprintf(stderr, "Error: Bad value for %s override\n", name);
			return -1;
		}
	} else if (strcmp(name, "loader.verbose") == 0) {
		if (strcmp(value, "true") == 0) {
			_config.verbose = 1;
		} else if (strcmp(value, "false") == 0) {
			_config.verbose = 0;
		} else {
			fprintf(stderr, "Error: Bad value for %s override\n", name);
			return -1;
		}
	} else if (strcmp(name, "loader.warn_envars") == 0) {
		if (strcmp(value, "true") == 0) {
			_config.warn_envars = 1;
		} else if (strcmp(value, "false") == 0) {
			_config.warn_envars = 0;
		} else {
			fprintf(stderr, "Error: Bad value for %s override\n", name);
			return -1;
		}
	}
	return 0;
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
			if (_nanos6_config_parse_individual_override(current_option, separator + 1))
				return -1;
			*separator = '=';
		} // Otherwise maybe an invalid or empty variable, but more sanity checks are done on the runtime initialization

		current_option = strtok(NULL, ",");
	}

	free(nstring);

	return 0;
}

static void _nanos6_loader_check_envars(void)
{
	if (!_config.warn_envars)
		return;

	const char *prefix = "NANOS6";
	const int plen = strlen(prefix);

	const int nsuffixes = 3;
	const char *suffix[3] = { "_CONFIG=", "_CONFIG_OVERRIDE=", "_HOME=" };
	int suffixlen[3];

	for (int s = 0; s < nsuffixes; ++s) {
		suffixlen[s] = strlen(suffix[s]);
	}

	int var = 0, warn = 0;
	while (environ[var]) {
		if (strncmp(environ[var], prefix, plen) == 0) {
			int found = 0;
			for (int s = 0; !found && s < nsuffixes; ++s) {
				found = (strncmp(environ[var]+plen, suffix[s], suffixlen[s]) == 0);
			}

			if (!found) {
				if (!warn)
					fprintf(stderr, "Warning: Irrelevant NANOS6 environment variables detected! The following variables are ignored:\n\t");
				warn = 1;

				char *last = strchr(environ[var], '=');
				if (last == NULL) {
					fprintf(stderr, "%s  ", environ[var]);
				} else {
					fprintf(stderr, "%.*s  ", (int) (last - environ[var]), environ[var]);
				}
			}
		}
		++var;
	}

    if (warn) {
		fprintf(stderr, "\n\n");
		fprintf(stderr, "From now on, the behavior of the Nanos6 runtime can be tuned using a configuration file in TOML format.\n");
		fprintf(stderr, "The default configuration file is located at the documentation directory of the Nanos6 installation.\n");
		fprintf(stderr, "In this installation, the default configuration file is:\n\t%s\n\n", _nanos6_default_config_path);

		fprintf(stderr, "We recommend to take a look at the configuration file to see the different options that Nanos6 provides.\n");
		fprintf(stderr, "Additionally, we recommend to copy that default file to your directory and change the options that you want to override.\n");
		fprintf(stderr, "The Nanos6 runtime will only interpret the first configuration file found according to the following order:\n");
		fprintf(stderr, "\t1. The file pointed by the NANOS6_CONFIG environment variable.\n");
		fprintf(stderr, "\t2. The nanos6.toml file found in the current working directory.\n");
		fprintf(stderr, "\t3. The nanos6.toml file found in the installation path (default file).\n\n");

		fprintf(stderr, "For your information, you are currently using the configuration file:\n\t%s\n\n", _nanos6_config_path);

		fprintf(stderr, "Alternatively, you can override configuration options using the NANOS6_CONFIG_OVERRIDE environment variable.\n");
		fprintf(stderr, "The contents of this variable have to follow the format \"key1=value1,key2=value2,key3=value3,...\".\n");
		fprintf(stderr, "For instance, you can execute the command below to run the CTF instrumentation with discrete dependencies:\n");
		fprintf(stderr, "\tNANOS6_CONFIG_OVERRIDE=\"version.dependencies=discrete,version.instrument=ctf\" ./ompss-2-program\n\n");

		fprintf(stderr, "Therefore, the only relevant NANOS6 variables are NANOS6_CONFIG and NANOS6_CONFIG_OVERRIDE; the rest are ignored by the runtime.\n");
		fprintf(stderr, "For more information, please check the OmpSs-2 User Guide (https://pm.bsc.es/ftp/ompss-2/doc/user-guide).\n");
		fprintf(stderr, "Note that you can disable this warning by setting the option 'loader.warn_envars' to false.\n\n");
    }
}

// Find and parse the Nanos6 configuration file
// The file used (path) is shared with the runtime. However, the runtime will parse independently the file
// This is because we don't know here which are the expected data types of each variable
// Additionally, every config option used by the loader is included in the [loader] section
int _nanos6_loader_parse_config(void)
{
	char errbuf[200];

	_nanos6_init_config();

	if (_nanos6_find_config()) {
		return -1;
	}

	// Open found config file for reading
	FILE *f = fopen(_nanos6_config_path, "r");
	if (f == NULL) {
		fprintf(stderr, "Error: Failed to open config file for reading: %s\n", strerror(errno));
		return -1;
	}

	// Parse the file
	toml_table_t *conf = toml_parse_file(f, errbuf, sizeof(errbuf));
	fclose(f);

	if (conf == NULL) {
		fprintf(stderr, "Error: Failed to parse config file: %s\n", errbuf);
		return -1;
	}

	// Find the [loader] section in the file
	toml_table_t *loader_section = toml_table_in(conf, "loader");
	toml_table_t *version_section = toml_table_in(conf, "version");
	if (loader_section || version_section) {
		if (_nanos6_parse_config_tables(loader_section, version_section)) {
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

	// Print the configuration file path if the loader is in verbose mode
	if (_config.verbose)
		fprintf(stderr, "Nanos6 loader parsed configuration from file: %s\n", _nanos6_config_path);

	// Check if there are irrelevant Nanos6 envars defined
	_nanos6_loader_check_envars();

	return 0;
}
