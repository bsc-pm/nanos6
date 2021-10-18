/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#include <babeltrace2/babeltrace.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include <linux/limits.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#define GRAPH_LOG_LEVEL BT_LOGGING_LEVEL_ERROR

#ifndef PRV_LIB_PATH
#define PRV_LIB_PATH "."
#endif

#define MAX_FILTER 16

#define err(...) fprintf(stderr, __VA_ARGS__);

static bt_graph *
create_graph(const char *input_trace, const char *output_dir,
		int split_events, long *filter_events, int num_filters,
	       	int quiet)
{
	bt_graph *graph;
	const bt_component_class_source *source_class;
	const bt_component_class_sink *sink_class;
	const bt_component_class_filter *muxer_class;

	const bt_component_source *source;
	const bt_component_filter *muxer;
	const bt_component_sink *sink;

	bt_value *source_params, *inputs_array;
	bt_value *sink_params;

	const bt_plugin *ctf_plugin, *utils_plugin, *prv_plugin;
	const bt_plugin_set *prv_plugin_set;

	const bt_port_output *source_out;
	const bt_port_input *sink_in;

	const bt_port_output *muxer_out;
	const bt_port_input *muxer_in;

	const bt_connection *connection;
	uint64_t nports, i, nplugins;
	int ret;

	graph = bt_graph_create(bt_get_maximal_mip_version());

	if(!graph)
	{
		err("bt_graph_create failed\n");
		exit(EXIT_FAILURE);
	}


	if(bt_plugin_find("ctf",
			BT_FALSE, BT_FALSE, BT_TRUE, BT_TRUE, BT_TRUE,
			&ctf_plugin) != BT_PLUGIN_FIND_STATUS_OK)
	{
		err("bt_plugin_find failed for 'ctf'\n");
		exit(EXIT_FAILURE);
	}

	if(bt_plugin_find("utils",
			BT_FALSE, BT_FALSE, BT_TRUE, BT_TRUE, BT_TRUE,
			&utils_plugin) != BT_PLUGIN_FIND_STATUS_OK)
	{
		err("bt_plugin_find failed for 'utils'\n");
		exit(EXIT_FAILURE);
	}

	if((ret = bt_plugin_find_all_from_file(PRV_LIB_PATH "/libprv.so",
				BT_TRUE,
				&prv_plugin_set))
			!= BT_PLUGIN_FIND_ALL_FROM_FILE_STATUS_OK)
	{
		err("bt_plugin_find_all_from_file failed for 'libprv.so'\n");
		err("the search path is %s\n", PRV_LIB_PATH "/libprv.so");
		err("ret code = %d\n", ret);
		exit(EXIT_FAILURE);
	}

	nplugins = bt_plugin_set_get_plugin_count(prv_plugin_set);
	assert(nplugins == 1);

	prv_plugin = bt_plugin_set_borrow_plugin_by_index_const(prv_plugin_set, 0);


	source_class = bt_plugin_borrow_source_component_class_by_name_const(
			ctf_plugin, "fs");

	if(!source_class)
	{
		err("getting source class failed\n");
		exit(EXIT_FAILURE);
	}

	sink_class = bt_plugin_borrow_sink_component_class_by_name_const(
			prv_plugin, "prv");

	if(!sink_class)
	{
		err("getting sink class failed\n");
		exit(EXIT_FAILURE);
	}

	muxer_class = bt_plugin_borrow_filter_component_class_by_name_const(
			utils_plugin, "muxer");

	if(!muxer_class)
	{
		err("getting muxer class failed\n");
		exit(EXIT_FAILURE);
	}

	/* Add the source to the graph */

	source_params = bt_value_map_create();
	assert(source_params);

	inputs_array = bt_value_array_create();
	assert(inputs_array);

	bt_value_array_append_string_element(inputs_array, input_trace);

	if(bt_value_map_insert_entry(source_params, "inputs", inputs_array)
			!= BT_VALUE_MAP_INSERT_ENTRY_STATUS_OK)
	{
		err("bt_value_map_insert_entry failed\n");
		exit(EXIT_FAILURE);
	}

	if(bt_graph_add_source_component(graph, source_class,
			"source.ctf.fs", source_params,
			GRAPH_LOG_LEVEL, &source)
			!= BT_GRAPH_ADD_COMPONENT_STATUS_OK)
	{
		err("adding source to the graph failed\n");
		exit(EXIT_FAILURE);
	}

	/* Add the muxer to the graph */

	if(bt_graph_add_filter_component(graph, muxer_class,
			"filter.utils.muxer", NULL,
			GRAPH_LOG_LEVEL, &muxer)
			!= BT_GRAPH_ADD_COMPONENT_STATUS_OK)
	{
		err("adding muxer to the graph failed\n");
		exit(EXIT_FAILURE);
	}

	/* Add the sink to the graph */

	sink_params = bt_value_map_create();
	assert(sink_params);

	if(bt_value_map_insert_string_entry(sink_params,
				"output_dir", output_dir)
			!= BT_VALUE_MAP_INSERT_ENTRY_STATUS_OK)
	{
		err("bt_value_map_insert_string_entry failed\n");
		exit(EXIT_FAILURE);
	}

	if(bt_value_map_insert_bool_entry(sink_params,
				"split_events", split_events)
			!= BT_VALUE_MAP_INSERT_ENTRY_STATUS_OK)
	{
		err("bt_value_map_insert_bool_entry failed\n");
		exit(EXIT_FAILURE);
	}

	if(bt_value_map_insert_bool_entry(sink_params,
				"quiet", quiet)
			!= BT_VALUE_MAP_INSERT_ENTRY_STATUS_OK)
	{
		err("bt_value_map_insert_bool_entry failed\n");
		exit(EXIT_FAILURE);
	}

	bt_value *filters = bt_value_array_create();
	assert(filters);

	for (int i = 0; i < num_filters; ++i) {
		if(bt_value_array_append_signed_integer_element(filters,
			filter_events[i])
			!= BT_VALUE_ARRAY_APPEND_ELEMENT_STATUS_OK)
		{
			err("bt_value_array_append_signed_integer_element failed\n");
			exit(EXIT_FAILURE);
		}
	}

	if(bt_value_map_insert_entry(sink_params,
				"filters", filters)
			!= BT_VALUE_MAP_INSERT_ENTRY_STATUS_OK)
	{
		err("bt_value_map_insert_entry failed\n");
		exit(EXIT_FAILURE);
	}

	if(bt_graph_add_sink_component(graph, sink_class,
			"sink.ctf.fs", sink_params,
			GRAPH_LOG_LEVEL, &sink)
			!= BT_GRAPH_ADD_COMPONENT_STATUS_OK)
	{
		err("adding sink to the graph failed\n");
		exit(EXIT_FAILURE);
	}

	/* Connect all input ports to the muxer */

	nports = bt_component_source_get_output_port_count(source);

	for(i=0; i<nports; i++)
	{
		source_out = bt_component_source_borrow_output_port_by_index_const(
				source, i);

		if(!source_out)
		{
			err("getting output port from source failed\n");
			exit(EXIT_FAILURE);
		}

		muxer_in = bt_component_filter_borrow_input_port_by_index_const(
				muxer, i);

		if(!muxer_in)
		{
			err("getting input port from muxer failed\n");
			exit(EXIT_FAILURE);
		}

		if(bt_graph_connect_ports(graph, source_out, muxer_in,
					&connection)
				!= BT_GRAPH_CONNECT_PORTS_STATUS_OK)
		{
			err("bt_graph_connect_ports failed\n");
			exit(EXIT_FAILURE);
		}
	}

	/* Connect the muxer output to the sink */

	muxer_out = bt_component_filter_borrow_output_port_by_index_const(
			muxer, 0UL);

	if(!muxer_out)
	{
		err("getting muxer output port failed\n");
		exit(EXIT_FAILURE);
	}

	sink_in = bt_component_sink_borrow_input_port_by_index_const(
			sink, 0UL);

	if(!sink_in)
	{
		err("getting input port failed\n");
		exit(EXIT_FAILURE);
	}

	if(bt_graph_connect_ports(graph, muxer_out, sink_in,
				&connection)
			!= BT_GRAPH_CONNECT_PORTS_STATUS_OK)
	{
		err("bt_graph_connect_ports failed\n");
		exit(EXIT_FAILURE);
	}

	return graph;
}

int
mkpath(char *file_path, mode_t mode, int last)
{
	char *p;

	assert(file_path && *file_path);
	for(p = strchr(file_path + 1, '/'); p; p = strchr(p + 1, '/'))
	{
		*p = '\0';
		if (mkdir(file_path, mode) == -1) {
			if (errno != EEXIST) {
				*p = '/';
				return -1;
			}
		}
		*p = '/';
	}

	if(last && mkdir(file_path, mode) == -1)
		if (errno != EEXIST)
			return -1;

	return 0;
}

void
usage(int argc, char *argv[])
{
	fprintf(stderr, "Usage: %s [-jq] [-o <dir>] [-f <types>] <trace>\n", argv[0]);
	fprintf(stderr, "\n");
	fprintf(stderr, "  Convert CTF traces to PRV\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "The specified <trace> must contain a directory\n"
			"called \"ctf\" inside.\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "The output is placed in the output directory\n"
			"optionally specified with the \"-o\" option.\n"
			"By default the directory is at <trace>/prv\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Use -j to join multiple events in a single line.\n");
	fprintf(stderr, "This results in smaller traces but harder to\n");
	fprintf(stderr, "manipulate with common text processing tool.\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Use -f to specify a list of comma-delimited\n");
	fprintf(stderr, "Paraver event types. Only events matching the\n");
	fprintf(stderr, "specified type numbers will be written in the\n");
	fprintf(stderr, "PRV file if this option is enabled.\n");
	fprintf(stderr, "Don't use spaces between commas.\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "Use -q to be quiet.\n");

	exit(EXIT_FAILURE);
}

void parse_filters(long filter_events[MAX_FILTER], int *num_filters, char *arg)
{
	char *event;
	int i = 0;

	event = strtok(arg, ",");

	while (event != NULL) {
		if (i >= MAX_FILTER) {
			fprintf(stderr, "too many filters\n");
			exit(EXIT_FAILURE);
		}

		errno = 0;
		filter_events[i] = strtol(event, NULL, 10);

		if (errno) {
			perror("could not parse filter event type\n");
			exit(EXIT_FAILURE);
		}

		++i;
		event = strtok(NULL, ",");
	}

	*num_filters = i;
}

int main(int argc, char *argv[])
{
	bt_graph *graph;
	char input_trace[PATH_MAX];
	char output_dir[PATH_MAX];
	long filter_events[MAX_FILTER];
	int num_filters;
	const char *input_dir;
	int opt, split_events, quiet;

	output_dir[0] = '\0';

	split_events = 1;
	quiet = 0;

	while ((opt = getopt(argc, argv, "jqo:f:h")) != -1)
	{
		switch (opt) {
		case 'j':
			split_events = 0;
			break;
		case 'q':
			quiet = 1;
			break;
		case 'o':
			if(strlen(optarg) >= PATH_MAX)
			{
				fprintf(stderr, "output dir too large\n");
				exit(EXIT_FAILURE);
			}
			strcpy(output_dir, optarg);
			break;
		case 'f':
			parse_filters(filter_events, &num_filters, optarg);
			break;
		case 'h':
		default: /* '?' */
			usage(argc, argv);
		}
	}

	if (optind >= argc) {
		fprintf(stderr, "Missing trace\n");
		usage(argc, argv);
	}

	input_dir = argv[optind];

	if(snprintf(input_trace, PATH_MAX, "%s/ctf/user",
				input_dir) >= PATH_MAX)
	{
		fprintf(stderr, "input path too large\n");
		exit(EXIT_FAILURE);
	}

	if(output_dir[0] == '\0')
	{
		if(snprintf(output_dir, PATH_MAX, "%s/prv",
					input_dir) >= PATH_MAX)
		{
			fprintf(stderr, "input path too large\n");
			exit(EXIT_FAILURE);
		}
	}

	if(mkpath(output_dir, 0755, 1))
	{
		perror("cannot create directories");
		exit(EXIT_FAILURE);
	}

	graph = create_graph(input_trace, output_dir, split_events,
			filter_events, num_filters, quiet);

	if(bt_graph_run(graph) != BT_GRAPH_RUN_STATUS_OK)
	{
		err("bt_graph_run failed\n");
		exit(EXIT_FAILURE);
	}

	bt_graph_put_ref(graph);

	return 0;
}
