/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <assert.h>
#include <stddef.h>

#include "intercept-main-common.h"
#include "main-wrapper.h"
#include "loader.h"


struct startup_info {
	void *sda_base;
	int (*main) (int, char **, char **);
	void (*init) (void);
	void (*fini) (void);
};

typedef int libc_start_main_function_t(
	int argc,
	char **argv,
	char **envp,
	void *auxvec,
	void (*rtld_fini) (void),
	struct startup_info *startupInfo,
	char **stackOnEntry
);


__attribute__ ((visibility ("hidden")))  libc_start_main_function_t *_nanos6_loader_next_libc_start_main = NULL;


#pragma GCC visibility push(default)

//! \brief This function overrides the function of the same name and is in charge of loading the Nanos6 runtime
int __libc_start_main(
	int argc,
	char **argv,
	char **envp,
	void *auxvec,
	void (*rtld_fini) (void),
	struct startup_info *startupInfo,
	char **stackOnEntry
) {
	_nanos6_resolve_next_start_main("__libc_start_main");
	
	assert(_nanos6_loader_wrapped_main == 0);
	_nanos6_loader_wrapped_main = startupInfo->main;
	
	struct startup_info newStartupInfo = { startupInfo->sda_base, _nanos6_loader_main, startupInfo->init, startupInfo->fini };
	
	// Continue with the "normal" startup sequence
	return _nanos6_loader_next_libc_start_main(argc, argv, envp, auxvec, rtld_fini, &newStartupInfo, stackOnEntry);
}

#pragma GCC visibility pop
