/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_BOOTSTRAP_H
#define NANOS6_BOOTSTRAP_H

#include <stddef.h>

#include "major.h"


#pragma GCC visibility push(default)


// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_bootstrap_api
enum nanos6_bootstrap_api_t { nanos6_bootstrap_api = 2 };


#ifdef __cplusplus
extern "C" {
#endif


//! \brief Initialize the runtime at least to the point that it will accept tasks
void nanos6_preinit(void);

//! \brief Continue with the rest of the runtime initialization
void nanos6_init(void);

//! \brief Returns true if the runtime system can execute the main application's
//!        main function.
//!
//! \returns true if the runtime can execute the main task, or false otherwise.
int nanos6_can_run_main(void);

//! \brief Registers a callback that needs to be called during shutdown
//!
//! This function registers a callback function and a pointer to arguments that should
//! be passed to this callback. The callback will be called by the runtime system during
//! the shutdown phase
void nanos6_register_completion_callback(void (*shutdown_callback)(void *), void *callback_args);

//! \brief Force the runtime to be shut down
// 
// This function is used to shut down the runtime
void nanos6_shutdown(void);


#ifdef __cplusplus
}
#endif


#pragma GCC visibility pop

#endif /* NANOS6_BOOTSTRAP_H */
