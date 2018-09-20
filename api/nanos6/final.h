/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_FINAL_H
#define NANOS6_FINAL_H

#include "major.h"


#pragma GCC visibility push(default)


// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_final_api
enum nanos6_final_api_t { nanos6_final_api = 1 };


#ifdef __cplusplus
extern "C" {
#endif


//! \brief Check if running in a final context
signed int nanos6_in_final(void);

//! \brief Check if running in a serial context
signed int nanos6_in_serial_context(void);

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif /* NANOS6_FINAL_H */
