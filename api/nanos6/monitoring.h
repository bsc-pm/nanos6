/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_MONITORING_H
#define NANOS6_MONITORING_H

#include "major.h"

#pragma GCC visibility push(default)


// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_monitoring_api
enum nanos6_monitoring_api_t { nanos6_monitoring_api = 1 };

#ifdef __cplusplus
extern "C" {
#endif

//! \brief Get a prediction of the elapsed execution time
//! \return The predicted elapsed execution time (microseconds) or 0
//! if not available
double nanos6_get_predicted_elapsed_time(void);

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif /* NANOS6_MONITORING_H */
