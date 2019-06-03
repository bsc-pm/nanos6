/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/monitoring.h>

#include <Monitoring.hpp>


extern "C" double nanos6_get_predicted_elapsed_time(void)
{
	return Monitoring::getPredictedElapsedTime();
}
