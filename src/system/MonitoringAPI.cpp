#include <nanos6/monitoring.h>

#include <Monitoring.hpp>


extern "C" double nanos6_get_predicted_elapsed_time()
{
	return Monitoring::getPredictedElapsedTime();
}
