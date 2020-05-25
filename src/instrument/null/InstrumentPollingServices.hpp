/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_POLLING_SERVICES_HPP
#define INSTRUMENT_NULL_POLLING_SERVICES_HPP

#include <cstdint>

#include "../api/InstrumentPollingServices.hpp"

namespace Instrument {

	inline void pollingServiceEnter(
		__attribute__((unused)) uint8_t id
	) {
	}

	inline void pollingServiceExit() {}

	inline void pollingServiceRegister(
		__attribute__((unused)) const char *name,
		__attribute__((unused)) uint8_t id
	) {
	}

}

#endif //INSTRUMENT_NULL_POLLING_SERVICES_HPP

