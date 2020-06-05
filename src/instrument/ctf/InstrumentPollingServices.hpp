/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_POLLING_SERVICES_HPP
#define INSTRUMENT_CTF_POLLING_SERVICES_HPP

#include <cstdint>

#include "tracepoints.hpp"
#include "../api/InstrumentPollingServices.hpp"


namespace Instrument {

	inline void pollingServiceEnter(uint8_t id)
	{
		tp_polling_service_enter(id);
	}

	inline void pollingServiceExit()
	{
		tp_polling_service_exit();
	}

	inline void pollingServiceRegister(const char *name, uint8_t id)
	{
		tp_polling_service_register(name, id);
	}

}

#endif //INSTRUMENT_CTF_POLLING_SERVICES_HPP
