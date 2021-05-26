/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2021 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/polling.h>

#include "lowlevel/FatalErrorHandler.hpp"


extern "C" void nanos6_register_polling_service(char const *, nanos6_polling_service_t, void *)
{
	FatalErrorHandler::fail("Polling services API is no longer supported");
}

extern "C" void nanos6_unregister_polling_service(char const *, nanos6_polling_service_t, void *)
{
	FatalErrorHandler::fail("Polling services API is no longer supported");
}
