/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_VERBOSE_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_VERBOSE_THREAD_MANAGEMENT_HPP


#include "../generic_ids/GenericIds.hpp"
#include "InstrumentComputePlaceId.hpp"

#include "../api/InstrumentThreadManagement.hpp"

#include <support/StringComposer.hpp>


namespace Instrument {
	void createdExternalThread_private(/* OUT */ external_thread_id_t &threadId, std::string const &name);
	
	template<typename... TS>
	void createdExternalThread(/* OUT */ external_thread_id_t &threadId, TS... nameComponents)
	{
		createdExternalThread_private(threadId, StringComposer::compose(nameComponents...));
	}
}


#endif // INSTRUMENT_VERBOSE_THREAD_MANAGEMENT_HPP
