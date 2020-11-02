/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_VERBOSE_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_VERBOSE_THREAD_MANAGEMENT_HPP


#include "InstrumentComputePlaceId.hpp"
#include "instrument/api/InstrumentThreadManagement.hpp"
#include "instrument/generic_ids/GenericIds.hpp"
#include "support/StringSupport.hpp"


namespace Instrument {
	void createdExternalThread_private(/* OUT */ external_thread_id_t &threadId, std::string const &name);
	
	template<typename... TS>
	void createdExternalThread(/* OUT */ external_thread_id_t &threadId, TS... nameComponents)
	{
		createdExternalThread_private(threadId, StringSupport::compose(nameComponents...));
	}
}


#endif // INSTRUMENT_VERBOSE_THREAD_MANAGEMENT_HPP
