/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_DATA_ACCESS_ID_HPP
#define INSTRUMENT_EXTRAE_DATA_ACCESS_ID_HPP


#include "InstrumentTaskId.hpp"

#include "../api/InstrumentDataAccessTypes.hpp"


namespace Instrument {
	//! This is the default data access identifier for the instrumentation.
	//! It should be redefined in an identically named file within each instrumentation implementation.
	struct data_access_id_t {
		DataAccessType _accessType;
		bool _weak;
		access_object_type_t _objectType;
		task_id_t _originator;
		
		data_access_id_t()
			: _originator()
		{
		}
		
		data_access_id_t(DataAccessType accessType, bool weak, access_object_type_t objectType, task_id_t originatorTaskId)
			: _accessType(accessType), _weak(weak), _objectType(objectType), _originator(originatorTaskId)
		{
		}
		
		bool operator==(__attribute__((unused)) data_access_id_t const &other) const
		{
			return true;
		}
		bool operator!=(__attribute__((unused)) data_access_id_t const &other) const
		{
			return false;
		}
	};
}

#endif // INSTRUMENT_EXTRAE_DATA_ACCESS_ID_HPP
