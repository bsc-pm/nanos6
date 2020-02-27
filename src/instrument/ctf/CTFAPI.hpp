/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFAPI_HPP
#define CTFAPI_HPP

namespace CTFAPI {

	//static inline void emit_SimpleEvent (extrae_type_t type, extrae_value_t value)
	//{
	//	ExtraeSymbolResolver<void, &Instrument::_Extrae_event_symbolName, extrae_type_t, extrae_value_t>::call(type, value);
	//}
	
	void tracepoint(void);
}

#endif // CTFAPI_HPP
