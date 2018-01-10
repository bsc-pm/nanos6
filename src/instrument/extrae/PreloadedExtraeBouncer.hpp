/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2017 Barcelona Supercomputing Center (BSC)
*/


#ifndef PRELOADED_EXTRAE_BOUNCER_HPP
#define PRELOADED_EXTRAE_BOUNCER_HPP


#include "extrae_user_events.h"
#include "extrae_types.h"


#include "ExtraeSymbolLiterals.hpp"
#include "ExtraeSymbolResolver.hpp"


extern "C" {

inline void Extrae_change_num_threads (unsigned n)
{
	ExtraeSymbolResolver<void, &Instrument::_Extrae_change_num_threads_symbolName, unsigned>::call(n);
}


inline void Extrae_define_event_type (extrae_type_t *type, char *type_description, unsigned *nvalues, extrae_value_t *values, char **values_description)
{
	ExtraeSymbolResolver<void, &Instrument::_Extrae_define_event_type_symbolName, extrae_type_t *, char *, unsigned *, extrae_value_t *, char **>::call(type, type_description, nvalues, values, values_description);
}


inline void Extrae_emit_CombinedEvents (struct extrae_CombinedEvents *ce)
{
	ExtraeSymbolResolver<void, &Instrument::_Extrae_emit_CombinedEvents_symbolName, struct extrae_CombinedEvents *>::call(ce);
}


inline void Extrae_fini (void)
{
	ExtraeSymbolResolver<void, &Instrument::_Extrae_fini_symbolName>::call();
}


inline void Extrae_get_version (unsigned *major, unsigned *minor, unsigned *revision)
{
	ExtraeSymbolResolver<void, &Instrument::_Extrae_get_version_symbolName, unsigned *, unsigned *, unsigned *>::call(major, minor, revision);
}


inline void Extrae_init (void)
{
	ExtraeSymbolResolver<void, &Instrument::_Extrae_init_symbolName>::call();
}


inline void Extrae_init_UserCommunication (struct extrae_UserCommunication *uc)
{
	ExtraeSymbolResolver<void, &Instrument::_Extrae_init_UserCommunication_symbolName, struct extrae_UserCommunication *>::call(uc);
}


inline void Extrae_init_CombinedEvents (struct extrae_CombinedEvents *ce)
{
	ExtraeSymbolResolver<void, &Instrument::_Extrae_init_CombinedEvents_symbolName, struct extrae_CombinedEvents *>::call(ce);
}


inline void Extrae_register_codelocation_type (extrae_type_t t1, extrae_type_t t2, const char* s1, const char *s2)
{
	ExtraeSymbolResolver<void, &Instrument::_Extrae_register_codelocation_type_symbolName, extrae_type_t, extrae_type_t, const char *, const char *>::call(t1, t2, s1, s2);
}


inline void Extrae_register_function_address (void *ptr, const char *funcname, const char *modname, unsigned line)
{
	ExtraeSymbolResolver<void, &Instrument::_Extrae_register_function_address_symbolName, void *, const char *, const char *, unsigned>::call(ptr, funcname, modname, line);
}


inline void Extrae_set_numthreads_function (unsigned (*numthreads_function)(void))
{
	ExtraeSymbolResolver<void, &Instrument::_Extrae_set_numthreads_function_symbolName, unsigned (*)(void)>::call(numthreads_function);
}


inline void Extrae_set_threadid_function (unsigned (*threadid_function)(void))
{
	ExtraeSymbolResolver<void, &Instrument::_Extrae_set_threadid_function_symbolName, unsigned (*)(void)>::call(threadid_function);
}

} // extern "C"

#endif // PRELOADED_EXTRAE_BOUNCER_HPP
