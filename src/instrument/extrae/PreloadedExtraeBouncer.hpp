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


namespace ExtraeAPI {
	
	static inline void change_num_threads (unsigned n)
	{
		ExtraeSymbolResolver<void, &Instrument::_Extrae_change_num_threads_symbolName, unsigned>::call(n);
	}
	
	
	static inline void define_event_type (extrae_type_t *type, char *type_description, unsigned *nvalues, extrae_value_t *values, char **values_description)
	{
		ExtraeSymbolResolver<void, &Instrument::_Extrae_define_event_type_symbolName, extrae_type_t *, char *, unsigned *, extrae_value_t *, char **>::call(type, type_description, nvalues, values, values_description);
	}
	
	
	static inline void emit_CombinedEvents (struct extrae_CombinedEvents *ce)
	{
		ExtraeSymbolResolver<void, &Instrument::_Extrae_emit_CombinedEvents_symbolName, struct extrae_CombinedEvents *>::call(ce);
	}
	
	
	static inline void fini (void)
	{
		ExtraeSymbolResolver<void, &Instrument::_Extrae_fini_symbolName>::call();
	}
	
	
	static inline void get_version (unsigned *major, unsigned *minor, unsigned *revision)
	{
		ExtraeSymbolResolver<void, &Instrument::_Extrae_get_version_symbolName, unsigned *, unsigned *, unsigned *>::call(major, minor, revision);
	}
	
	
	static inline void init (void)
	{
		ExtraeSymbolResolver<void, &Instrument::_Extrae_init_symbolName>::call();
	}
	
	
	static inline void init_UserCommunication (struct extrae_UserCommunication *uc)
	{
		ExtraeSymbolResolver<void, &Instrument::_Extrae_init_UserCommunication_symbolName, struct extrae_UserCommunication *>::call(uc);
	}
	
	
	static inline void init_CombinedEvents (struct extrae_CombinedEvents *ce)
	{
		ExtraeSymbolResolver<void, &Instrument::_Extrae_init_CombinedEvents_symbolName, struct extrae_CombinedEvents *>::call(ce);
	}
	
	
	static inline void register_codelocation_type (extrae_type_t t1, extrae_type_t t2, const char* s1, const char *s2)
	{
		ExtraeSymbolResolver<void, &Instrument::_Extrae_register_codelocation_type_symbolName, extrae_type_t, extrae_type_t, const char *, const char *>::call(t1, t2, s1, s2);
	}
	
	
	static inline void register_function_address (void *ptr, const char *funcname, const char *modname, unsigned line)
	{
		ExtraeSymbolResolver<void, &Instrument::_Extrae_register_function_address_symbolName, void *, const char *, const char *, unsigned>::call(ptr, funcname, modname, line);
	}
	
	
	static inline void set_numthreads_function (unsigned (*numthreads_function)(void))
	{
		ExtraeSymbolResolver<void, &Instrument::_Extrae_set_numthreads_function_symbolName, unsigned (*)(void)>::call(numthreads_function);
	}
	
	
	static inline void set_threadid_function (unsigned (*threadid_function)(void))
	{
		ExtraeSymbolResolver<void, &Instrument::_Extrae_set_threadid_function_symbolName, unsigned (*)(void)>::call(threadid_function);
	}
	
	
	// De-Fortranized interface
	static inline void define_event_type (extrae_type_t type, char const *type_description, unsigned nvalues, extrae_value_t *values, char const * const *values_description)
	{
		define_event_type(&type, (char *) type_description, &nvalues, values, (char **) values_description);
	}
	
} // namespace ExtraeAPI


#endif // PRELOADED_EXTRAE_BOUNCER_HPP
