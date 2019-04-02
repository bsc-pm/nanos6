/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2017 Barcelona Supercomputing Center (BSC)
*/


#ifndef EXTERNAL_THREAD_HPP
#define EXTERNAL_THREAD_HPP


#include <InstrumentExternalThreadLocalData.hpp>
#include <InstrumentThreadManagement.hpp>
#include "CPUDependencyData.hpp"

#include <support/StringComposer.hpp>

#include <string>


class ExternalThread {
private:
	//! Thread Local Storage variable to point back to the ExternalThread that is running the code
	static __thread ExternalThread *_currentExternalThread;
	
	std::string _name;
	Instrument::external_thread_id_t _instrumentationId;
	Instrument::ExternalThreadLocalData _instrumentationData;
	
	//! Dependency data to interact with the dependecy system
	CPUDependencyData _dependencyData;
	
	
public:
	template<typename... TS>
	ExternalThread(TS... nameComponents)
		: _name(StringComposer::compose(nameComponents...)), _instrumentationData(_name)
	{
	}
	
	static inline void setCurrentExternalThread(ExternalThread *externalThread)
	{
		_currentExternalThread = externalThread;
	}
	static inline ExternalThread *getCurrentExternalThread()
	{
		return _currentExternalThread;
	}
	
	Instrument::ExternalThreadLocalData const &getInstrumentationData() const
	{
		return _instrumentationData;
	}
	Instrument::ExternalThreadLocalData &getInstrumentationData()
	{
		return _instrumentationData;
	}
	
	Instrument::external_thread_id_t getInstrumentationId() const
	{
		return _instrumentationId;
	}
	
	inline void preinitializeExternalThread()
	{
		ExternalThread::setCurrentExternalThread(this);
		Instrument::precreatedExternalThread(/* OUT */ _instrumentationId);
	}
	
	
	inline void initializeExternalThread(bool preinitialize = true)
	{
		ExternalThread::setCurrentExternalThread(this);
		if (preinitialize) {
			Instrument::precreatedExternalThread(/* OUT */ _instrumentationId);
		}
		Instrument::createdExternalThread(_instrumentationId, _name);
	}
	
};


#endif // EXTERNAL_THREAD_HPP

