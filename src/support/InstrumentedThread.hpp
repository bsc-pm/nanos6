#ifndef INSTRUMENTED_THREAD_HPP
#define INSTRUMENTED_THREAD_HPP


#include <InstrumentThreadId.hpp>


class InstrumentedThread {
protected:
	Instrument::thread_id_t _instrumentationId;
	
	
public:
	Instrument::thread_id_t getInstrumentationId() const
	{
		return _instrumentationId;
	}
	
	void setInstrumentationId(Instrument::thread_id_t const &instrumentationId)
	{
		_instrumentationId = instrumentationId;
	}
};


#endif // INSTRUMENTED_THREAD_HPP
