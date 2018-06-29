/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef BACKTRACE_RECORDING_HPP
#define BACKTRACE_RECORDING_HPP


#include <BacktraceWalker.hpp>
#include <CodeAddressInfo.hpp>
#include <lowlevel/SpinLock.hpp>

#include <ostream>
#include <vector>


class RecordedBacktrace {
private:
	typedef std::vector<Instrument::BacktraceWalker::address_t> container_t;
	
	container_t _backtrace;
	
	// NOTE: we use a functor instead of a lambda to force inline expansion
	struct Functor {
		container_t &_backtrace;
		
		__attribute__((always_inline)) Functor(container_t &bt)
			: _backtrace(bt)
		{
		}
		
		__attribute__((always_inline)) void operator()(Instrument::BacktraceWalker::address_t address, int)
		{
			_backtrace.push_back(address);
		}
	};
	
public:
	enum max_frames_enum_t {
		max_frames = 20
	};
	
	RecordedBacktrace()
		: _backtrace()
	{
	}
	
	RecordedBacktrace(RecordedBacktrace &&other)
		: _backtrace(std::move(other._backtrace))
	{
	}
	
	RecordedBacktrace &operator=(RecordedBacktrace &&other)
	{
		_backtrace = std::move(other._backtrace);
		return *this;
	}
	
	__attribute__((always_inline)) void capture(int skipFrames = 0)
	{
		_backtrace.clear();
		Instrument::BacktraceWalker::walk(max_frames, skipFrames, Functor(_backtrace));
	}
	
	friend inline std::ostream &operator<<(std::ostream &o, RecordedBacktrace const &bt);
};


inline std::ostream &operator<<(std::ostream &o, RecordedBacktrace const &bt)
{
	static SpinLock lock;
	lock.lock();
	
	CodeAddressInfo::init();
	for (auto address : bt._backtrace) {
		CodeAddressInfo::Entry const &entry = CodeAddressInfo::resolveAddress(address, /* A return address */ true);
		for (auto frame : entry._inlinedFrames) {
			o << "\t" << CodeAddressInfo::getSourceLocation(frame._sourceLocationId)
				<< "\t" << CodeAddressInfo::getFunctionName(frame._functionId)
				<< std::endl;
		}
	}
	
	lock.unlock();
	
	return o;
}


#endif // BACKTRACE_RECORDING_HPP
