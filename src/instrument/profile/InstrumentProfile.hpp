/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_PROFILE_HPP
#define INSTRUMENT_PROFILE_HPP

#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/SpinLock.hpp"

#include "Address.hpp"
#include "InstrumentThreadId.hpp"
#include "InstrumentThreadLocalData.hpp"

#include <BacktraceWalker.hpp>
#include <CodeAddressInfo.hpp>
#include <instrument/support/sampling/SigProf.hpp>

#include <list>
#include <map>
#include <string>
#include <vector>


namespace Instrument {
	class Profile {
	private:
		// Environment variables
		EnvironmentVariable<long> _profilingNSResolution;
		EnvironmentVariable<long> _profilingBacktraceDepth;
		EnvironmentVariable<long> _profilingBufferSize;
		
		
		typedef uint32_t freq_t;
		
		
		// All the buffers with the collected samples
		SpinLock _bufferListSpinLock;
		std::list<address_t *> _bufferList;
		
		
		class SymbolicBacktrace : public std::vector<CodeAddressInfo::Entry> {
		public:
			SymbolicBacktrace(size_t frames)
			: std::vector<CodeAddressInfo::Entry>(frames, CodeAddressInfo::Entry())
			{
			}
			
			bool operator==(SymbolicBacktrace const &other) const
			{
				if (size() != other.size()) {
					return false;
				}
				
				for (size_t position = 0; position < size(); position++) {
					if ((*this)[position] != other[position]) {
						return false;
					}
				}
				
				return true;
			}
			
			bool operator!=(SymbolicBacktrace const &other) const
			{
				return !((*this) == other);
			}
			
			bool operator<(SymbolicBacktrace const &other) const
			{
				size_t position = 0;
				while (true) {
					if (size() == position) {
						if (other.size() == position) {
							// Equal
							return false;
						} else {
							// this < other
							return true;
						}
					} else if (other.size() == position) {
						// this > other
						return false;
					}
					if ((*this)[position] < other[position]) {
						return true;
					} else if ((*this)[position] > other[position]) {
						return false;
					} else {
						position++;
					}
				}
			}
		};
		
		
		// Function identifiers that map to their frequency
		std::map<CodeAddressInfo::function_id_t, freq_t> _id2sourceFunctionFrequency;
		
		// Line number identifiers that map to their frequency
		std::map<CodeAddressInfo::source_location_id_t, freq_t> _id2sourceLineFrequency;
		
		
		class Backtrace : public std::vector<address_t> {
		public:
			Backtrace(size_t frames)
				: std::vector<address_t>(frames, 0)
			{
			}
			
			bool operator<(Backtrace const &other) const
			{
				size_t position = 0;
				while (true) {
					if (size() == position) {
						if (other.size() == position) {
							// Equal
							return false;
						} else {
							// this < other
							return true;
						}
					} else if (other.size() == position) {
						// this > other
						return false;
					}
					if ((*this)[position] < other[position]) {
						return true;
					} else if ((*this)[position] > other[position]) {
						return false;
					} else {
						position++;
					}
				}
			}
		};
		
		
		// Singleton object
		static Profile _singleton;
		
		static void signalHandler(Sampling::ThreadLocalData &threadLocal);
		
		
		void doShutdown();
		void doCreatedThread();
		void threadEnable();
		void threadDisable();
		void lightweightThreadEnable();
		void lightweightThreadDisable();
		
		
	public:
		Profile()
			: _profilingNSResolution("NANOS6_PROFILE_NS_RESOLUTION", 1000),
			_profilingBacktraceDepth("NANOS6_PROFILE_BACKTRACE_DEPTH", 4),
			_profilingBufferSize("NANOS6_PROFILE_BUFFER_SIZE", /* 1 second */ 1000000000UL / _profilingNSResolution),
			_bufferListSpinLock(), _bufferList()
		{
		}
		
		static inline void init()
		{
			Sampling::SigProf::setPeriod(_singleton._profilingNSResolution);
			Sampling::SigProf::setHandler(&signalHandler);
			Sampling::SigProf::init();
		}
		static inline void shutdown()
		{
			_singleton.doShutdown();
		}
		
		static inline void createdThread()
		{
			_singleton.doCreatedThread();
		}
		
		static inline void enableForCurrentThread()
		{
			Sampling::SigProf::enableThread();
		}
		static inline void disableForCurrentThread()
		{
			Sampling::SigProf::disableThread();
		}
		
		static inline void lightweightEnableForCurrentThread()
		{
			if (BacktraceWalker::involves_libc_malloc) {
				Sampling::SigProf::lightweightEnableThread();
			} else {
				ThreadLocalData &threadLocal = getThreadLocalData();
				threadLocal._inMemoryAllocation--;
			}
		}
		static inline void lightweightDisableForCurrentThread()
		{
			if (BacktraceWalker::involves_libc_malloc) {
				Sampling::SigProf::lightweightDisableThread();
			} else {
				ThreadLocalData &threadLocal = getThreadLocalData();
				threadLocal._inMemoryAllocation++;
			}
		}
		
		static inline long getBufferSize()
		{
			return _singleton._profilingBufferSize;
		}
	};
}


#endif // INSTRUMENT_PROFILE_HPP
