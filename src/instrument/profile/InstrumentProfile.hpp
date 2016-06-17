#ifndef INSTRUMENT_PROFILE_HPP
#define INSTRUMENT_PROFILE_HPP

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif

#include <signal.h>
#include <time.h>


#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/SpinLock.hpp"

#include "InstrumentThreadId.hpp"

#include <list>
#include <map>
#include <string>
#include <vector>


namespace Instrument {
	class Profile {
		// Environment variables
		EnvironmentVariable<long> _profilingNSResolution;
		EnvironmentVariable<long> _profilingBacktraceDepth;
		EnvironmentVariable<long> _profilingBufferSize;
		
		// Basic types
		typedef void *address_t;
		typedef uint32_t id_t;
		typedef uint32_t freq_t;
		
		// All the buffers with the collected samples
		SpinLock _bufferListSpinLock;
		std::list<address_t *> _bufferList;
		
		// Thread-local information
		struct PerThread {
			timer_t _profilingTimer;
			address_t *_currentBuffer;
			long _nextBufferPosition;
		};
		static __thread PerThread _perThread;
		static bool _enabled;
		
		// Map between the address space and the executable objects
		struct MemoryMapSegment {
			std::string _filename;
			size_t _offset;
			size_t _length;
			
			MemoryMapSegment()
				: _filename(), _offset(0), _length(0)
			{
			}
		};
		std::map<address_t, MemoryMapSegment> _executableMemoryMap;
		
		// Backtrace (possibly inline) step information
		struct AddrInfoStep {
			id_t _functionId;
			id_t _sourceLineId;
			
			AddrInfoStep()
				: _functionId(0), _sourceLineId(0)
			{
			}
		};
		
		// Backtrace of a single address (may contain inlined nested calls, hence the list)
		typedef std::list<AddrInfoStep> AddrInfo;
		
		AddrInfo _unknownAddrInfo;
		
		
		struct NameAndFrequency {
			std::string _name;
			freq_t _frequency;
			
			NameAndFrequency()
				: _name(), _frequency(0)
			{
			}
			
			NameAndFrequency(std::string const &name)
				: _name(name), _frequency(0)
			{
			}
			
			NameAndFrequency(std::string &&name)
				: _name(std::move(name)), _frequency(0)
			{
			}
		};
		
		// Map of addresses to their information
		std::map<void *, AddrInfo> _addr2Cache;
		
		// Function identifiers that map to their corresponding names and their histogram
		id_t _nextSourceFunctionId;
		std::map<id_t, NameAndFrequency> _id2sourceFunction;
		std::map<std::string, id_t> _sourceFunction2id;
		
		// Line number identifiers that map to their corresponding textual description and their histogram
		id_t _nextSourceLineId;
		std::map<id_t, NameAndFrequency> _id2sourceLine;
		std::map<std::string, id_t> _sourceLine2id;
		
		
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
		
		static void sigprofHandler(int signal, siginfo_t *sigInfo, void *signalContext);
		
		inline AddrInfo const &resolveAddress(address_t address);
		void buildExecutableMemoryMap(pid_t pid);
		
		void doInit();
		void doShutdown();
		thread_id_t doCreatedThread();
		
	public:
		Profile()
			: _profilingNSResolution("NANOS_PROFILE_NS_RESOLUTION", 1000),
			_profilingBacktraceDepth("NANOS_PROFILE_BACKTRACE_DEPTH", 4),
			_profilingBufferSize("NANOS_PROFILE_BUFFER_SIZE", /* 1 second */ 1000000000UL / _profilingNSResolution),
			_bufferListSpinLock(), _bufferList()
		{
		}
		
		static inline void init()
		{
			_singleton.doInit();
		}
		static inline void shutdown()
		{
			_singleton.doShutdown();
		}
		
		static inline thread_id_t createdThread()
		{
			return _singleton.doCreatedThread();
		}
	};
}


#endif // INSTRUMENT_PROFILE_HPP
