#include "InstrumentStats.hpp"


namespace Instrument {
	namespace Stats {
		thread_local ThreadInfo *_threadStats;
		
		SpinLock _threadInfoListSpinLock;
		std::list<ThreadInfo *> _threadInfoList;
		
		Timer _totalTime(true);
	}
}
