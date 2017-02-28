#include "InstrumentStats.hpp"


namespace Instrument {
	namespace Stats {
		thread_local ThreadInfo *_threadStats;
		
		std::atomic<int> _currentPhase(0);
		std::vector<Timer> _phaseTimes;
		
		SpinLock _threadInfoListSpinLock;
		std::list<ThreadInfo *> _threadInfoList;
		
		Timer _totalTime(true);
	}
}
