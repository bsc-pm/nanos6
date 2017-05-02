#include <performance/HardwareCountersThreadLocalData.hpp>
#include <performance/HardwareCountersThreadLocalDataImplementation.hpp>

#include "InstrumentStats.hpp"


namespace Instrument {
	namespace Stats {
		RWTicketSpinLock _phasesSpinLock;
		int _currentPhase(0);
		std::vector<Timer> _phaseTimes;
		
		SpinLock _threadInfoListSpinLock;
		std::list<ThreadInfo *> _threadInfoList;
		
		Timer _totalTime(true);
	}
}
