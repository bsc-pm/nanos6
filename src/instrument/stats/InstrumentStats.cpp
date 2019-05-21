/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#include "InstrumentStats.hpp"

#include <InstrumentHardwareCountersThreadLocalData.hpp>


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
