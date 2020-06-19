/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFCONTEXTCPUHARDWARECOUNTERS_HPP
#define CTFCONTEXTCPUHARDWARECOUNTERS_HPP

#include "CTFContext.hpp"

namespace CTFAPI {

	class CTFContextCPUHardwareCounters : public CTFContext {
	public:
		CTFContextCPUHardwareCounters();
		~CTFContextCPUHardwareCounters() {}

		void writeContext(void **buf);
	};
}

#endif //CTFCONTEXTCPUHARDWARECOUNTERS_HPP
