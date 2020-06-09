/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFCONTEXTTASKHARDWARECOUNTERS_HPP
#define CTFCONTEXTTASKHARDWARECOUNTERS_HPP

#include "CTFContext.hpp"

namespace CTFAPI {

	class CTFContextTaskHardwareCounters : public CTFContext {
	public:
		CTFContextTaskHardwareCounters();
		~CTFContextTaskHardwareCounters() {}

		void writeContext(void **buf);
	};
}

#endif //CTFCONTEXTTASKHARDWARECOUNTERS_HPP
