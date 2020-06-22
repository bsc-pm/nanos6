/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFCONTEXTCPUHARDWARECOUNTERS_HPP
#define CTFCONTEXTCPUHARDWARECOUNTERS_HPP

#include "CTFEventContext.hpp"

namespace CTFAPI {

	class CTFContextCPUHardwareCounters : public CTFEventContext {
	public:
		CTFContextCPUHardwareCounters(ctf_stream_id_t streamMask);
		~CTFContextCPUHardwareCounters() {}

		void writeContext(void **buf, ctf_stream_id_t streamId) override;
	};
}

#endif //CTFCONTEXTCPUHARDWARECOUNTERS_HPP
