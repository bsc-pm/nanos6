/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFCONTEXTTASKHARDWARECOUNTERS_HPP
#define CTFCONTEXTTASKHARDWARECOUNTERS_HPP

#include "CTFEventContext.hpp"

namespace CTFAPI {

	class CTFContextTaskHardwareCounters : public CTFEventContext {
	public:
		CTFContextTaskHardwareCounters(ctf_stream_id_t streamMask);
		~CTFContextTaskHardwareCounters() {}

		void writeContext(void **buf, ctf_stream_id_t streamId) override;
	};
}

#endif //CTFCONTEXTTASKHARDWARECOUNTERS_HPP
