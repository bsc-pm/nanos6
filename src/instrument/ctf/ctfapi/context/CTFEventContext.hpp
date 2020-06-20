/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFEVENTCONTEXT_HPP
#define CTFEVENTCONTEXT_HPP

#include <string>

#include "CTFContext.hpp"
#include "../CTFTypes.hpp"

namespace CTFAPI {

	enum ctf_contexes {
		CTFContextTaskHWC = 1,
		CTFContextCPUHWC
	};

	class CTFEventContext : public CTFContext {
	protected:
		ctf_stream_id_t _streamMask;
		std::string eventMetadata;

		CTFEventContext(ctf_stream_id_t streamMask)
			: CTFContext(), _streamMask(streamMask) {}
	public:
		virtual ~CTFEventContext() {}

		virtual void writeContext(void **, ctf_stream_id_t) {}

		inline ctf_stream_id_t getStreamMask() const
		{
			return _streamMask;
		}

		inline size_t getSize(ctf_stream_id_t streamId) const
		{
			return (_streamMask & streamId)? size : 0;
		}

		const char *getEventMetadata() const
		{
			return eventMetadata.c_str();
		}
	};
}

#endif // CTFEVENTCONTEXT_HPP
