/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFCONTEXTUNBOUNDED_HPP
#define CTFCONTEXTUNBOUNDED_HPP

#include "lowlevel/threads/ExternalThread.hpp"

#include "CTFContext.hpp"
#include "instrument/ctf/ctfapi/CTFAPI.hpp"
#include "instrument/ctf/ctfapi/CTFTypes.hpp"


namespace CTFAPI {
	class CTFStreamContextUnbounded : public CTFContext {
	public:
		CTFStreamContextUnbounded() : CTFContext()
		{
			// dataStrucutresMetadata content will be copied into:
			//    "struct unbounded unbounded;"
			// which belongs to the "unbounded" CTF Stream. There is
			// no need to set the eventMetadata property of
			// CTFContext as the entry is already hardcored into
			// CTFMetadata for simplicity. We probably will never
			// have more than two types of streams.

			dataStructuresMetadata = "struct unbounded {\n\tuint16_t tid;\n};\n\n";
			size = sizeof(ctf_thread_id_t);
		}

		~CTFStreamContextUnbounded() {}

		inline size_t getSize() const
		{
			return size;
		}

		inline void writeContext(void **buf)
		{
			ExternalThread *currentExternalThread = ExternalThread::getCurrentExternalThread();
			assert(currentExternalThread != nullptr);
			ctf_thread_id_t tid = currentExternalThread->getInstrumentationId().tid;
			CTFAPI::tp_write_args<>(buf, tid);
		}
	};
}

#endif //CTFCONTEXTUNBOUNDED_HPP
