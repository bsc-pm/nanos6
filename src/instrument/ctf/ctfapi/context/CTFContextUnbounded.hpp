/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFCONTEXTUNBOUNDED_HPP
#define CTFCONTEXTUNBOUNDED_HPP

#include "CTFContext.hpp"
#include "../CTFTypes.hpp"

namespace CTFAPI {
	class CTFContextUnbounded : public CTFContext {
	public:
		CTFContextUnbounded()
		{
			// dataStrucutresMetadata content will be copied into:
			//    "struct unbounded unbounded;"
			// which belongs to the "unbounded" CTF Stream. There is
			// no need to set the eventMetadata property of
			// CTFContext as the entry is already hardcored into
			// CTFMetadata for simplicity. We probably will never
			// have more than two types of streams.

			dataStructuresMetadata = "struct unbounded {\n\tuint16_t thread_id;\n};\n\n";
			size = sizeof(ctf_thread_id_t);
		}

		~CTFContextUnbounded() {}

		void writeContext(void **buf, ctf_thread_id_t threadId)
		{
			tp_write_args(buf, threadId);
		}
	};
}

#endif //CTFCONTEXTUNBOUNDED_HPP
