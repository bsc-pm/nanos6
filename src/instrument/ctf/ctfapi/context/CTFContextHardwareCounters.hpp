/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFCONTEXTHARDWARECOUNTERS_HPP
#define CTFCONTEXTHARDWARECOUNTERS_HPP

#include <cstdlib> // TODO remove me once we are done with rand
#include <vector>

#include "CTFContext.hpp"
#include "../CTFAPI.hpp"

namespace CTFAPI {

	class CTFContextHardwareCounters : public CTFContext {
	private:
		std::vector<const char *> hwc_ids;

	public:
		CTFContextHardwareCounters() : CTFContext()
		{
			// TODO use hardware counters API
			hwc_ids.push_back("_PAPI_TOT_INS");
			size += sizeof(uint64_t);
			hwc_ids.push_back("_PAPI_TOT_CYC");
			size += sizeof(uint64_t);

			eventMetadata.append("\t\tstruct hwc hwc;\n");

			dataStructuresMetadata.append("struct hwc {\n");
			for (auto it = hwc_ids.begin(); it != hwc_ids.end(); ++it) {
				dataStructuresMetadata.append("\tinteger { size = 64; align = 8; signed = 0; encoding = none; base = 10; } " + std::string((*it)) + ";\n");
			}
			dataStructuresMetadata.append("};\n\n");
		}

		~CTFContextHardwareCounters() {}

		void writeContext(void **buf)
		{
			uint64_t val = 1000 + rand()%1000;
			for (auto it = hwc_ids.begin(); it != hwc_ids.end(); ++it) {
				tp_write_args(buf, val);
				val++;
			}
		}
	};
}

#endif //CTFCONTEXTHARDWARECOUNTERS_HPP
