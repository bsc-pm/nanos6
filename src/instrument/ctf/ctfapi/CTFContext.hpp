/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFCONTEXT_HPP
#define CTFCONTEXT_HPP

#include <vector>
#include <string>

namespace CTFAPI {

	// forward declaration to break circular dependency with CTFAPI.hpp
	template<typename T>
	void tp_write_args(void **buf, T arg);

	enum ctf_contexes {
		CTFContextHWC = 1
	};

	class CTFContext {
	protected:
		size_t size;
		std::string eventMetadata;
		std::string dataStructuresMetadata;

	public:
		CTFContext() : size(0) {}
		virtual ~CTFContext() {}

		virtual void writeContext(__attribute__((unused)) void **buf) {}

		size_t getSize()
		{
			return size;
		}

		const char *getEventMetadata() const
		{
			return eventMetadata.c_str();
		}

		const char *getDataStructuresMetadata() const
		{
			return dataStructuresMetadata.c_str();
		}
	};

	class CTFContextHardwareCounters : public CTFContext {
	private:
		std::vector<const char *> hwc_ids;

	public:
		CTFContextHardwareCounters() : CTFContext()
		{
			// TODO build metadata dynamically in the constructor based on the
			// amount of requested hardware counters
			// TODO calculate size based on amount of requested HWC

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
			uint64_t val = 666;
			for (auto it = hwc_ids.begin(); it != hwc_ids.end(); ++it) {
				//val = HawdwareCounters::getValue(*it);
				tp_write_args(buf, val);
				val++;
			}
		}
	};

	//class CTFContextUnbounded : public CTFContext {
	//	metadata = "uint32_t thread_id;\n"
	//		   "uint16_t cpu\n";
	//
	//public:
	//	CTFContextUnbounded()
	//	{
	//		size = sizeof(uint32_t) + sizeof(uint16_t);
	//	}
	//
	//	void writeContext(void **buf, uint32_t threadId, uint16_t cpu_id)
	//	{
	//		core::tp_write_args(buf, threadId, cpu_id);
	//	}
	//};
}

#endif // CTFCONTEXT_HPP
