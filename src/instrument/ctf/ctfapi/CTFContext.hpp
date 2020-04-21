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
		std::string metadata;

	public:
		CTFContext() : size(0) {}

		void writeContext(__attribute__((unused)) void **buf) {}

		size_t getSize()
		{
			return size;
		}

		const char *getMetadata() const
		{
			return metadata.c_str();
		}
	};

	class CTFContextHardwareCounters : public CTFContext {
	private:
		std::vector<int> hwc_ids;

	public:
		CTFContextHardwareCounters() : CTFContext()
		{
			// TODO build metadata dynamically in the constructor based on the
			// amount of requested hardware counters
			// TODO calculate size based on amount of requested HWC

		//	size += sizeof(uint64_t);
		//	hwc_ids.push_back(33);

			int i = 0;
			for (auto it = hwc_ids.begin(); it < hwc_ids.end(); ++it) {
				metadata.append("		integer { size = 64; align = 8; signed = 0; encoding = none; base = 10; } _val" + std::to_string(i) + ";\n");
				i++;
			}
		}

		void writeContext(void **buf)
		{
			for (auto it = hwc_ids.begin(); it < hwc_ids.end(); ++it) {
				//val = HawdwareCounters::getValue(*it);
				uint64_t val = 666;
				tp_write_args(buf, val);
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
