/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFCONTEXT_HPP
#define CTFCONTEXT_HPP

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

		CTFContext() : size(0) {}
	public:
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
}

#endif // CTFCONTEXT_HPP
