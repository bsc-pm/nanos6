/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFTRACE_HPP
#define CTFTRACE_HPP

#include <string>
#include <cstdint>

namespace CTFAPI {
	// forward declaration to break circular dependency between CTFTrace and
	// CTFMetadata
	class CTFMetadata;

	class CTFTrace {
	public:
		static CTFTrace& getInstance()
		{
			static CTFTrace instance;
			return instance;
		}

	private:
		std::string _tracePath;
		std::string _userPath;
		std::string _kernelPath;
		CTFMetadata *_metadata;

		uint64_t _absoluteStartTime;
		uint16_t _totalCPUs;

		CTFTrace() {}
	public:
		CTFTrace(CTFTrace const&)       = delete;
		void operator=(CTFTrace const&) = delete;

		void setTracePath(const char* tracePath)
		{
			_tracePath = std::string(tracePath);
		}

		void setMetadata(CTFMetadata *metadata)
		{
			_metadata = metadata;
		}

		void setTotalCPUs(uint16_t totalCPUs)
		{
			_totalCPUs = totalCPUs;
		}

		uint16_t getTotalCPUs(void) const
		{
			return _totalCPUs;
		}

		inline uint64_t getAbsoluteStartTimestamp(void) {
			return _absoluteStartTime;
		}

		void createTraceDirectories(std::string &userPath, std::string &kernelPath);
		void initializeTraceTimer(void);
		void clean(void);
	};
}

#endif //CTFTRACE_HPP
