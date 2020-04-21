/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFTRACE_HPP
#define CTFTRACE_HPP

#include <string>
#include <cstdint>

namespace CTFAPI {
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

		uint64_t absoluteStartTime;

		CTFTrace() {}

	public:
		CTFTrace(CTFTrace const&)       = delete;
		void operator=(CTFTrace const&) = delete;

		void setTracePath(const char* tracePath)
		{
			_tracePath = std::string(tracePath);
		}

		inline uint64_t getAbsoluteStartTimestamp() {
			return absoluteStartTime;
		}

		void createTraceDirectories(std::string &userPath, std::string &kernelPath);
		void initializeTraceTimer();
	};
}

#endif //CTFTRACE_HPP
