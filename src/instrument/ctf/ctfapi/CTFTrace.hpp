/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFTRACE_HPP
#define CTFTRACE_HPP

#include <string>
#include <cstdint>

#include "lowlevel/EnvironmentVariable.hpp"
#include "support/config/ConfigVariable.hpp"

#include "CTFUserMetadata.hpp"
#include "CTFKernelMetadata.hpp"


namespace CTFAPI {

	class CTFTrace {
	public:
		static CTFTrace& getInstance()
		{
			static CTFTrace instance;
			return instance;
		}

		static int getTraceVersion()
		{
			return _traceVersion;
		}

	private:
		static ConfigVariable<std::string> _defaultTemporalPath;
		static ConfigVariable<std::string> _ctf2prvWrapper;
		static ConfigVariable<bool> _ctf2prvEnabled;
		static ConfigVariable<bool> _ctf2prvFast;
		static EnvironmentVariable<std::string> _systemPATH;
		static const int _traceVersion;

		std::string _finalTracePath;
		std::string _finalTraceBasePath;
		std::string _tmpTracePath;
		std::string _userPath;
		std::string _kernelPath;
		std::string _binaryName;
		std::string _logPreamble;
		uint64_t _pid;
		uint32_t _rank;
		uint32_t _numberOfRanks;

		CTFUserMetadata *_userMetadata;
		CTFKernelMetadata *_kernelMetadata;

		// Times in nanoseconds
		uint64_t _absoluteStartTime;
		uint64_t _absoluteEndTime;
		int64_t  _timeCorrection;

		uint16_t _totalCPUs;

		CTFTrace();
	public:
		CTFTrace(CTFTrace const&)       = delete;
		void operator=(CTFTrace const&) = delete;

		void setTracePath(const char* tracePath);
		void createTraceDirectories(std::string &basePath, std::string &userPath, std::string &kernelPath);
		void initializeTraceTimer();
		void finalizeTraceTimer();
		std::string searchPythonCommand(const char *command);
		void convertToParaver();
		std::string makeFinalTraceDirectory();
		void moveTemporalTraceToFinalDirectory();
		void clean();

		void setUserMetadata(CTFUserMetadata *metadata)
		{
			_userMetadata = metadata;
		}

		CTFUserMetadata *getUserMetadata() const
		{
			return _userMetadata;
		}

		void setKernelMetadata(CTFKernelMetadata *metadata)
		{
			_kernelMetadata = metadata;
		}

		CTFKernelMetadata *getKernelMetadata() const
		{
			return _kernelMetadata;
		}

		void setTotalCPUs(uint16_t totalCPUs)
		{
			_totalCPUs = totalCPUs;
		}

		uint16_t getTotalCPUs() const
		{
			return _totalCPUs;
		}

		uint64_t getPid() const
		{
			return _pid;
		}

		const char *getBinaryName() const
		{
			return _binaryName.c_str();
		}

		inline void setDistributedMemory(
			int64_t  timeCorrection,
			uint32_t rank,
			uint32_t nranks
		) {
			_timeCorrection = timeCorrection;
			_rank = rank;
			_numberOfRanks = nranks;
			if (isDistributedMemoryEnabled()) {
				_logPreamble = "rank " + std::to_string(getRank()) + ": ";
			}
		}

		inline bool isDistributedMemoryEnabled()
		{
			return (_numberOfRanks != 0);
		}

		inline uint32_t getRank() const
		{
			return _rank;
		}

		inline uint32_t getNumberOfRanks() const
		{
			return _numberOfRanks;
		}

		std::string getLogPreamble()
		{
			return _logPreamble;
		}

		inline uint64_t getAbsoluteStartTimestamp() const {
			return _absoluteStartTime;
		}

		inline uint64_t getAbsoluteEndTimestamp() const {
			return _absoluteEndTime;
		}

		inline int64_t getTimeCorrection() const {
			return _timeCorrection;
		}

		inline std::string getTemporalTracePath() const {
			return _tmpTracePath;
		}

		inline std::string getUserTracePath() const {
			return _userPath;
		}

		inline std::string getKernelTracePath() const {
			return _kernelPath;
		}
	};
}

#endif //CTFTRACE_HPP
