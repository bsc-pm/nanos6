/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_CPU_LOCAL_DATA_HPP
#define INSTRUMENT_CTF_CPU_LOCAL_DATA_HPP

#include "ctfapi/stream/CTFKernelStream.hpp"
#include "ctfapi/stream/CTFStream.hpp"

namespace Instrument {
	struct CPULocalData {
		CTFAPI::CTFStream *userStream;
		CTFAPI::CTFKernelStream *kernelStream;

		CPULocalData()
			: userStream(nullptr), kernelStream(nullptr)
		{
		}
	};

	CPULocalData *getCTFCPULocalData();
	CPULocalData *getCTFVirtualCPULocalData();
	void setCTFVirtualCPULocalData(CPULocalData *virtualCPULocalData);
}

#endif //INSTRUMENT_CTF_CPU_LOCAL_DATA_HPP
