/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTF_METADATA_HPP
#define CTF_METADATA_HPP


#include <cstdint>
#include <cstdio>
#include <string>

namespace CTFAPI {
	class CTFMetadata {
	public:
		static void collectCommonInformationAtInit();
		static void collectCommonInformationAtShutdown();
	protected:
		static const char *_meta_commonEnv;

		static std::string _cpuList;
		static uint32_t _externalThreadsCount;

		static void printCommonMetaEnv(FILE *f);
	};
}

#endif // CTF_METADTATA_HPP
