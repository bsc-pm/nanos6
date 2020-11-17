/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTF_METADATA_HPP
#define CTF_METADATA_HPP


#include <string>

namespace CTFAPI {
	class CTFMetadata {
	public:
		static void collectCommonInformation();
	protected:
		static std::string _cpuList;

	};
}

#endif // CTF_METADTATA_HPP
