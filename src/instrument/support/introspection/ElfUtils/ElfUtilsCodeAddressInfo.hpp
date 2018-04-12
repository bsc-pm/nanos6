/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_SUPPORT_INSTROSPECTION_ELFUTILS_CODE_ADDRESS_INFO_HPP
#define INSTRUMENT_SUPPORT_INSTROSPECTION_ELFUTILS_CODE_ADDRESS_INFO_HPP


#include <elfutils/libdwfl.h>

#include <string>

#include "../CodeAddressInfoBase.hpp"


class ElfUtilsCodeAddressInfo : public CodeAddressInfoBase {
private:
	static Dwfl *_dwfl;
	
	static inline std::string getDebugInformationEntryName(Dwarf_Die *debugInformationEntry);
	
public:
	static void init();
	static void shutdown();
	static Entry const &resolveAddress(void *address, bool callSiteFromReturnAddress = false);
	
};


#endif // INSTRUMENT_SUPPORT_INSTROSPECTION_ELFUTILS_CODE_ADDRESS_INFO_BASE_HPP
