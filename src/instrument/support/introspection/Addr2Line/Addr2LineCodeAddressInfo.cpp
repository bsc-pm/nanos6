/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "Addr2LineCodeAddressInfo.hpp"
#include "../DL/DLCodeAddressInfo.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

#include <unistd.h>
#include <sys/types.h>


std::map<void *, Addr2LineCodeAddressInfo::MemoryMapSegment> Addr2LineCodeAddressInfo::_executableMemoryMap;


void Addr2LineCodeAddressInfo::init()
{
	// Already initialized
	if (!_executableMemoryMap.empty()) {
		return;
	}
	
	DLCodeAddressInfo::init();
	
	pid_t pid = getpid();
	
	std::string mapsFilename;
	{
		std::ostringstream oss;
		oss << "/proc/" << pid << "/maps";
		mapsFilename = oss.str();
	}
	std::ifstream mapsFile(mapsFilename.c_str());
	
	if (!mapsFile.is_open()) {
		std::cerr << "Warning: cannot get the memory map of the process from '" << mapsFilename << "'" << std::endl;
		return;
	}
	
	std::istringstream splitter;
	std::string field;
	std::istringstream hexDecoder;
	hexDecoder.setf(std::ios::hex, std::ios::basefield);
	while (!mapsFile.eof() && !mapsFile.bad()) {
		std::string line;
		std::getline(mapsFile, line);
		
		if (mapsFile.eof()) {
			break;
		} else if (mapsFile.bad()) {
			std::cerr << "Warning: error getting the memory map of the process from '" << mapsFilename << "'" << std::endl;
			break;
		}
		
		splitter.clear();
		splitter.str(line);
		
		// Memory start address
		size_t baseAddress;
		std::getline(splitter, field, '-');
		hexDecoder.clear();
		hexDecoder.str(field);
		hexDecoder >> baseAddress;
		
		MemoryMapSegment &memoryMapSegment = _executableMemoryMap[(void *) baseAddress];
		
		// Memory end address + 1
		std::getline(splitter, field, ' ');
		hexDecoder.clear();
		hexDecoder.str(field);
		hexDecoder >> memoryMapSegment._length;
		memoryMapSegment._length -= baseAddress;
		
		// Permissions
		std::getline(splitter, field, ' ');
		
		// Offset
		std::getline(splitter, field, ' ');
		hexDecoder.clear();
		hexDecoder.str(field);
		hexDecoder >> memoryMapSegment._offset;
		
		// Device
		std::getline(splitter, field, ' ');
		
		// Inode
		long inode;
		splitter >> inode;
		
		// Path (if any)
		std::string path;
		std::getline(splitter, path);
		{
			size_t beginningOfPath = path.find_first_not_of(' ');
			if (beginningOfPath != std::string::npos) {
				path = path.substr(beginningOfPath);
				if (!path.empty() && (path[0] != '[')) {
					memoryMapSegment._filename = std::move(path);
				}
			}
		}
	}
	
	mapsFile.close();
}


void Addr2LineCodeAddressInfo::shutdown()
{
	DLCodeAddressInfo::shutdown();
}


Addr2LineCodeAddressInfo::Entry const &Addr2LineCodeAddressInfo::resolveAddress(void *address, bool callSiteFromReturnAddress)
{
	if (callSiteFromReturnAddress) {
		auto it = _returnAddress2Entry.find(address);
		if (it != _returnAddress2Entry.end()) {
			return it->second;
		}
	} else {
		auto it = _address2Entry.find(address);
		if (it != _address2Entry.end()) {
			return it->second;
		}
	}
	
	auto it = _executableMemoryMap.upper_bound(address);
	if (it == _executableMemoryMap.begin()) {
		// The address cannot be resolved
		
		// Fall back to resolving through DL
		return DLCodeAddressInfo::resolveAddress(address, callSiteFromReturnAddress);
	}
	it--;
	
	MemoryMapSegment const &memoryMapSegment = it->second;
	
	if (memoryMapSegment._filename.empty()) {
		// Fall back to resolving through DL
		return DLCodeAddressInfo::resolveAddress(address, callSiteFromReturnAddress);
	}
	
	Entry entry;
	
	entry._realAddress = address;
	if (callSiteFromReturnAddress) {
		// We cannot recover the call site address, but apparently addr2line jumps back to the previous line just by subtracting 1 byte
		entry._realAddress = (void *) ((size_t) address - 1);
	}
	
	size_t relativeAddress = (size_t)entry._realAddress - (size_t)it->first;
	
	std::ostringstream addr2lineCommandLine;
	addr2lineCommandLine << "addr2line -i -f -e " << memoryMapSegment._filename << " " << std::hex << relativeAddress;
	
	FILE *addr2lineOutput = popen(addr2lineCommandLine.str().c_str(), "r");
	if (addr2lineOutput == NULL) {
		perror("Error executing addr2line");
		exit(1);
	}
	
	char buffer[8192];
	buffer[8191] = 0;
	size_t length = fread(buffer, 1, 8191, addr2lineOutput);
	std::string cpp_buffer(buffer, length);
	pclose(addr2lineOutput);
	
	std::istringstream output(cpp_buffer);
	std::string mangledFunction;
	std::string sourceLine;
	std::getline(output, mangledFunction);
	std::getline(output, sourceLine);
	
	while (!output.eof()) {
		if ((mangledFunction != "??") && (sourceLine != "??:0") && (sourceLine != "??:?")) {
			// Add the current function and source location
			std::string function = Addr2LineCodeAddressInfo::demangleSymbol(mangledFunction);
			
			if (callSiteFromReturnAddress) {
				mangledFunction = mangledFunction + std::string(" [return address]");
				function = function + std::string(" [return address]");
			}
			
			InlineFrame currentFrame = functionAndSourceToFrame(mangledFunction, function, sourceLine);
			entry._inlinedFrames.push_back(currentFrame);
		}
		
		std::getline(output, mangledFunction);
		std::getline(output, sourceLine);
	}
	
	if (callSiteFromReturnAddress) {
		_returnAddress2Entry[address] = std::move(entry);
		return _returnAddress2Entry[address];
	} else {
		_address2Entry[address] = std::move(entry);
		return _address2Entry[address];
	}
}

