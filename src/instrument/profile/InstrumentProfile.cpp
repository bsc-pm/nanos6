/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#if HAVE_CONFIG_H
#include "config.h"
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif

#include "InstrumentProfile.hpp"
#include "InstrumentThreadLocalData.hpp"

#include "lowlevel/FatalErrorHandler.hpp"

#include <BacktraceWalker.hpp>
#include <instrument/support/sampling/SigProf.hpp>

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>




using namespace Instrument;


Instrument::Profile Instrument::Profile::_singleton;


void Instrument::Profile::signalHandler(Sampling::ThreadLocalData &samplingThreadLocal)
{
	ThreadLocalData &threadLocal = (ThreadLocalData &) samplingThreadLocal;
	
	long depth = _singleton._profilingBacktraceDepth;
	long bufferSize = _singleton._profilingBufferSize;
	
	// Pre-allocate the next buffer as soon as possible
	if (!threadLocal._inMemoryAllocation && (threadLocal._nextBuffer == nullptr)) {
		threadLocal.allocateNextBuffer(bufferSize);
		
		_singleton._bufferListSpinLock.lock();
		_singleton._bufferList.push_back(threadLocal._nextBuffer);
		_singleton._bufferListSpinLock.unlock();
	}
	
	// Replace the current buffer if necessary
	if (threadLocal._nextBufferPosition + depth > (bufferSize + 2)) {
		threadLocal._currentBuffer = threadLocal._nextBuffer;
		threadLocal._nextBuffer = nullptr;
		threadLocal._nextBufferPosition = 0;
	}
	
	// Either out of memory too many consecutive samples within a memory allocation
	if (threadLocal._currentBuffer == nullptr) {
		return;
	}
	
	BacktraceWalker::walk(
		depth,
		/* Skip */ 3,
		[&](void *address, __attribute__((unused)) int currentFrame) {
			threadLocal._currentBuffer[threadLocal._nextBufferPosition] = address;
			threadLocal._nextBufferPosition++;
		}
	);
	
	// End of backtrace mark
	threadLocal._currentBuffer[threadLocal._nextBufferPosition] = 0;
	threadLocal._nextBufferPosition++;
	
	// We keep always an end mark in the buffer and add it to the list of buffers.
	// This way we do not need to perform any kind of cleanup for the threads
	threadLocal._currentBuffer[threadLocal._nextBufferPosition] = 0; // The end mark
}


void Instrument::Profile::doCreatedThread()
{
	ThreadLocalData &threadLocal = getThreadLocalData();
	threadLocal.init(_profilingBufferSize);
	threadLocal.allocateNextBuffer(_profilingBufferSize);
	
	threadLocal._nextBufferPosition = 0;
	
	// We keep always an end mark in the buffer and add it to the list of buffers.
	// This way we do not need to perform any kind of cleanup for the threads
	
	threadLocal._currentBuffer[threadLocal._nextBufferPosition] = 0; // End of backtrace
	threadLocal._currentBuffer[threadLocal._nextBufferPosition+1] = 0; // End of buffer
	
	_bufferListSpinLock.lock();
	_bufferList.push_back(threadLocal._currentBuffer);
	_bufferListSpinLock.unlock();
	
	Sampling::SigProf::setUpThread(threadLocal);
	
	// We call the signal handler once since the first call to backtrace allocates memory.
	// If the signal is delivered within a memory allocation, the thread can deadlock.
	Sampling::SigProf::forceHandler();
	
	// Remove the sample
	threadLocal._nextBufferPosition = 0;
	threadLocal._currentBuffer[0] = 0; // End of backtrace
	threadLocal._currentBuffer[1] = 0; // End of buffer
	
	Sampling::SigProf::enableThread(threadLocal);
}


void Instrument::Profile::doShutdown()
{
	// After this, on the next profiling signal, the corresponding timer gets disarmed
	Sampling::SigProf::disable();
	std::atomic_thread_fence(std::memory_order_seq_cst);
	
	#if !defined(HAVE_BACKTRACE) && !defined(HAVE_LIBUNWIND)
	return;
	#endif
	
	
	CodeAddressInfo::init();
	
	
	// Build frequency tables and resolve address information
	std::map<address_t, freq_t> address2Frequency;
	std::map<Backtrace, freq_t> backtrace2Frequency;
	std::map<SymbolicBacktrace, freq_t> symbolicBacktrace2Frequency;
	
	{
		Backtrace backtrace(_profilingBacktraceDepth);
		SymbolicBacktrace symbolicBacktrace(_profilingBacktraceDepth);
		backtrace.clear();
		symbolicBacktrace.clear();
		int frame = 0;
		
		_bufferListSpinLock.lock();
		for (address_t *buffer : _bufferList) {
			long position = 0;
			frame = 0;
			
			while (position < _profilingBufferSize) {
				address_t address = buffer[position];
				
				if (address == 0) {
					if (frame == 0) {
						// End of buffer
						break;
					} else {
	// 					// End of backtrace
	// 					assert(frame <= _profilingBacktraceDepth);
	// 					for (; frame < _profilingBacktraceDepth; frame++) {
	// 						backtrace[frame] = 0;
	// 						symbolicBacktrace[frame].clear();
	// 					}
						
						// Increment the frequency of the backtrace
						{
							auto it = backtrace2Frequency.find(backtrace);
							if (it == backtrace2Frequency.end()) {
								backtrace2Frequency[backtrace] = 1;
							} else {
								it->second++;
							}
						}
						{
							auto it = symbolicBacktrace2Frequency.find(symbolicBacktrace);
							if (it == symbolicBacktrace2Frequency.end()) {
								symbolicBacktrace2Frequency[symbolicBacktrace] = 1;
							} else {
								it->second++;
							}
						}
						
						frame = 0;
						position++;
						backtrace.clear();
						symbolicBacktrace.clear();
						continue;
					}
				}
				
				CodeAddressInfo::Entry const &addrInfo = CodeAddressInfo::resolveAddress(address, /* return address? */ (position > 0));
				for (CodeAddressInfo::InlineFrame const &frameContents : addrInfo._inlinedFrames) {
					if (frameContents._functionId != id_t()) {
						_id2sourceFunctionFrequency[frameContents._functionId]++;
					}
					if (frameContents._sourceLocationId != id_t()) {
						_id2sourceLineFrequency[frameContents._sourceLocationId]++;
					}
				}
				
				backtrace.push_back(address); // In the backtrace we push the return address
				symbolicBacktrace.push_back(addrInfo);
				frame++;
				
				{
					auto it = address2Frequency.find(addrInfo._realAddress); // Record the statistics per call site (as opposed to return address)
					if (it != address2Frequency.end()) {
						it->second++;
					} else {
						address2Frequency[addrInfo._realAddress] = 1;
					}
				}
				
				position++;
			}
			free(buffer);
		}
		_bufferList.clear();
		_bufferListSpinLock.unlock();
	}
	
	
	std::map<freq_t, std::list<Backtrace>, std::greater<freq_t>> backtracesByFrequency;
	for (auto it : backtrace2Frequency) {
		backtracesByFrequency[it.second].push_back(it.first);
	}
	backtrace2Frequency.clear();
	
	std::map<freq_t, std::list<SymbolicBacktrace>, std::greater<freq_t>> symbolicBacktracesByFrequency;
	for (auto it : symbolicBacktrace2Frequency) {
		symbolicBacktracesByFrequency[it.second].push_back(it.first);
	}
	symbolicBacktrace2Frequency.clear();
	
	
	{
		std::ostringstream oss;
		oss << "backtrace-profile-by-address-" << getpid() << ".txt";
		
		std::ofstream backtraceProfile(oss.str().c_str());
		for (auto it : backtracesByFrequency) {
			std::list<Backtrace> const &backtraces = it.second;
			for (Backtrace const &backtrace : backtraces) {
				bool first = true;
				for (address_t address : backtrace) {
					if (address == 0) {
						break;
					}
					
					CodeAddressInfo::Entry const &addrInfo = CodeAddressInfo::resolveAddress(address, /* return address? */ !first);
					for (CodeAddressInfo::InlineFrame const &frame : addrInfo._inlinedFrames) {
						if (first) {
							// Frequency on the innermost function
							backtraceProfile << it.first;
							first = false;
						}
						
						CodeAddressInfo::FrameNames frameNames = CodeAddressInfo::getFrameNames(frame);
						backtraceProfile << "\t" << frameNames._function;
						backtraceProfile << "\t" << frameNames._sourceLocation;
						backtraceProfile << std::endl;
					}
				}
				
				if (!first) {
					backtraceProfile << std::endl;
				}
			}
		}
		backtraceProfile.close();
	}
	backtracesByFrequency.clear();
	
	
	{
		std::ostringstream oss;
		oss << "backtrace-profile-by-line-" << getpid() << ".txt";
		
		std::ofstream symbolicBacktraceProfile(oss.str().c_str());
		for (auto it : symbolicBacktracesByFrequency) {
			std::list<SymbolicBacktrace> const &symbolicBacktraces = it.second;
			for (SymbolicBacktrace const &symbolicBacktrace : symbolicBacktraces) {
				bool first = true;
				for (CodeAddressInfo::Entry const &addrInfo : symbolicBacktrace) {
					for (CodeAddressInfo::InlineFrame const &frame : addrInfo._inlinedFrames) {
						if (first) {
							// Frequency on the innermost function
							symbolicBacktraceProfile << it.first;
							first = false;
						}
						
						CodeAddressInfo::FrameNames frameNames = CodeAddressInfo::getFrameNames(frame);
						symbolicBacktraceProfile << "\t" << frameNames._function;
						symbolicBacktraceProfile << "\t" << frameNames._sourceLocation;
						symbolicBacktraceProfile << std::endl;
					}
				}
				
				if (!first) {
					symbolicBacktraceProfile << std::endl;
				}
			}
		}
		symbolicBacktraceProfile.close();
	}
	symbolicBacktracesByFrequency.clear();
	
	
	std::map<freq_t, std::list<address_t>, std::greater<freq_t>> addressesByFrequency;
	for (auto it : address2Frequency) {
		addressesByFrequency[it.second].push_back(it.first);
	}
	address2Frequency.clear();
	
	{
		std::ostringstream oss;
		oss << "inline-profile-" << getpid() << ".txt";
		
		std::ofstream inlineProfile(oss.str().c_str());
		for (auto it : addressesByFrequency) {
			std::list<address_t> const &addresses = it.second;
			for (address_t address : addresses) {
				CodeAddressInfo::Entry const &addrInfo = CodeAddressInfo::resolveAddress(address);
				if (!addrInfo.empty()) {
					// Frequency on the innermost function
					inlineProfile << it.first;
				}
				for (CodeAddressInfo::InlineFrame const &frame : addrInfo._inlinedFrames) {
					CodeAddressInfo::FrameNames frameNames = CodeAddressInfo::getFrameNames(frame);
					inlineProfile << "\t" << frameNames._function;
					inlineProfile << "\t" << frameNames._sourceLocation;
					inlineProfile << std::endl;
				}
			}
		}
		inlineProfile.close();
	}
	addressesByFrequency.clear();
	
	
	std::map<freq_t, std::list<CodeAddressInfo::function_id_t>, std::greater<freq_t>> functionsByFrequency;
	for (auto it : _id2sourceFunctionFrequency) {
		functionsByFrequency[it.second].push_back(it.first);
	}
	
	{
		std::ostringstream oss;
		oss << "function-profile-" << getpid() << ".txt";
		
		std::ofstream functionProfile(oss.str().c_str());
		for (auto it : functionsByFrequency) {
			std::list<CodeAddressInfo::function_id_t> const &functions = it.second;
			for (CodeAddressInfo::function_id_t functionId : functions) {
				functionProfile << it.first << "\t" << CodeAddressInfo::getFunctionName(functionId) << "\n";
			}
		}
		functionProfile.close();
	}
	
	std::map<freq_t, std::list<CodeAddressInfo::source_location_id_t>, std::greater<freq_t>> linesByFrequency;
	for (auto it : _id2sourceLineFrequency) {
		linesByFrequency[it.second].push_back(it.first);
	}
	
	{
		std::ostringstream oss;
		oss << "line-profile-" << getpid() << ".txt";
		
		std::ofstream lineProfile(oss.str().c_str());
		for (auto it : linesByFrequency) {
			std::list<CodeAddressInfo::source_location_id_t> const &lines = it.second;
			for (CodeAddressInfo::source_location_id_t sourceLocationId : lines) {
				lineProfile << it.first << "\t" << CodeAddressInfo::getSourceLocation(sourceLocationId) << "\n";
			}
		}
		lineProfile.close();
	}
	
	CodeAddressInfo::shutdown();
}

