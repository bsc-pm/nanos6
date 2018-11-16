/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "ReductionInfo.hpp"

#include <cassert>
#include <sys/mman.h>

#include <executors/threads/CPUManager.hpp>
#include <hardware/HardwareInfo.hpp>

#include <InstrumentReductions.hpp>

#include <executors/threads/WorkerThread.hpp>
#include <InstrumentThreadInstrumentationContextImplementation.hpp>

ReductionInfo::ReductionInfo(DataAccessRegion region, reduction_type_and_operator_index_t typeAndOperatorIndex,
		std::function<void(void*, void*, size_t)> initializationFunction, std::function<void(void*, void*, size_t)> combinationFunction) :
	_paddedRegionSize(((region.getSize() + HardwareInfo::getCacheLineSize() - 1)/HardwareInfo::getCacheLineSize())*HardwareInfo::getCacheLineSize()),
	_region(region), _typeAndOperatorIndex(typeAndOperatorIndex),
	_initializationFunction(std::bind(initializationFunction, std::placeholders::_1, _region.getStartAddress(), std::placeholders::_2)),
	_combinationFunction(combinationFunction)
{
	const long nCpus = CPUManager::getTotalCPUs();
	assert(nCpus > 0);
	
	void *storage = mmap(nullptr, _paddedRegionSize*nCpus,
			PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, /* fd */ -1, /* offset */ 0);
	
	FatalErrorHandler::check(storage != MAP_FAILED, "Cannot allocate ",
			_paddedRegionSize*nCpus, " bytes");
	
	_storage = DataAccessRegion(storage, _paddedRegionSize*nCpus);
	
	_isCpuStorageInitialized.resize(nCpus, false);
	
	_sizeCounter = _region.getSize();
}

ReductionInfo::~ReductionInfo()
{
	int error = munmap(_storage.getStartAddress(), _storage.getSize());
	
	FatalErrorHandler::check(error == 0, "Cannot deallocate ",
			_storage.getSize(), " bytes");
}

reduction_type_and_operator_index_t ReductionInfo::getTypeAndOperatorIndex() const {
	return _typeAndOperatorIndex;
}

const DataAccessRegion& ReductionInfo::getOriginalRegion() const {
	return _region;
}

DataAccessRegion ReductionInfo::getCPUPrivateStorage(size_t virtualCpuId) {
	__attribute__((unused)) const long nCpus = CPUManager::getTotalCPUs();
	assert(nCpus > 0);
	assert(virtualCpuId < (size_t)nCpus);
	
	void *cpuStorage = ((char*)_storage.getStartAddress()) + _paddedRegionSize*virtualCpuId;
	
	// Lock required to access _isCpuStorageInitialized simultaneously
	std::lock_guard<spinlock_t> guard(_lock);
	
	if (!_isCpuStorageInitialized[virtualCpuId]) {
		Instrument::enterInitializePrivateReductionStorage(
			/* reductionInfo */ *this,
			DataAccessRegion(cpuStorage, _region.getSize())
		);
		
		_initializationFunction(cpuStorage, _region.getSize());
		_isCpuStorageInitialized[virtualCpuId] = true;
		
		Instrument::exitInitializePrivateReductionStorage(
			/* reductionInfo */ *this,
			DataAccessRegion(cpuStorage, _region.getSize())
		);
	}
	
	Instrument::retrievedPrivateReductionStorage(
		/* reductionInfo */ *this,
		DataAccessRegion(cpuStorage, _region.getSize())
	);
	
	return DataAccessRegion(cpuStorage, _region.getSize());
}

bool ReductionInfo::combineRegion(const DataAccessRegion& region, const boost::dynamic_bitset<>& reductionCpuSet) {
	const long nCpus = CPUManager::getTotalCPUs();
	assert(nCpus > 0);
	
	assert(reductionCpuSet.size() > 0);
	
	for (size_t i = 0; i < (size_t)nCpus; ++i) {
		if (reductionCpuSet[i]) {
			void *originalRegion = ((char*)_region.getStartAddress()) + ((char*)region.getStartAddress() - (char*)_region.getStartAddress());
			void *cpuStorage = ((char*)_storage.getStartAddress()) + _paddedRegionSize*i + ((char*)region.getStartAddress() - (char*)_region.getStartAddress());
			
			Instrument::enterCombinePrivateReductionStorage(
				/* reductionInfo */ *this,
				DataAccessRegion(cpuStorage, region.getSize()),
				DataAccessRegion(originalRegion, region.getSize())
			);
			
			_combinationFunction(originalRegion, cpuStorage, region.getSize());
			
			Instrument::exitCombinePrivateReductionStorage(
				/* reductionInfo */ *this,
				DataAccessRegion(cpuStorage, region.getSize()),
				DataAccessRegion(originalRegion, region.getSize())
			);
		}
	}
	
	_sizeCounter -= region.getSize();
	
	return _sizeCounter == 0;
}
