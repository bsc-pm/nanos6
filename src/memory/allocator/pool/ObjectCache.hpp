#ifndef __OBJECT_CACHE_HPP__
#define __OBJECT_CACHE_HPP__

#include "lowlevel/SpinLock.hpp"
#include "hardware/HardwareInfo.hpp"
#include "NUMAObjectCache.hpp"
#include "CPUObjectCache.hpp"
#include "executors/threads/CPU.hpp"
#include "executors/threads/WorkerThread.hpp"

template<typename T>
class ObjectCache {
	
	/** An object cache is built in two layers, one CPU and one NUMA layer.
	 *
	 * Allocations will happen through the local CPU cache, or the external
	 * object cache. The CPUObjectCache will invoke the NUMAObjectCache to
	 * get more objects if it runs out of objects.
	 *
	 * Deallocations will happen to the CPUObjectCache of the current CPU.
	 * If the object does not belong to that CPUObjectCache (it belongs in
	 * a different NUMA node) the object will be returned to the
	 * NUMAObjectCache in order to be used from the CPUObjectCache of the
	 * respective NUMA node. */
	NUMAObjectCache<T> *_NUMACache;
	std::vector<CPUObjectCache<T> *> _CPUCaches;
	CPUObjectCache<T> *_externalObjectCache;
	SpinLock _externalLock;
	
public:
	ObjectCache()
	{
		size_t numaNodeCount = HardwareInfo::getMemoryPlaceCount(nanos6_device_t::nanos6_host_device);
		size_t cpuCount = HardwareInfo::getComputePlaceCount(nanos6_device_t::nanos6_host_device);
		HostInfo *deviceInfo = (HostInfo *)HardwareInfo::getDeviceInfo(nanos6_device_t::nanos6_host_device);
		std::vector<ComputePlace *> cpus = deviceInfo->getComputePlaces();
		
		_NUMACache = new NUMAObjectCache<T>(numaNodeCount);
		_CPUCaches.resize(cpuCount);
		for (size_t i = 0; i < cpuCount; ++i) {
			CPU *cpu = (CPU *)cpus[i];
			_CPUCaches[i] = new CPUObjectCache<T>(
						_NUMACache,
						cpu->_NUMANodeId,
						numaNodeCount
					);
		}
		_externalObjectCache = new CPUObjectCache<T>(
						_NUMACache,
						/* NUMA Id */ 0,
						numaNodeCount
					);
	}
	
	~ObjectCache()
	{
		delete _NUMACache;
		for (auto it : _CPUCaches) {
			delete it;
		}
		delete _externalObjectCache;
	}
	
	template<typename... TS>
	inline T *newObject(TS &&... args)
	{
		CPU *cpu = nullptr;
		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		if (thread != nullptr) {
			cpu = thread->getComputePlace();
		}
		
		if (thread == nullptr || cpu == nullptr) {
			std::lock_guard<SpinLock> guard(_externalLock);
			return _externalObjectCache->newObject(std::forward<TS>(args)...);
		} else {
			size_t cpuId = cpu->_virtualCPUId;
			return _CPUCaches[cpuId]->newObject(std::forward<TS>(args)...);
		}
	}
	
	inline void deleteObject(T *ptr)
	{
		CPU *cpu = nullptr;
		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		if (thread != nullptr) {
			cpu = thread->getComputePlace();
		}
		
		if (thread == nullptr || cpu == nullptr) {
			std::lock_guard<SpinLock> guard(_externalLock);
			_externalObjectCache->deleteObject(ptr);
		} else {
			size_t cpuId = cpu->_virtualCPUId;
			_CPUCaches[cpuId]->deleteObject(ptr);
		}
	}
};

#endif /* __OBJECT_CACHE_HPP__ */
