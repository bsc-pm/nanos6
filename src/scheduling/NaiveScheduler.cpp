#include "NaiveScheduler.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "hardware/HardwareInfo.hpp"
#include "hardware/places/CPUPlace.hpp"
#include "tasks/Task.hpp"
#include "memory/directory/Directory.hpp"
#include "memory/Globals.hpp"

#include <algorithm>
#include <cassert>
#include <mutex>


NaiveScheduler::NaiveScheduler()
{
}

NaiveScheduler::~NaiveScheduler()
{
}


Task *NaiveScheduler::getReplacementTask(__attribute__((unused)) CPU *hardwarePlace)
{
	if (!_unblockedTasks.empty()) {
		Task *replacementTask = _unblockedTasks.front();
		_unblockedTasks.pop_front();
		
		assert(replacementTask != nullptr);
		
		return replacementTask;
	} else {
		return nullptr;
	}
}


void NaiveScheduler::cpuBecomesIdle(CPU *cpu)
{
	_idleCPUs.push_front(cpu);
}


CPU *NaiveScheduler::getIdleCPU()
{
	if (!_idleCPUs.empty()) {
		CPU *idleCPU = _idleCPUs.front();
		_idleCPUs.pop_front();
		
		return idleCPU;
	}
	
	return nullptr;
}

CPU *NaiveScheduler::getLocalityCPU(Task * task)
{
    //! Check if task has already a cache assigned.
    GenericCache * destCache = task->getCache();
    CPU *idleCPU = nullptr;
    if(destCache == nullptr) {
        //! If no cache assigned, ask directory information about locality in each cache.
        size_t * cachesData = (size_t *) malloc(MAX_CACHES * sizeof(size_t));
        memset(cachesData, 0, MAX_CACHES*sizeof(size_t));
        Directory::analyze(task->getDataAccesses(), cachesData);
        int bestCache = 0;
        int oldBestCache = -1;
        size_t max = cachesData[0];
        //! Get the index of the cache with best locality.
        for(int i=0; i<MAX_CACHES; i++) {
            if(cachesData[i] > max) {
                max = cachesData[i];
                bestCache = i;
            }
        }
        //! Try to get a CPU which can access to the best cache.
        while(idleCPU == nullptr) {
            //! Iterate over all the idleCPUs until finding one associated with the best cache.
            for(unsigned int i = 0; i < _idleCPUs.size(); i++) {
                idleCPU = _idleCPUs[i];
                if(idleCPU->getMemoryPlace(bestCache) != nullptr) {
                    //! idleCPU with access to the best cache found. Erase the CPU from idleCPUs and assign the cache to the task.
                    _idleCPUs.erase(_idleCPUs.begin()+i);
                    destCache = HardwareInfo::getMemoryNode(bestCache)->getCache();
                    task->setCache(destCache);
                    return idleCPU;
                }
                idleCPU = nullptr;
            }
            //! If no CPU has been found with access to the best cache, try to find the next best cache until finding a idleCPU 
            //! that can access the chosen cache.
            cachesData[bestCache] = 0;
            max = cachesData[0];
            oldBestCache = bestCache;
            bestCache = 0;
            for(int i=0; i<MAX_CACHES; i++) {
                if(cachesData[i] > max) {
                    max = cachesData[i];
                    bestCache = i;
                }
            }
            if(bestCache == oldBestCache)
                return nullptr;
        }
    }
    else{
        int bestCache = task->getCache()->getIndex();
        for(unsigned int i = 0; i < _idleCPUs.size(); i++) {
            idleCPU = _idleCPUs[i];
            if(idleCPU->getMemoryPlace(bestCache) != nullptr) {
                //! idleCPU with access to the best cache found. Erase the CPU from idleCPUs and assign the cache to the task.
                _idleCPUs.erase(_idleCPUs.begin()+i);
                return idleCPU;
            }
            idleCPU = nullptr;
        }
    }

    return nullptr;
}

ComputePlace * NaiveScheduler::addReadyTask(Task *task, __attribute__((unused)) ComputePlace *hardwarePlace, __attribute__((unused)) ReadyTaskHint hint)
{
	std::lock_guard<SpinLock> guard(_globalLock);
	_readyTasks.push_front(task);
	
    return getLocalityCPU(task);
	//return getIdleCPU();
}


void NaiveScheduler::taskGetsUnblocked(Task *unblockedTask, __attribute__((unused)) ComputePlace *hardwarePlace)
{
	std::lock_guard<SpinLock> guard(_globalLock);
	_unblockedTasks.push_front(unblockedTask);
}


Task *NaiveScheduler::getReadyTask(__attribute__((unused)) ComputePlace *hardwarePlace, __attribute__((unused)) Task *currentTask)
{
	Task *task = nullptr;
	
	std::lock_guard<SpinLock> guard(_globalLock);
	
	// 1. Get an unblocked task
	task = getReplacementTask((CPU *) hardwarePlace);
	if (task != nullptr) {
		return task;
	}
	
	// 2. Or get a ready task
	if (!_readyTasks.empty()) {
		task = _readyTasks.front();
		_readyTasks.pop_front();
		
		assert(task != nullptr);
		
		return task;
	}
	
	// 3. Or mark the CPU as idle
	cpuBecomesIdle((CPU *) hardwarePlace);
	
	return nullptr;
}


ComputePlace *NaiveScheduler::getIdleComputePlace(bool force)
{
	std::lock_guard<SpinLock> guard(_globalLock);
	if (force || !_readyTasks.empty() || !_unblockedTasks.empty()) {
		return getIdleCPU();
	} else {
		return nullptr;
	}
}

