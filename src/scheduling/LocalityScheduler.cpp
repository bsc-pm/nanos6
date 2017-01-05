#include "LocalityScheduler.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/CPUManager.hpp"
#include "hardware/HardwareInfo.hpp"
#include "hardware/places/CPUPlace.hpp"
#include "tasks/Task.hpp"
#include "memory/directory/Directory.hpp"
#include "memory/Globals.hpp"

#include <algorithm>
#include <cassert>
#include <mutex>


LocalityScheduler::LocalityScheduler() : SchedulerInterface(true)
{
}

LocalityScheduler::~LocalityScheduler()
{
}


Task *LocalityScheduler::getReplacementTask(__attribute__((unused)) CPU *hardwarePlace)
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


CPU *LocalityScheduler::getLocalityCPU(Task * task)
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
            idleCPU = CPUManager::getLocalityIdleCPU(bestCache);
            if (idleCPU != nullptr) {
                destCache = HardwareInfo::getMemoryNode(bestCache)->getCache();
                task->setCache(destCache);
                return idleCPU;
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
        return CPUManager::getLocalityIdleCPU(bestCache);
    }

    return nullptr;
}

ComputePlace * LocalityScheduler::addReadyTask(Task *task, __attribute__((unused)) ComputePlace *hardwarePlace, __attribute__((unused)) ReadyTaskHint hint)
{
	std::lock_guard<SpinLock> guard(_globalLock);
	_readyTasks.push_front(task);
	
    return getLocalityCPU(task);
	//return CPUManager::getIdleCPU();
}


void LocalityScheduler::taskGetsUnblocked(Task *unblockedTask, __attribute__((unused)) ComputePlace *hardwarePlace)
{
	std::lock_guard<SpinLock> guard(_globalLock);
	_unblockedTasks.push_front(unblockedTask);
}


Task *LocalityScheduler::getReadyTask(__attribute__((unused)) ComputePlace *hardwarePlace, __attribute__((unused)) Task *currentTask)
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
	CPUManager::cpuBecomesIdle((CPU *) hardwarePlace);
	
	return nullptr;
}


ComputePlace *LocalityScheduler::getIdleComputePlace(bool force)
{
	std::lock_guard<SpinLock> guard(_globalLock);
	if (force || !_readyTasks.empty() || !_unblockedTasks.empty()) {
		return CPUManager::getIdleCPU();
	} else {
		return nullptr;
	}
}

