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

#define _unused(x) ((void)(x))

//! Disabling copies for a while
//LocalityScheduler::LocalityScheduler() : SchedulerInterface(true)
LocalityScheduler::LocalityScheduler() : SchedulerInterface(false)
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
    _unused(task);
    //! Check if task has already a cache assigned.
    //GenericCache * destCache = task->getCache();
    //CPU *idleCPU = nullptr;
    //if(destCache == nullptr) {
    //    //! If no cache assigned, ask directory information about locality in each cache.
    //    size_t * cachesData = (size_t *) malloc(MAX_CACHES * sizeof(size_t));
    //    memset(cachesData, 0, MAX_CACHES*sizeof(size_t));
    //    Directory::analyze(task->getDataAccesses(), cachesData);
    //    int bestCache = 0;
    //    int oldBestCache = -1;
    //    size_t max = cachesData[0];
    //    //! Get the index of the cache with best locality.
    //    for(int i=0; i<MAX_CACHES; i++) {
    //        if(cachesData[i] > max) {
    //            max = cachesData[i];
    //            bestCache = i;
    //        }
    //    }

    //    //! Try to get a CPU which can access to the best cache.
    //    while(idleCPU == nullptr) {
    //        idleCPU = CPUManager::getLocalityIdleCPU(bestCache);
    //        if (idleCPU != nullptr) {
    //            destCache = HardwareInfo::getMemoryNode(bestCache)->getCache();
    //            task->setCache(destCache);
    //            return idleCPU;
    //        }

    //        //! If no CPU has been found with access to the best cache, try to find the next best cache until finding a idleCPU 
    //        //! that can access the chosen cache.
    //        cachesData[bestCache] = 0;
    //        max = cachesData[0];
    //        oldBestCache = bestCache;
    //        bestCache = 0;
    //        /* The loop can start at 1 because the 0 is done in the "initialization" */
    //        for(int i=1; i<MAX_CACHES; i++) {
    //            if(cachesData[i] > max) {
    //                max = cachesData[i];
    //                bestCache = i;
    //            }
    //        }
    //        if(bestCache == oldBestCache)
    //            return nullptr;
    //    }
    //}
    //else{
    //    int bestCache = task->getCache()->getIndex();
    //    return CPUManager::getLocalityIdleCPU(bestCache);
    //}

    return nullptr;
}

ComputePlace * LocalityScheduler::addReadyTask(Task *task, __attribute__((unused)) ComputePlace *hardwarePlace, __attribute__((unused)) ReadyTaskHint hint)
{
    /** ALL THIS WORK CAN BE DONE WITHOUT THE LOCK **/

    /* Given that there must be a ready queue per NUMA node, some 
       homeNode analysis must be done to know where to enqueue 
       the task. Directory provides a method for that purpose.
     */
    std::vector<double> scores = Directory::computeNUMANodeAffinity(task->getDataAccesses());
    /* Iterate over the vector (as the vector size shouldn't be so big, iterating over the vector 
       shouldn't be costly) to choose the biggest score.
     */
    double max_score = scores[0];
    unsigned int best_node = 0;
    /* The loop can start at 1 because the 0 is done in the "initialization" */
    for(unsigned int i=1; i<scores.size(); i++) {
        if(scores[i] > max_score) {
            best_node = i;
            max_score = scores[i];
        }
    }

    /** LOCK NEEDED **/
	std::lock_guard<SpinLock> guard(_globalLock);
    /* Enqueue in the readyQueue corresponding to best_node.
     */
    _readyQueues[best_node].push_front(task);
	
    return CPUManager::getNUMALocalityIdleCPU(best_node);
    //return getLocalityCPU(task);
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
	//if (!_readyTasks.empty()) {
	//	task = _readyTasks.front();
	//	_readyTasks.pop_front();
    size_t NUMANodeId = ((CPU *) hardwarePlace)->_NUMANodeId;
    std::deque<Task *> &readyQueue = _readyQueues[NUMANodeId];
    if (!readyQueue.empty()) {
        //task = readyQueue.front();
        //readyQueues.pop_front();
        double max_score = 0.0;
        std::deque<Task*>::iterator toErase = readyQueue.end();
		for(std::deque<Task*>::iterator it = readyQueue.begin(); it != readyQueue.end(); it++) {
            double score = Directory::computeTaskAffinity(*it, NUMANodeId);
            if(score >= max_score) {
                max_score = score;
                toErase = it;
                task = *it;
            }
        }
        readyQueue.erase(toErase);
        //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
        //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
        //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
        std::cerr << "Going to execute task " << task->getTaskInfo()->task_label << " with score " << max_score << "." << std::endl;
        //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
        //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
        //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;

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
	//if (force || !_readyTasks.empty() || !_unblockedTasks.empty()) {
    bool remainingTasks = false;
    for(auto&& readyQueue : _readyQueues) {
        if(!readyQueue.second.empty()) {
            remainingTasks = true;
            break;
        }
    }
	if (force || remainingTasks || !_unblockedTasks.empty()) {
		return CPUManager::getIdleCPU();
	} else {
		return nullptr;
	}
}


void LocalityScheduler::addReadyQueue(std::size_t node_id) {
    std::deque<Task *> readyQueue;
    _readyQueues[node_id] = readyQueue;
}
