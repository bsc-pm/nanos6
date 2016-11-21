#ifndef NUMA_CACHE_CPP
#define NUMA_CACHE_CPP

#include <iostream>
#include <set>
#include <cassert>
#include "NUMACache.hpp" 
#include "../../hardware/Machine.hpp" 
#include "../directory/Directory.hpp"

void * NUMACache::allocate(std::size_t size) {
    return malloc(size);
}

void NUMACache::deallocate(void * ptr) {
    free(ptr);
}

void NUMACache::copyData(int sourceCache, Task *task , unsigned int copiesToDo = 1) {
	std::lock_guard<SpinLock> guard(_lock);
    unsigned int copiesDone = 0;

    //! Disabled for testing purposes
    //assert(Machine::getMachine()->getMemoryNode(sourceCache)->getCache()!=this);

    //! Check whether there is any data access.
    if(task->getDataSize() == 0) {
        addReadyTask(task);
        if(_verbose) {
            std::cout << "[CACHE " << _index << "]: Task " << task << " is ready to be executed because it has no data to be copied." << std::endl;
        }
        return;
    }
    
    //! Do a first round to block the data that is already in the cache to avoid evictions of that data.
    for(auto it = task->getDataAccesses()._accesses.begin(); it != task->getDataAccesses()._accesses.end(); it++ ) {
        if(!(*it).isCached()) {
            auto it2 = _replicas.find((*it).getAccessRange().getStartAddress());
            if(it2 != _replicas.end()) {
                //! The data is already in the cache. Check with the directory whether it is correct or outdated. 
                if(it2->second._version == Directory::copy_version(it2->first)) {
                    //! The data in the cache is correct.
                    //! Increment the _refCount to avoid evictions.
                    it2->second._refCount++;
                    //! Update _lastUse.
                    it2->second._lastUse = ++_count;
                    //! Update _cachedBytes but it must be done only once.
                    task->addCachedBytes(it2->second._size);
                    ////! Mark data access as cached
                    (*it).setCached(true);
                    if(_verbose) {
                        std::cout << "[CACHE " << _index << "]: DataAccess with address " << it2->first << " and size " << (*it).getAccessRange().getSize() 
                            << " required by task " << task << " is already in the cache and has been blocked." << std::endl;
                    }
                }
                else {
                    if(_verbose) {
                        std::cout << "[CACHE " << _index << "]: DataAccess with address " << it2->first << " and size " << (*it).getAccessRange().getSize() 
                            << " required by task " << task << " is already in the cache but current version does not match directory version." << std::endl;
                    }
                }
            }
        }
    }

    //! Iterate over the task data accesses to check if they are already in the cache
    for(auto it = task->getDataAccesses()._accesses.begin(); it != task->getDataAccesses()._accesses.end(); it++ ) {
        //! Check whether all the copies are cached. If so, add the task to the ready queue.
        if(!task->hasPendingCopies()) {
            if(_verbose) {
                std::cout << "[CACHE " << _index << "]: Task " << task << " is ready to be executed because it does not have pending copies." << std::endl;
            }
            break;
        }

        //! Check whether this thread has already done all the desired copies.
        if(copiesDone >= copiesToDo) { 
            if(_verbose) {
                std::cout << "[CACHE " << _index << "]: Task " << task << " is going back to ready queue because it has performed all the desired copies." << std::endl;
            }
            break;
        }

        //! The data is not in the cache yet. Bring it.
        if(!(*it).isCached()) {
            if(_verbose) {
                std::cout << "[CACHE " << _index << "]: Data with startAddress " << (*it).getAccessRange().getStartAddress() << " and size " 
                    << (*it).getAccessRange().getSize() << " required by task " << task
                    << " is not in the cache yet so we should bring it. First, try to allocate data in the cache." << std::endl;
            }
            //! Allocate
            std::size_t replicaSize = (*it).getAccessRange().getSize();
            void * replicaAddress = allocate(replicaSize);
            if(replicaAddress == nullptr || _replicas.size() >= 5) { 
                if(_verbose) {
                    std::cout << "[CACHE " << _index << "]: Data with startAddress " << (*it).getAccessRange().getStartAddress() << " and size " 
                        << (*it).getAccessRange().getSize() << " required by task " << task << " has failed allocation. Try to perform evictions." << std::endl;
                }
                //! If allocate fails it means that there is no space left in the cache. Perform evictions.
                //! TODO: The extra restriction of only 6 replicas is just for debug purposes!!
                while(replicaAddress == nullptr || _replicas.size() >= 5) {
                    bool canEvict = evict();
                    replicaAddress = allocate(replicaSize);
                    if(!canEvict) { 
                        if(_verbose)
                            std::cout << "[CACHE " << _index << "]: Data with startAddress " << (*it).getAccessRange().getStartAddress() << " and size " 
                                << (*it).getAccessRange().getSize() << " required by task " << task 
                                << " has failed allocation and cannot perform more evictions. Try releasing its own copies." << std::endl;
                        releaseCopies(task);
                        addReadyTask(task);
                        return;
                    }
                }
                if(_verbose) {
                    std::cout << "[CACHE " << _index << "]: Data with startAddress " << (*it).getAccessRange().getStartAddress() << " and size " 
                        << (*it).getAccessRange().getSize() << " required by task " << task << " had successful allocation." << std::endl;
                }
            }

            //! Copy data
            replicaInfo_t newReplica;
            //! Try to do it from the sourceCache provided by the directory/scheduler.
            //! Ask the sourceCache for the data
            if(_verbose) {
                std::cout << "[CACHE " << _index << "]: Try to bring data with startAddress " << (*it).getAccessRange().getStartAddress() << " and size " 
                    << (*it).getAccessRange().getSize() << " required by task " << task << " from sourceCache provided by the Scheduler." << std::endl;
            }
            replicaInfo_t * sourceReplica;
            if(sourceCache != 0) 
                sourceReplica = Machine::getMachine()->getMemoryNode(sourceCache)->getCache()->getReplicaInfo((*it).getAccessRange().getStartAddress());
            else
                sourceReplica->_physicalAddress = nullptr;
            if(sourceReplica->_physicalAddress == nullptr) {
                if(_verbose) {
                    std::cout << "[CACHE " << _index << "]: Try to bring data with startAddress " << (*it).getAccessRange().getStartAddress() << " and size " 
                        << (*it).getAccessRange().getSize() << " required by task " << task 
                        << " from sourceCache provided by the Scheduler has failed. Try from homeNode." << std::endl;
                }
                //! sourceCache failed, do it from homeNode
                //! TODO: this address is the same than (*it).getAccessRange().getStartAddress()? The homeNode is equivalent to the user address space?
                int homeNode = (*it)._homeNode; 
                //sourceReplica = Machine::getMachine()->getMemoryNode(homeNode)->getCache()->getReplicaInfo((*it).getAccessRange().getStartAddress());
                //if(sourceReplica->_physicalAddress == nullptr)
                //    //! homeNode also failed, nothing to do. 
                //    //! TODO: homeNode should always have the data, so maybe this is a fatal error
                //    return;
                //! TEMPORARY!!! Copy from user address space.
                sourceReplica->_physicalAddress = (*it).getAccessRange().getStartAddress();
                sourceReplica->_version = 0;
            }

            if(_verbose) {
                std::cout << "[CACHE " << _index << "]: Actually bring data with startAddress " << (*it).getAccessRange().getStartAddress() << " and size " 
                    << (*it).getAccessRange().getSize() << " required by task " << task << "." << std::endl;
            }
            //! Actually perform the copy
            memcpy(replicaAddress, sourceReplica->_physicalAddress, replicaSize);
            //! Update replicaInfo
            newReplica._physicalAddress = replicaAddress;
            newReplica._size = replicaSize;
            newReplica._dirty = (*it)._type == READ_ACCESS_TYPE ? false : true ;
            newReplica._version = sourceReplica->_version + newReplica._dirty;
            newReplica._refCount = 1;
            newReplica._lastUse = ++_count;
            //! Insert into _replicas
            _replicas[(*it).getAccessRange().getStartAddress()] = newReplica;

            //! Notify insert to directory
            //! Directory returns version. Check wether it is okay with ours.
            if(_verbose) {
                std::cout << "[CACHE " << _index << "]: Notify directory data with startAddress " << (*it).getAccessRange().getStartAddress() << " and size " 
                    << (*it).getAccessRange().getSize() << " required by task " << task << " has been inserted." << std::endl;
            }
            int dirVersion = Directory::insert_copy((*it).getAccessRange().getStartAddress(), replicaSize, _index, newReplica._dirty);
            if(dirVersion != newReplica._version) { 
                if(_verbose) {
                    std::cout << "[CACHE " << _index << "]: Data with startAddress " << (*it).getAccessRange().getStartAddress() << " and size " 
                        << (*it).getAccessRange().getSize() << " required by task " << task 
                        << " copied from sourceCache/homeNode does not match directory version." << std::endl;
                }
                //! If versions does not match, the data copied is probably outdated. FatalError?
                //! Use the version given by the directory.
                newReplica._version = dirVersion;
            }

            //! Update the number of cachedBytes in the task->
            task->addCachedBytes(replicaSize);

            ////! Mark data access as cached
            (*it).setCached(true);

            ++copiesDone;
        }
    }

    //! Check whether all the copies are cached. If so, add the task to the ready queue.
    //if(!task->hasPendingCopies()) {
    //! As of now, there is no preready and ready differentiation in the queue, so add it always.
        addReadyTask(task);
    //}
}

void NUMACache::flush() {
    //! Iterate over all the replicas and copy them back to the user address space if dirty. Then, clear the cache. 
    //! Clearing the cache is needed because a flush is performed in a taskwait. At this moment, the user recovers 
    //! the control of the data and the runtime knows nothing about it. If after a taskwait, the user issues a task 
    //! with some previously cached data, this data could be changed between the taskwait and the new task, so we 
    //! must copy it again from the user address space. Hence, it is useless to maintain the data in the cache after 
    //! a flush.
	std::lock_guard<SpinLock> guard(_lock);
    for(auto& replica : _replicas) {
        if(_verbose) {
            std::cout << "[CACHE " << _index << "]: Data with address " << replica.first << " is going to be flushed." << std::endl;
        }
        //! The replica is dirty and it is the last version, copy back.
        if(replica.second._dirty && replica.second._version == Directory::copy_version(replica.first)) {
            if(_verbose) {
                std::cout << "[CACHE " << _index << "]: Data with address " << replica.first 
                    << " is dirty and matches last version of the directory, so it has to be copied back to homeNode and notify directory before flushing." << std::endl;
            }
            //! TODO: Copy back to homeNode (The homeNode can be found in the TaskDataAccesses) 
            //! First, copy back to user address space.
            memcpy(replica.first, replica.second._physicalAddress, replica.second._size); 
            //! Inform directory the data is not in the cache anymore.
            Directory::erase_copy(replica.first, _index);
        }
        //! Free space of the replica
        deallocate(replica.second._physicalAddress);
        if(_verbose) {
            std::cout << "[CACHE " << _index << "]: Data with address " << replica.first << " has been flushed." << std::endl;
        }
    }
    //! Clear _replicas
    _replicas.clear();
}

bool NUMACache::evict() {
    std::set< std::pair<long unsigned int, void *> > candidates;
    //! Look for candidates to be evicted.
    for(auto& replica : _replicas) {
        //! Iterate through all the replicas to look for those that are not being used now.
        if(replica.second._refCount == 0) {
            //! If they are not in use, store them in a set that orders by the lastUse.
            //! This way, the first elements of the set are those least recently used.
            candidates.insert(std::make_pair(replica.second._lastUse, replica.first));
        }
    }
    if(candidates.empty()) {
        return false;
    }
    else {
        //! Evict the least recently used candidate.
        auto candidate = candidates.begin();
        if(_verbose) {
            std::cout << "[CACHE " << _index << "]: Data with address " << candidate->second << " is going to be evicted." << std::endl; 
        }
        replicaInfo_t * replica = &_replicas[candidate->second];
        //! The replica is dirty and it is the last version.
        //! TODO: For performance, it could be also checked if the data is anywhere else. If it is, avoid copy back.
        if(replica->_dirty && replica->_version == Directory::copy_version(candidate->second)) {
            if(_verbose) {
                std::cout << "[CACHE " << _index << "]: Data with address " << candidate->second 
                    << " is dirty and matches last version of the directory, so it has to be copied back to homeNode and notify directory before evicting." << std::endl;
            }
            //! TODO: Copy back to homeNode (The homeNode can be found in the TaskDataAccesses) 
            //! First, copy back to user address space.
            memcpy(candidate->second, replica->_physicalAddress, replica->_size); 
            //! Inform directory the data is not in the cache anymore.
            Directory::erase_copy(candidate->second, _index);
        }
        //! Remove replica from _replicas
        _replicas.erase(candidate->second);
        //! Free space of the replica
        deallocate(replica->_physicalAddress);

        if(_verbose) {
            std::cout << "[CACHE " << _index << "]: Data with address " << candidate->second << " has been evicted." << std::endl;
        }

        return true;
    }
}

#endif //NUMA_CACHE_CPP
