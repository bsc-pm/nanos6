#ifndef NUMA_CACHE_CPP
#define NUMA_CACHE_CPP

#include <set>
#include "NUMACache.hpp" 
#include "../../hardware/Machine.hpp" 

void * NUMACache::allocate(std::size_t size) {
    return malloc(size);
}

void NUMACache::deallocate(void * ptr) {
    free(ptr);
}

void NUMACache::copyData(unsigned int sourceCache, unsigned int homeNode, Task task /*, unsigned int copiesToDo */) {
    //! TODO: Think about concurrency...
    unsigned int copiesDone = 0;

    //! Iterate over the task data accesses to check if they are already in the cache
    for(auto it = task.getDataAccesses()._accesses.begin(); it != task.getDataAccesses()._accesses.end(); it++ ) {
        //! TODO: Do a first round to block the data that is already in the cache to avoid evictions of that data.
        auto it2 = _replicas.find((*it).getAccessRange().getStartAddress());
        bool bringData = false;
        if(it2 != _replicas.end()) {
            //! The data is already in the cache. Check with the directory whether it is correct or outdated. 
            //if(it2->second._version != Directory::getVersion(it2->first)) {
                //! The data in the cache is outdated.
                //bringData = true;
            //}
            //else {
                //! The data in the cache is correct.
                //! Increment the _refCount to avoid evictions.
                //it2->second._refCount++;
                //! Update _lastUse.
                //it2->second._lastUse = ++_count;
            //}
        }
        else {
            //! The data is not in the cache yet. Bring it.
            bringData = true;
        }

        if(bringData) {
            //! Allocate
            std::size_t replicaSize = (*it).getAccessRange().getSize();
            void * replicaAddress = allocate(replicaSize);
            if(replicaAddress == NULL) { 
                //! If allocate fails it means that there is no space left in the cache. Perform evictions.
                //! TODO: Add some extra cutoff? This can imply deadlock.
                while(replicaAddress == NULL) {
                    bool canEvict = evict();
                    replicaAddress = allocate(replicaSize);
                    if(!canEvict) {
                        //! No more evictions are possible at this moment. Fatal error?
                        return;
                    }
                }
            }

            //! Copy data
            replicaInfo_t newReplica;
            //! Try to do it from the sourceCache provided by the directory/scheduler.
            //! Ask the sourceCache for the data
            replicaInfo_t * sourceReplica = Machine::getMachine()->getMemoryNode(sourceCache)->getCache()->getReplicaInfo((*it).getAccessRange().getStartAddress());
            if(sourceReplica->_physicalAddress == NULL) {
                //! sourceCache failed, do it from homeNode
                //! TODO: this address is the same than (*it).getAccessRange().getStartAddress()? The homeNode is equivalent to the user address space?
                sourceReplica = Machine::getMachine()->getMemoryNode(homeNode)->getCache()->getReplicaInfo((*it).getAccessRange().getStartAddress());
                if(sourceReplica->_physicalAddress == NULL)
                    //! homeNode also failed, nothing to do. 
                    //! TODO: homeNode should always have the data, so maybe this is a fatal error
                    return;
            }

            //! Actually perform the copy
            memcpy(replicaAddress, sourceReplica->_physicalAddress, replicaSize);
            //! Update replicaInfo
            newReplica._physicalAddress = replicaAddress;
            newReplica._size = replicaSize;
            newReplica._dirty = (*it)._type == READ_ACCESS_TYPE ? false : true ;
            newReplica._version = sourceReplica->_version + newReplica._dirty;
            newReplica._refCount++;
            newReplica._lastUse = ++_count;
            //! Insert into _replicas
            _replicas[(*it).getAccessRange().getStartAddress()] = newReplica;

            //! Notify insert to directory
            //! TODO: directory returns version. Check wether it is okay with ours.
            // unsigned int dirVersion = directory.insert(whatever, newReplica._dirty);
            // if(dirVersion != newReplica._version) fatalError? 

            //! Update the number of cachedBytes in the task.
            unsigned int cachedBytes = task.addCachedBytes(replicaSize);

            ++copiesDone;
        }
        
        if(copiesDone /* >= copiesToDo */ ) 
            return;
    }
}

void NUMACache::flush() {
    //! Iterate over all the replicas and copy them back to the user address space if dirty. Then, clear the cache. 
    //! Clearing the cache is needed because a flush is performed in a taskwait. At this moment, the user recovers 
    //! the control of the data and the runtime knows nothing about it. If after a taskwait, the user issues a task 
    //! with some previously cached data, this data could be changed between the taskwait and the new task, so we 
    //! must copy it again from the user address space. Hence, it is useless to maintain the data in the cache after 
    //! a flush.
    for(auto& replica : _replicas) {
        //! The replica is dirty and it is the last version, copy back.
        if(replica.second._dirty /*&& replica.second._version == Directory::getVersion(replica.first) */) {
            //! TODO: Copy back to homeNode or to user address space??
            //! First, copy back to user address space.
            memcpy(replica.first, replica.second._physicalAddress, replica.second._size); 
            //! Inform directory the data is not in the cache anymore.
            //Directory::remove(candidate.second);
        }
        //! Remove replica from _replicas
        _replicas.erase(replica.first);
        //! Free space of the replica
        deallocate(replica.second._physicalAddress);
    }
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
        replicaInfo_t * replica = &_replicas[candidate->second];
        //! The replica is dirty and it is the last version.
        //! TODO: For performance, it could be also checked if the data is anywhere else. If it is, avoid copy back.
        if(replica->_dirty /*&& replica->_version == Directory::getVersion(candidate->second)*/) {
            //! TODO: Copy back to homeNode or to user address space??
            //! First, copy back to user address space.
            memcpy(candidate->second, replica->_physicalAddress, replica->_size); 
            //! Inform directory the data is not in the cache anymore.
            //Directory::remove(candidate->second);
        }
        //! Remove replica from _replicas
        _replicas.erase(candidate->second);
        //! Free space of the replica
        deallocate(replica->_physicalAddress);

        return true;
    }
}

#endif //NUMA_CACHE_CPP
