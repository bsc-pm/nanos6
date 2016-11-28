#ifndef NUMA_CACHE_CPP
#define NUMA_CACHE_CPP

#include <iostream>
#include <set>
#include <cassert>
#include "NUMACache.hpp" 
#include "hardware/Machine.hpp" 
#include "memory/directory/Directory.hpp"

void * NUMACache::allocate(std::size_t size) {
    return malloc(size);
}

void NUMACache::deallocate(void * ptr) {
    free(ptr);
}

void NUMACache::copyData(int sourceCache, Task *task, unsigned int copiesToDo = 1) {
    assert(task->hasPendingCopies() && "task without pending copies requesting copyData");
    if(task->getDataSize() == 0 || copiesToDo == 0) {
        addReadyTask(task);
        if(_verbose) {
            std::stringstream ss;
            ss << "Task " << task << " cannot perform any copy"; 
            verboseMsg(ss.str()); 
        }
        return;
    }

    //! Do a first round to block the data that is already in the cache to avoid evictions of that data.
    //! No actual copies are done in this loop.
    for(DataAccess& dataAccess : task->getDataAccesses()._accesses) {
        //! If access already marked as cached, ignore it. Just process those not cached yet.
        if(!dataAccess.isCached()) {
            replicas_t::iterator replica = _replicas.find(dataAccess.getAccessRange().getStartAddress()); 
            //! Check if data is already in the cache.
            if(replica != _replicas.end()) {
                //! Data is already in the cache.
                //! All cases needs to lock data, do it.
                replica->second._refCount++;
                //! out access type -> does not need up to date version -> Update replicaInfo+directory and lock data (already done).
                //! in/inout access type -> needs last version 
                //!     *if data is up to date -> Update replicaInfo+directory and lock data (already done).
                //!     *if data is outdated -> Just lock data (already done), the copy will be performed in the main loop.
                if(dataAccess._type == WRITE_ACCESS_TYPE || 
                   (Directory::getVersion(dataAccess.getAccessRange().getStartAddress()) == replica->second._version)) {
                    //! (out access) or (in/inout access and up to date).
                    //! Update replicaInfo+directory and lock data (already done). 
                    replica->second._dirty = (dataAccess._type != READ_ACCESS_TYPE);
                    replica->second._version += replica->second._dirty;
                    replica->second._lastUse = ++_count;
                    int dirVersion = Directory::insertCopy(/*address*/ dataAccess.getAccessRange().getStartAddress(), 
                                                           /*size*/ dataAccess.getAccessRange().getSize(),
                                                           /*homeNode*/ dataAccess._homeNode,
                                                           /*cache index*/ _index, 
                                                           /*increment*/ dataAccess._type != READ_ACCESS_TYPE);
                    assert(dirVersion==replica->second._version && "Versions must match");
                    //! Update task cachedBytes.
                    task->addCachedBytes(dataAccess.getAccessRange().getSize());
                    //! Mark dataAccess as cached.
                    dataAccess.setCached(true);
                }
            }
        }
    }

    unsigned int copiesDone = 0;

    //! Main loop. Copies are performed here. 
    for(DataAccess& dataAccess : task->getDataAccesses()._accesses) {
        if(!task->hasPendingCopies() || copiesDone >= copiesToDo)
            break;
        //! If access already marked as cached, ignore it. Just process those not cached yet.
        if(!dataAccess.isCached()) {
            replicas_t::iterator replica = _replicas.find(dataAccess.getAccessRange().getStartAddress()); 
            //! Check if data is already in the cache.
            if(replica != _replicas.end()) {
                //! Data is already in the cache. However, if we have arrived here with data marked as not cached, it means that 
                //! data is outdated, otherwise first round would have mark it as cached. Therefore, bring last version of data.
                //! Moreover, only in/inout accesses can reach this point of the code so this check is not needed. This is because 
                //! if it is an out access, in case of being in the cache, first round has already marked it as cached, in case of 
                //! not being present in the cache, it goes to the else branch.
                //! Assertions just for debug.
                //assert(replica->second._version == Directory::getVersion(dataAccess.getAccessRange().getStartAddress()) && "Versions must match");
                //assert(dataAccess._type != WRITE_ACCESS_TYPE && "out copies cannot reach this point");
                //! Bring data and update replicaInfo+directory.
                //! Data can be found either in sourceCache provided by the scheduler or in the homeNode.
                assert(_index != sourceCache && "When sourceCache and cache are the same, copy is not required");
                assert(_index != dataAccess._homeNode && "When data's homenode and cache are the same, copy is not required");
                replicaInfo_t * sourceReplica = nullptr;
                if(sourceCache != -1)
                    sourceReplica = Machine::getMemoryNode(sourceCache)->getCache()->getReplicaInfo(dataAccess.getAccessRange().getStartAddress());
                if(sourceReplica->_physicalAddress == nullptr) {
                    //! sourceCache failed, do it from homeNode
                    sourceReplica = Machine::getMemoryNode(dataAccess._homeNode)->getCache()->getReplicaInfo(dataAccess.getAccessRange().getStartAddress());
                    //! homeNode also failed, nothing to do. 
                    assert(sourceReplica->_physicalAddress != nullptr && "Cannot copy data from homeNode");
                }
                //! TODO: memcpy should be wrapped!!
                memcpy(replica->second._physicalAddress, sourceReplica->_physicalAddress, dataAccess.getAccessRange().getSize());
                replica->second._dirty = sourceReplica->_dirty || (dataAccess._type != READ_ACCESS_TYPE);
                replica->second._version = sourceReplica->_version + replica->second._dirty;
                replica->second._lastUse = ++_count;
                int dirVersion = Directory::insertCopy(/*address*/ dataAccess.getAccessRange().getStartAddress(), 
                        /*size*/ dataAccess.getAccessRange().getSize(),
                        /*homeNode*/ dataAccess._homeNode,
                        /*cache index*/ _index, 
                        /*increment*/ replica->second._dirty);
                assert(dirVersion==replica->second._version && "Versions must match");
            }
            else{
                //! Data is not in the cache.
                replicaInfo_t newReplica;
                newReplica._physicalAddress = nullptr;
                if(dataAccess._homeNode != _index) {
                    //! If data's homeNode is not this cache, allocate space in the cache and physicalAddress is this allocated space.
                    //! In this case, it is evictable.
                    //! Create new replicaInfo and fill it.
                    newReplica._size = dataAccess.getAccessRange().getSize();
                    newReplica._version = 0;
                    newReplica._dirty = (dataAccess._type != READ_ACCESS_TYPE);
                    newReplica._refCount = 1;
                    newReplica._lastUse = ++_count;
                    //newReplica._evictable = true;
                    //allocate
                    void * replicaAddress = allocate(dataAccess.getAccessRange().getSize());
                    newReplica._physicalAddress =  replicaAddress; 
                    bool allocated = (replicaAddress == nullptr) /*debug purposes*/ || (_replicas.size() >= 5);
                    if(!allocated) {
                        while(!allocated) {
                            bool canEvict = evict();
                            if(!canEvict) {
                                releaseCopies(task);
                                addReadyTask(task);
                                return;
                            }
                            replicaAddress = allocate(dataAccess.getAccessRange().getSize());
                            allocated = (replicaAddress == nullptr) /*debug purposes*/ || (_replicas.size() >= 5);
                        }
                    }
                    if(dataAccess._type != WRITE_ACCESS_TYPE) {
                        //! Moreover, if access is in/inout, we need the data up to date. Bring it.
                        //! Data can be found either in sourceCache provided by the scheduler or in the homeNode.
                        assert(_index != sourceCache && "When sourceCache and cache are the same, copy is not required");
                        assert(_index != dataAccess._homeNode && "When data's homenode and cache are the same, copy is not required");
                        replicaInfo_t * sourceReplica = nullptr;
                        if(sourceCache != -1)
                            sourceReplica = Machine::getMemoryNode(sourceCache)->getCache()->getReplicaInfo(dataAccess.getAccessRange().getStartAddress());
                        if(sourceReplica->_physicalAddress == nullptr) {
                            //! sourceCache failed, do it from homeNode
                            sourceReplica = Machine::getMemoryNode(dataAccess._homeNode)->getCache()->getReplicaInfo(dataAccess.getAccessRange().getStartAddress());
                            //! homeNode also failed, nothing to do. 
                            assert(sourceReplica->_physicalAddress != nullptr && "Cannot copy data from homeNode");
                        }
                        //! TODO: memcpy should be wrapped!!
                        memcpy(replicaAddress, sourceReplica->_physicalAddress, dataAccess.getAccessRange().getSize());
                        newReplica._dirty = newReplica._dirty || sourceReplica->_dirty;
                        newReplica._version = sourceReplica->_version + newReplica._dirty;
                    }
                    //! Insert replica into _replicas.
                    _replicas[dataAccess.getAccessRange().getStartAddress()] = newReplica;
                }
                else {
                    //! If this cache is the homeNode, we must check if homeNode's data is up to date.
                    //! If so, there is no problem. Otherwise, we must check the access type. If the 
                    //! access type is only write, even with outdated data there is no problem because 
                    //! it is going to be overwritten. However, if access includes read, we must bring 
                    //! data.
                    if(!Directory::isHomeNodeUpToDate(dataAccess.getAccessRange().getStartAddress()) && 
                        (dataAccess._type != WRITE_ACCESS_TYPE)) {
                        //! Bring data. It implies asking directory where is it and requesting that cache 
                        //! a writeBack.
                        cache_mask caches = Directory::getCaches(dataAccess.getAccessRange().getStartAddress());
                        int cacheIndex = -1;
                        for(int i=0; i<caches.size(); ++i) {
                            if(caches.test(i)) {
                                cacheIndex = i;
                                break;
                            }
                        }
                        Machine::getMemoryNode(cacheIndex)->getCache()->writeBack(dataAccess.getAccessRange().getStartAddress());
                        assert(Directory::isHomeNodeUpToDate(dataAccess.getAccessRange().getStartAddress()));
                    }
                }
                //! All cases needs insert into directory, do it.
                int dirVersion = Directory::insertCopy(/*address*/ dataAccess.getAccessRange().getStartAddress(), 
                                                       /*size*/ dataAccess.getAccessRange().getSize(),
                                                       /*homeNode*/ dataAccess._homeNode,
                                                       /*cache index*/ _index, 
                                                       /*increment*/ (dataAccess._type != READ_ACCESS_TYPE));
                if(newReplica._physicalAddress != nullptr)
                    assert(dirVersion==newReplica._version && "Versions must match");
            }
            ++copiesDone;
            //! Update task cachedBytes.
            task->addCachedBytes(dataAccess.getAccessRange().getSize());
            //! Mark dataAccess as cached.
            dataAccess.setCached(true);
        }
    }

    //! Reenqueue task either to make the remaining copies or to be executed.
    addReadyTask(task);
}

void NUMACache::flush() {
    //! Iterate over all the replicas and copy them back to the homeNode if the replica is dirty and it is the last version and 
    //! nobody else has the data. Write back implies notify directory the removal of this cache and the update in the homeNode.
    //! For all replicas: invalidate (version=-1), set refCount to 0, set dirty to false, set lastUse to 0.
    //! There is no need to actually clear the cache because if after a flush comes again the same dataAccess, at least we don't need 
    //! to reallocate the space even though we do need to copy the data again.
	std::lock_guard<SpinLock> guard(_lock);
    for(auto& replica : _replicas) {
        //! The replica is dirty and it is the last version, and nobody else has the data: copy back to homeNode, notifiy removal 
        //! to directory and notify update in the homeNode to directory.
        if(replica.second._dirty && replica.second._version == Directory::getVersion(replica.first) && 
           ((Directory::getCaches(replica.first).to_ulong() == (unsigned long)(1<<_index)) && !Directory::isHomeNodeUpToDate(replica.first))) {
            writeBack(replica.first);
        }
        replica.second._version = -1;
        replica.second._dirty = false;
        replica.second._refCount.store(0);
        replica.second._lastUse = 0;
    }
}

bool NUMACache::evict() {
    std::set< std::pair<long unsigned int, void *> > candidates;
    //! Look for candidates to be evicted.
    for(auto& replica : _replicas) {
        //! Iterate through all the replicas to look for those that are not being used now.
        if(replica.second._refCount.load() == 0) {
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
        //! The replica is dirty and it is the last version, and nobody else has the data: copy back to homeNode, notifiy removal 
        //! to directory and notify update in the homeNode to directory.
        if(replica->_dirty && replica->_version == Directory::getVersion(candidate->second) && 
           ((Directory::getCaches(candidate->second).to_ulong() == (unsigned long)(1<<_index)) && !Directory::isHomeNodeUpToDate(candidate->second))) {
            writeBack(candidate->second);
        }
        //! Free space of the replica
        deallocate(replica->_physicalAddress);
        //! Remove replica from _replicas
        _replicas.erase(candidate->second);

        return true;
    }
}

void NUMACache::writeBack(void* address) {
    replicaInfo_t * replica = &_replicas[address];
    assert(replica != nullptr && "Replica must be present");
    assert(replica->_version == Directory::getVersion(address) && "Replica must be updated");
    //! First, copy back to homeNode.
    memcpy(address, replica->_physicalAddress, replica->_size); 
    //! Notify directory the data is not in the cache anymore.
    Directory::eraseCopy(address, _index);
    //! Notify directory the data is now up to date in the homeNode.
    Directory::setHomeNodeUpToDate(address, true);
}

#endif //NUMA_CACHE_CPP
