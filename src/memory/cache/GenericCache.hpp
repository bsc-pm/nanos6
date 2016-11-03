#ifndef GENERIC_CACHE_HPP
#define GENERIC_CACHE_HPP

#include <unordered_map>
#include <string.h>
#include <atomic>
#include "../../tasks/Task.hpp"
#include "scheduling/Scheduler.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/CPU.hpp"
//#include <boost/pool/pool_alloc.hpp>
//#include "CacheObject.hpp"

class GenericCache {
public: 
    struct replicaInfo_t {
        //! Address in the cache address space.
        void * _physicalAddress;
        //! Size of the replica
        std::size_t _size;
        //! Version of the replica.
        int _version;
        //! in or out
        bool _dirty;
        //! Is there someone using the replica?
        std::atomic_uint _refCount;
        //! Last use to determine evictions
        long unsigned int _lastUse;

        replicaInfo_t& operator=(replicaInfo_t& other) {
            this->_physicalAddress = other._physicalAddress;
            this->_size = other._size;
            this->_version = other._version;
            this->_dirty = other._dirty;
            this->_refCount.store(other._refCount);
            this->_lastUse = other._lastUse;
        }
    };
protected:
    typedef std::unordered_map<void *, replicaInfo_t> replicas_t;
    //! List of replicas. The key is the logical address in the OmpSs address space (i.e. the address of the user space). The value is the physical address 
    //! where the replica is actually stored in the cache, the version of the replica, a dirty byte and a reference counter.
    replicas_t _replicas;
    //! Pool allocator
    //boost::pool_allocator<char> _pool;
    //! Counter to determine the last use of the replicas. Not thread-protected because it is not critical to have two replicas with the same lastUse.
    long unsigned int _count; 
    int _index;
    void addReadyTask(Task *task) {
        WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
        ComputePlace *hardwarePlace = currentThread->getComputePlace();
        ComputePlace *idleComputePlace = Scheduler::addReadyTask(task, hardwarePlace, SchedulerInterface::SchedulerInterface::NO_HINT);
        assert((currentThread != nullptr) || (idleComputePlace == nullptr)); // The main task is added before the scheduler

        if (idleComputePlace != nullptr) {
            ThreadManager::resumeIdle((CPU *) idleComputePlace);
        }
    }
public:
    GenericCache(int index) : _count(0), _index(index) {}
    virtual ~GenericCache() {}
    virtual void * allocate(std::size_t size) = 0;
    virtual void deallocate(void * ptr) = 0;
    virtual void copyData(unsigned int sourceCache, unsigned int homeNode, Task * task) = 0;
    virtual void flush() = 0;
    virtual bool evict() = 0;
    virtual replicaInfo_t * getReplicaInfo(void * key) {
        replicaInfo_t *res = new replicaInfo_t;
        auto it = _replicas.find(key);
        if(it == _replicas.end()) {
            res->_physicalAddress = NULL;
            res->_version = -1;
            res->_dirty = false;
            res->_refCount = 0;
        }
        else
            *res = (_replicas.find(key))->second;
        return res;
    }
};

#endif //GENERIC_CACHE_HPP
