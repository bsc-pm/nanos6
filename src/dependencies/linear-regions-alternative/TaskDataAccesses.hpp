#ifndef TASK_DATA_ACCESSES_HPP
#define TASK_DATA_ACCESSES_HPP

#include <atomic>
#include <bitset>
#include <cassert>
#include <mutex>

#include "BottomMapEntry.hpp"
#include "IntrusiveLinearRegionMap.hpp"
#include "IntrusiveLinearRegionMapImplementation.hpp"
#include "TaskDataAccessLinkingArtifacts.hpp"
#include "lowlevel/SpinLock.hpp"


struct DataAccess;
class Task;


struct TaskDataAccesses {
	typedef SpinLock spinlock_t;
	
	typedef IntrusiveLinearRegionMap<
		DataAccess,
		boost::intrusive::function_hook< TaskDataAccessLinkingArtifacts >
	> accesses_t;
	typedef IntrusiveLinearRegionMap<
		BottomMapEntry,
		boost::intrusive::function_hook< BottomMapEntryLinkingArtifacts >
	> subaccess_bottom_map_t;
	
#ifndef NDEBUG
	enum flag_bits {
		HAS_BEEN_DELETED_BIT=0,
		TOTAL_FLAG_BITS
	};
	typedef std::bitset<TOTAL_FLAG_BITS> flags_t;
#endif
	
	spinlock_t _lock;
	accesses_t _accesses;
	subaccess_bottom_map_t _subaccessBottomMap;
	
	int _removalCountdown;
#ifndef NDEBUG
	flags_t _flags;
#endif
	
	TaskDataAccesses()
		: _lock(),
		_accesses(),
		_subaccessBottomMap(),
		_removalCountdown(0)
#ifndef NDEBUG
		,_flags()
#endif
	{
	}
	
	~TaskDataAccesses();
	
	TaskDataAccesses(TaskDataAccesses const &other) = delete;
	
#ifndef NDEBUG
	bool hasBeenDeleted() const
	{
		return _flags[HAS_BEEN_DELETED_BIT];
	}
	flags_t::reference hasBeenDeleted()
	{
		return _flags[HAS_BEEN_DELETED_BIT];
	}
#endif
	
	bool isRemovable()
	{
		return (_removalCountdown == 0);
	}
	
	void increaseRemovalCount(int amount = 1)
	{
		_removalCountdown += amount;
	}
	
	bool decreaseRemovalCount(int amount = 1)
	{
		int countdown = (_removalCountdown -= amount);
		assert(countdown >= 0);
		
		return (countdown == 0);
	}
};


typedef typename TaskDataAccessLinkingArtifacts::hook_type TaskDataAccessesHook;


struct TaskDataAccessHooks {
	TaskDataAccessesHook _accessesHook;
};


#endif // TASK_DATA_ACCESSES_HPP
