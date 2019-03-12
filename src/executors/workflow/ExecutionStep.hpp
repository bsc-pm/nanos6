#ifndef __EXECUTION_STEP_HPP__
#define __EXECUTION_STEP_HPP__

#include "dependencies/DataAccessType.hpp"
#include "lowlevel/SpinLock.hpp"

#include <mutex>
#include <vector>
#include <DataAccessRegion.hpp>

class DataAccess;
class MemoryPlace;

namespace ExecutionWorkflow {
	
	struct RegionTranslation {
		DataAccessRegion _hostRegion;
		void *_deviceStartAddress;
		
		RegionTranslation(DataAccessRegion hostRegion, void *deviceStartAddress)
			: _hostRegion(hostRegion), _deviceStartAddress(deviceStartAddress)
		{
		}
		
		RegionTranslation()
			: _hostRegion(), _deviceStartAddress(nullptr)
		{
		}
	};
	
	// NOTE: objects of this class self-destruct when they finish
	class Step {
	protected:
		//! pending previous steps
		int _countdownToRelease;
		
		//! list of successor Steps
		std::vector<Step *> _successors;
		
		//! lock to protect access to _successors
		//! TODO: maybe it makes sense to create an AtomicVector type
		SpinLock _lock;
		
	public:
		Step()
			: _countdownToRelease(0), _successors(), _lock()
		{
		}
		
		virtual ~Step()
		{
		}
		
		//! \brief add one pending predecessor to the Step
		inline void addPredecessor()
		{
			std::lock_guard<SpinLock> guard(_lock);
			++_countdownToRelease;
		}
		
		//! \brief Add a successor Step
		inline void addSuccessor(Step *step)
		{
			std::lock_guard<SpinLock> guard(_lock);
			_successors.push_back(step);
		}
		
		//! \brief Decrease the number of predecessors and start the Step execution
		//!        if the Step became ready.
		//!
		//! \returns true if the next Step is ready to start.
		inline bool release()
		{
			std::lock_guard<SpinLock> guard(_lock);
			--_countdownToRelease;
			assert(_countdownToRelease >= 0);
			
			if (_countdownToRelease == 0) {
				return true;
			}
			
			return false;
		}
		
		//! \brief Release successor steps
		inline void releaseSuccessors()
		{
			/* Commenting out the following lock, because it is
			 * actually not needed and it might lead to a deadlock.
			 * At this point, the Workflow is created and the
			 * _successors vector will not be further modified. */
			//std::lock_guard<SpinLock> guard(_lock);
			for (auto step: _successors) {
				if (step->release()) {
					step->start();
				}
			}
		}
		
		//! \brief Returns true if Step is ready to run
		inline bool ready() const
		{
			return (_countdownToRelease == 0);
		}
		
		//! \brief start the execution of a Step
		virtual void start()
		{
			releaseSuccessors();
			delete this;
		}
	};
	
	class DataLinkStep : public Step {
	protected:
		DataAccess const *_access;
		
		//! The number of bytes that this Step has to link
		std::atomic<size_t> _bytes_to_link;
		
	public:
		DataLinkStep(DataAccess const *access);
		
		virtual void linkRegion(
			DataAccessRegion const &region,
			MemoryPlace *location,
			bool read,
			bool write
		) = 0;
		
		virtual void start() = 0;
	};
	
	class DataReleaseStep : public Step {
	protected:
		//! The access for which this steps handles releases
		DataAccess const *_access;
		
		//! The number of bytes that this Step has to release
		std::atomic<size_t> _bytes_to_release;
		
	public:
		DataReleaseStep(DataAccess const *access);
		
		//! Release a region
		virtual inline void releaseRegion(
			__attribute__((unused))DataAccessRegion const &region,
			__attribute__((unused))DataAccessType type,
			__attribute__((unused))bool weak,
			__attribute__((unused))MemoryPlace *location
		) {
		}
		
		virtual void start();
	};
};

#endif /* __EXECUTION_STEP_HPP__ */
