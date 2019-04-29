#ifndef __EXECUTION_STEP_HPP__
#define __EXECUTION_STEP_HPP__

#include <DataAccessRegion.hpp>
#include "dependencies/DataAccessType.hpp"

class DataAccess;
class MemoryPlace;

namespace ExecutionWorkflow {
	
	struct RegionTranslation {
		RegionTranslation(
			__attribute__((unused))DataAccessRegion hostRegion,
			__attribute__((unused))void *_deviceStartAddress
		) {
		}
		
		RegionTranslation()
		{
		}
	};
	
	class Step {
	public:
		Step()
		{
		}
		
		~Step()
		{
		}
		
		inline void addPredecessor()
		{
		}

		inline void addSuccessor(
			__attribute__((unused))Step *step
		) {
		}
		
		inline bool release()
		{
			return true;
		}
		
		inline void releaseSuccessors()
		{
		}
		
		inline bool ready() const
		{
			return true;
		}
		
		inline void start()
		{
		}
	};
	
	class DataLinkStep : public Step {
	public:
		DataLinkStep(
			__attribute__((unused))DataAccess const *access
		) : Step() 
		{
		}
		
		inline void linkRegion(
			__attribute__((unused))DataAccessRegion const &region,
			__attribute__((unused))MemoryPlace *location,
			__attribute__((unused))bool read,
			__attribute__((unused))bool write
		) {
		}
	};
	
	class DataReleaseStep : public Step {
	public:	
		DataReleaseStep(
			__attribute__((unused))DataAccess const *access
		) : Step()
		{
		}
		
		inline void releaseRegion(
			__attribute__((unused))DataAccessRegion const &region,
			__attribute__((unused))MemoryPlace *location
		) {
		}
	};
}

#endif /* __EXECUTION_STEP_HPP__ */
