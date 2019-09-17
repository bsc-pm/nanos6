/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "DataAccessRegion.hpp"

#ifndef REDUCTION_INFO_HPP
#define REDUCTION_INFO_HPP

/*
 *  This is a placeholder file, needed for the nanos6 verbose instrumentation to compile
 */

class ReductionInfo {
	private:
	DataAccessRegion _region;

	public:

	const DataAccessRegion& getOriginalRegion() const {
		return _region;
	}
};


#endif // REDUCTION_INFO_HPP
