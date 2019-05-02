/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef CLUSTER_TASK_CONTEXT_HPP
#define CLUSTER_TASK_CONTEXT_HPP

class ClusterNode;

namespace TaskOffloading {
	
	class ClusterTaskContext {
	public:
		ClusterTaskContext(void *, ClusterNode *)
		{
		}
		
		inline void *getRemoteDescriptor()
		{
			return nullptr;
		}
		
		inline ClusterNode *getRemoteNode()
		{
			return nullptr;
		}
	};
}

#endif /* CLUSTER_TASK_CONTEXT_HPP */
