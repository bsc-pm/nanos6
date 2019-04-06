/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CLUSTER_MANAGER_HPP
#define CLUSTER_MANAGER_HPP

#include <atomic>
#include <cassert>
#include <string>
#include <vector>

#include "cluster/messenger/Messenger.hpp"

#include <ClusterNode.hpp>

class ClusterMemoryNode;

class ClusterManager {
public:
	//! ShutdownCallback function to call during shutdown in the cases where
	//! the runtime does not run the main function
	class ShutdownCallback {
		void (*_function)(void *);
		void *_args;
	public:
		ShutdownCallback(void (*func)(void *), void *args) :
			_function(func), _args(args)
		{
		}
		
		inline void invoke()
		{
			assert(_function != nullptr);
			_function(_args);
		}
	};
	
private:
	//! Number of cluster nodes
	static int _clusterSize;
	
	/** A vector of all ClusterNodes in the system.
	 *
	 * We might need to make this a map later on, when we start
	 * adding/removing nodes
	 */
	static std::vector<ClusterNode *> _clusterNodes;
	
	//! ClusterNode object of the current node
	static ClusterNode *_thisNode;
	
	//! ClusterNode of the master node
	static ClusterNode *_masterNode;
	
	//! Messenger object for cluster communication.
	static Messenger *_msn;
	
	//! The ShutdownCallback for this ClusterNode.
	//! At the moment this is an atomic variable, because we might have
	//! to poll for this, until it's set from external code. For example,
	//! this could happen if a remote node tries to shutdown (because
	//! we received a MessageSysFinish before the loader setting the
	//! callback.
	static std::atomic<ShutdownCallback *> _callback;
	
	//! Internal helper function to initialize cluster support with a
	//! particular communicator type
	static void initializeCluster(std::string const &commType);
	
	//! private constructor. This is a singleton.
	ClusterManager()
	{}
public:
	//! \brief Initialize the ClusterManager
	static void initialize();
	
	//! \brief Shutdown the ClusterManager
	static void shutdown();
	
	//! \brief Get the ClusterNode with index 'id'
	//!
	//! \param[in] id is the index of the ClusterNode we request
	//!
	//! \returns The ClusterNode object with index 'id'
	static inline ClusterNode *getClusterNode(int id)
	{
		return _clusterNodes[id];
	}
	
	//! \brief Get the current ClusterNode
	//!
	//! \returns the ClusterNode object of the current node
	static inline ClusterNode *getCurrentClusterNode()
	{
		return _thisNode;
	}
	
	//! \brief Get The ClusterMemoryNode with index id;
	//!
	//! \param[in] id is the index of the ClusterMemoryNode we request
	//!
	//! \returns The ClusterMemoryNode object with index 'id'
	static inline ClusterMemoryNode *getMemoryNode(int id)
	{
		assert(_clusterNodes[id] != nullptr);
		return _clusterNodes[id]->getMemoryNode();
	}
	
	//! \brief Get the current ClusterMemoryNode
	//!
	//! \returns the ClusterMemoryNode object of the current node
	static inline ClusterMemoryNode *getCurrentMemoryNode()
	{
		assert(_thisNode != nullptr);
		return _thisNode->getMemoryNode();
	}
	
	//! \brief Check if current node is the master
	//!
	//! \returns true if the current node is the master
	static inline bool isMasterNode()
	{
		return _masterNode == _thisNode;
	}
	
	//! \brief Get the number of cluster nodes
	//!
	//! \returns the number of cluster nodes
	static inline int clusterSize()
	{
		return _clusterSize;
	}
	
	//! \brief Check if we run in cluster mode
	//!
	//! We run in cluster mode, if we have compiled with cluster support,
	//! we have enabled Cluster at runtime and we run with more than one
	//! Cluster nodes.
	//!
	//! \returns true if we run in cluster mode
	static inline bool inClusterMode()
	{
		return _clusterSize > 1;
	}

	//! \brief Test Messages for completion
	//!
	//! This is just a wrapper on top of the Messenger API
	//!
	//! \param[in] messages is a deque containing Message objects
	//!		to check for completion
	//! \param[in] completed is a deque which, upon return will
	//!		contain completed messages
	static inline void testMessageCompletion(
		std::deque<Message *> &messages,
		std::deque<Message *> &completed
	) {
		_msn->testMessageCompletion(messages, completed);
	}
	
	//! \brief A barrier across all cluster nodes
	//!
	//! This is a collective operation. It needs to be invoked by all
	//! cluster nodes, otherwise a deadlock will occur. Execution of the
	//! cluster node will be blocked until all nodes reach at the matching
	//! synchronization point.
	static inline void synchronizeAll()
	{
		assert(_msn != nullptr);
		_msn->synchronizeAll();
	}
	
	//! \brief Set a callback function to invoke when we have to shutdown
	//!
	//! The callback is of the form 'void callback(void*)' and it will be
	//! invoked when we have to shutdown the runtime instance
	//!
	//! \param[in] func is the callback function
	//! \param[in] args is the callback function argument
	static inline void setShutdownCallback(void (*func)(void *), void *args)
	{
		_callback.store(new ShutdownCallback(func, args));
	}
	
	//! \brief Get the shutdown callback
	//!
	//! \returns the ShutdownCallback
	static inline ShutdownCallback *getShutdownCallback()
	{
		return _callback.load();
	}
};

#endif /* CLUSTER_MANAGER_HPP */
