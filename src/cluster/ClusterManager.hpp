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
#include <MessageDataFetch.hpp>
#include <MessageDataSend.hpp>

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
	
	//! A vector of all ClusterNodes in the system.
	//!
	//! We might need to make this a map later on, when we start
	//! adding/removing nodes
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
	
	//! \brief Get a vector containing all ClusterNode objects
	//!
	//! \returns A vector containing all ClusterNode objects
	static inline std::vector<ClusterNode *> const &getClusterNodes()
	{
		return _clusterNodes;
	}
	
	//! \brief Get the ClusterNode representing the master node
	//!
	//! \returns the master node ClusterNode
	static inline ClusterNode *getMasterNode()
	{
		return _masterNode;
	}
	
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

	//! \brief Check for incoming messages
	//!
	//! This is just a wrapper on top of the Messenger API
	//!
	//! \returns a Message object if one has been received otherwise,
	//!		nullptr
	static inline Message *checkMail()
	{
		assert(_msn != nullptr);
		return _msn->checkMail();
	}
	
	//! \brief Send a Message to a remote Node
	//!
	//! This is just a wrapper on top of the Messenger API
	//!
	//! \param[in] msg is the Message to send
	//! \param[in] recipient is the remote node to send the Message
	//! \param[in] if block is true the the call will block until the
	//!		Message is sent
	static inline void sendMessage(Message *msg,
			ClusterNode const *recipient, bool block = false)
	{
		assert(_msn != nullptr);
		assert(msg != nullptr);
		assert(recipient != nullptr);
		_msn->sendMessage(msg, recipient, block);
	}

	//! \brief Test Messages for completion
	//!
	//! This is just a wrapper on top of the Messenger API
	//!
	//! \param[in] messages is a vector containing Message objects
	//!		to check for completion
	static inline void testMessageCompletion(
		std::vector<Message *> &messages
	) {
		assert(_msn != nullptr);
		_msn->testMessageCompletion(messages);
	}
	
	//! \brief Test DataTransfers for completion
	//!
	//! This is just a wrapper on top of the Messenger API
	//!
	//! \param[in] transfers is a vector containing DataTransfer objects
	//!		to check for completion
	static inline void testDataTransferCompletion(
		std::vector<DataTransfer *> &transfers
	) {
		assert(_msn != nullptr);
		_msn->testDataTransferCompletion(transfers);
	}
	
	//! \brief Fetch a DataAccessRegion from a remote node
	//!
	//! \param[in] region is the address region to fetch
	//! \param[in] from is the remote MemoryPlace we are fetching from
	//! \param[in] messageId is the index of the Message with which this
	//!		DataTransfer is related
	//! \param[in] if block is true the call will block until the data is
	//!		received
	//!
	//! \returns a DataTransfer object if data was received non-blocking,
	//!		otherwise nullptr
	static inline DataTransfer *fetchDataRaw(
		DataAccessRegion const &region,
		MemoryPlace const *from,
		int messageId,
		bool block = false
	) {
		assert(_msn != nullptr);
		assert(from != nullptr);
		
		ClusterNode const *remoteNode = getClusterNode(from->getIndex());
		return _msn->fetchData(region, remoteNode, messageId, block);
	}
	
	//! \brief Send a DataAccessRegion to a remote node
	//!
	//! \param[in] region is the address region to send
	//! \param[in] to is the remote MemoryPlace we are sending to
	//! \param[in] messageId is the index of the Message with which this
	//!		DataTransfer is related
	//! \param[in] if block is true the call will block until the data is
	//!		sent
	//!
	//! \returns a DataTransfer object if data was sent non-blocking,
	//!		otherwise nullptr
	static inline DataTransfer *sendDataRaw(
		DataAccessRegion const &region,
		MemoryPlace const *to,
		int messageId,
		bool block = false
	) {
		assert(_msn != nullptr);
		assert(to != nullptr);
		
		ClusterNode const *remoteNode = getClusterNode(to->getIndex());
		return _msn->sendData(region, remoteNode, messageId, block);
	}
	
	//! \brief Initiate a data fetch operation
	//!
	//! \param[in] region is the local region we want to update with data
	//!		from the remote node
	//! \param[in] from is the MemoryPlace we fetch the data from. This
	//!		must be a cluster memory place
	//! \param[in] block determines whether the operation will be blocking.
	//!		If block is true then upon return, the data will have
	//!		been succesfully fetched and region will be updated.
	//!
	//! \returns In non-blocking mode, this method returns a DataTransfer
	//!		object which can be used to track the completion of the
	//!		data transfer. In blocking mode this always returns
	//!		nullptr
	static inline DataTransfer *fetchData(DataAccessRegion const &region,
		MemoryPlace const *from, bool block = false)
	{
		assert(_msn != nullptr);
		assert(from != nullptr);
		
		ClusterNode const *remoteNode =
			getClusterNode(from->getIndex());
		
		//! At the moment we do not translate addresses on remote
		//! nodes, so the region we are fetching, on the remote node is
		//! the same as the local one
		MessageDataFetch msg(_thisNode, region);
		_msn->sendMessage(&msg, remoteNode, true);
		
		return fetchDataRaw(region, from, msg.getId(), block);
	}
	
	//! \brief Initiate a data send operation
	//!
	//! \param[in] region is the local region we send to the remote node
	//! \param[in] to is the MemoryPlace we send the data to. This must be a
	//!		cluster memory place
	//! \param[in] block determines whether the operation will be blocking.
	//!		If block is true then upon return, the data will have
	//!		been succesfully sent and region is allowed to be
	//!		modified.
	//!
	//! \returns In non-blocking mode, this method returns a DataTransfer
	//!		object which can be used to track the completion of the
	//!		data transfer. In blocking mode this always returns
	//!		nullptr
	static inline DataTransfer *sendData(DataAccessRegion const &region,
		MemoryPlace const *to, bool block = false)
	{
		assert(_msn != nullptr);
		assert(to != nullptr);
		
		ClusterNode const *remoteNode =
			getClusterNode(to->getIndex());
		
		//! At the moment we do not translate addresses on remote
		//! nodes, so the region we are sending, on the remote node is
		//! the same as the local one
		MessageDataSend msg(_thisNode, region);
		_msn->sendMessage(&msg, remoteNode, true);
		
		return sendDataRaw(region, to, msg.getId(), block);
	}
	
	//! \brief A barrier across all cluster nodes
	//!
	//! This is a collective operation. It needs to be invoked by all
	//! cluster nodes, otherwise a deadlock will occur. Execution of the
	//! cluster node will be blocked until all nodes reach at the matching
	//! synchronization point.
	static inline void synchronizeAll()
	{
		if (inClusterMode()) {
			assert(_msn != nullptr);
			_msn->synchronizeAll();
		}
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
