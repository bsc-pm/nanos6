/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#include "ClusterManager.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "messages/MessageSysFinish.hpp"
#include "messenger/Messenger.hpp"
#include "polling-services/ClusterPollingServices.hpp"
#include "system/RuntimeInfo.hpp"

#include <ClusterNode.hpp>

int ClusterManager::_clusterSize;
std::vector<ClusterNode *> ClusterManager::_clusterNodes;
ClusterNode *ClusterManager::_thisNode = nullptr;
ClusterNode *ClusterManager::_masterNode = nullptr;
Messenger *ClusterManager::_msn = nullptr;
std::atomic<ClusterManager::ShutdownCallback *> ClusterManager::_callback;


void ClusterManager::initializeCluster(std::string const &commType)
{
	_msn = GenericFactory<std::string, Messenger*>::getInstance().create(commType);
	assert(_msn);
	
	/** These are communicator-type indices. At the moment we have an
	 * one-to-one mapping between communicator-type and runtime-type
	 * indices for cluster nodes */
	_clusterSize = _msn->getClusterSize();
	int nodeIndex = _msn->getNodeIndex();
	int masterIndex = _msn->getMasterIndex();
	
	_clusterNodes.resize(_clusterSize);
	for (int i = 0; i < _clusterSize; ++i) {
		_clusterNodes[i] = new ClusterNode(i, i);
	}
	
	_thisNode = _clusterNodes[nodeIndex];
	_masterNode = _clusterNodes[masterIndex];
	
	if (inClusterMode()) {
		ClusterPollingServices::initialize();
	}
	
	_msn->synchronizeAll();
	_callback.store(nullptr);
}

void ClusterManager::initialize()
{
	EnvironmentVariable<std::string> commType("NANOS6_COMMUNICATION", "disabled");
	RuntimeInfo::addEntry("cluster_communication", "Cluster Communication Implementation", commType);
	
	/** If a communicator has not been specified through the
	 * NANOS6_COMMUNCIATION environment variable we will not
	 * initialize the cluster support of Nanos6 */
	if (commType.getValue() != "disabled") {
		initializeCluster(commType.getValue());
		return;
	}
	
	_thisNode = new ClusterNode(0, 0);
	_masterNode = _thisNode;
	_clusterNodes.emplace_back(_thisNode);
	_clusterSize = 1;
}

void ClusterManager::shutdown()
{
	if (isMasterNode() && inClusterMode()) {
		for (ClusterNode *slaveNode : _clusterNodes) {
			if (slaveNode != _thisNode) {
				MessageSysFinish msg(_thisNode);
				_msn->sendMessage(&msg, slaveNode, true);
			}
		}
		
		_msn->synchronizeAll();
	}
	
	for (auto &node : _clusterNodes) {
		delete node;
	}
	
	if (inClusterMode()) {
		ClusterPollingServices::shutdown();
	}
	
	delete _msn;
	
	_thisNode = nullptr;
	_masterNode = nullptr;
}
