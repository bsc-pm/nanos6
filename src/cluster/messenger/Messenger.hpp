/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef __MESSENGER_HPP__
#define __MESSENGER_HPP__

#include <vector>
#include <string>
#include <DataAccessRegion.hpp>
#include <support/GenericFactory.hpp>

class Message;
class ClusterNode;

class Messenger {
public:
	Messenger()
	{
	}
	
	virtual ~Messenger()
	{
	}
	
	/** Send a message to a remote node.
	 *
	 * \param msg is the Message to send
	 * \param toNode is the receiver node
	 */
	virtual void sendMessage(Message *msg, ClusterNode *toNode) = 0;
	
	/** Send a message to multiple remote nodes.
	 *
	 * \param msg is the Message to send
	 * \param toNodes is a vector of nodes to send the message to
	 */
	virtual void sendMessage(Message *msg, std::vector<ClusterNode *> const &toNodes) = 0;
	
	//! A barrier across all nodes
	virtual void synchronizeAll(void) = 0;
	
	/** Send a data region to a remote node, related to a previous message.
	 *
	 * \param region is the data region to send
	 * \param toNode is the receiver node
	 */
	virtual void sendData(const DataAccessRegion &region, const ClusterNode *toNode) = 0;
	
	/** Receive a data region from a remote node, related to a previous message
	 *
	 * \param region is the data region to fetch
	 * \param fromNode is the node to fetch the data from
	 * 	  with this data transfer
	 */
	virtual void fetchData(const DataAccessRegion &region, const ClusterNode *fromNode) = 0;
	
	/** Check for incoming messages
	 *
	 * Invoke the messenger to check from incoming messages
	 *
	 * \return A pointer to a message or nullptr if none has been received
	 */
	virtual Message *checkMail() = 0;
	
	//! Get the index of the current node
	virtual int getNodeIndex() const = 0;
	
	//! Get the index of the master node
	virtual int getMasterIndex() const = 0;
	
	//! Get the size of the Cluster
	virtual int getClusterSize() const = 0;
	
	//! Returns true if this is the master node
	virtual bool isMasterNode() const = 0;
};

#define REGISTER_MSN_CLASS(NAME, CREATEFN) \
	GenericFactory<std::string, Messenger*>::getInstance().emplace(NAME, CREATEFN)

#endif /* __MESSENGER_HPP__ */
