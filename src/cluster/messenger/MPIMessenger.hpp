/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef __MPI_MESSENGER_H__
#define __MPI_MESSENGER_H__

#include <vector>

#pragma GCC visibility push(default)
#include <mpi.h>
#pragma GCC visibility pop

#include "Messenger.hpp"

class Message;
class ClusterPlace;

class MPIMessenger : public Messenger {
private:
	int wrank, wsize;
	MPI_Comm INTRA_COMM, PARENT_COMM;
	
public:
	MPIMessenger();
	~MPIMessenger();
	
	void sendMessage(Message *msg, ClusterNode *toNode);
	void sendMessage(Message *msg, std::vector<ClusterNode *> const &toNodes);
	void synchronizeAll(void);
	void sendData(const DataAccessRegion &region, const ClusterNode *toNode);
	void fetchData(const DataAccessRegion &region, const ClusterNode *fromNode);
	Message *checkMail();
	
	inline int getNodeIndex() const
	{
		return wrank;
	}
	
	inline int getMasterIndex() const
	{
		return 0;
	}
	
	inline int getClusterSize() const
	{
		return wsize;
	}
	
	inline bool isMasterNode() const
	{
		return wrank == 0;
	}
};

//! Register MPIMessenger with the object factory
namespace
{
	Messenger *createMPImsn() { return new MPIMessenger; }
	
	const bool __attribute__((unused))_registered_MPI_msn =
		REGISTER_MSN_CLASS("mpi-2sided", createMPImsn);
}

#endif /* __MPI_MESSENGER_H__ */
