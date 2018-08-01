#include "MPIMessenger.hpp"
#include "cluster/messages/Message.hpp"
#include <ClusterNode.hpp>

#include "alloca.h"
#include <cstdlib>
#include <vector>

#pragma GCC visibility push(default)
#include <mpi.h>
#pragma GCC visibility pop

MPIMessenger::MPIMessenger()
{
	int support;
	
	MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &support);
	if (support != MPI_THREAD_MULTIPLE) {
		std::cerr << "Could not initialize multithreaded MPI" << std::endl;
		abort();
	}
	
	//! make sure that MPI errors are returned in the COMM_WORLD
	MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
	
	//! Save the parent communicator
	MPI_Comm_get_parent(&PARENT_COMM);
	
	//! Create a new communicator
	MPI_Comm_dup(MPI_COMM_WORLD, &INTRA_COMM);
	
	//! make sure the new communicator returns errors
	MPI_Comm_set_errhandler(INTRA_COMM, MPI_ERRORS_RETURN);
	MPI_Comm_rank(INTRA_COMM, &wrank);
	MPI_Comm_size(INTRA_COMM, &wsize);
}

MPIMessenger::~MPIMessenger()
{
	//! Release the intra-communicator
	MPI_Comm_free(&INTRA_COMM);
	MPI_Finalize();
}

void MPIMessenger::sendMessage(Message *msg, ClusterNode *toNode)
{
	int ret;
	Message::Deliverable *delv = msg->getDeliverable();
	const int MPI_to = toNode->getCommIndex();
	
	assert(MPI_to < wsize && MPI_to != wrank);
	assert(delv->header.size != 0);
	
	/*! At the moment we use the Message type as the MPI
	 * tag of the communication */
	int tag = delv->header.type;
	ret = MPI_Send((void *)delv, sizeof(delv->header) + delv->header.size,
			MPI_BYTE, MPI_to, tag, INTRA_COMM);
	if (ret != MPI_SUCCESS) {
		MPI_Abort(INTRA_COMM, ret);
	}
}

void MPIMessenger::sendMessage(Message *msg, std::vector<ClusterNode *> const &toNodes)
{
	Message::Deliverable *delv = msg->getDeliverable();
	assert(delv->header.size != 0);
	
	const int nr_nodes = toNodes.size();
	assert(nr_nodes != 0);
	
	/*! At the moment we use the Message type as the MPI
	 * tag of the communication */
	const int tag = delv->header.type;
	
	/*! nr_nodes should be in the order of 1000s, so alloca should
	 * probably be the best option now */
	MPI_Request *requests = (MPI_Request *) alloca(nr_nodes * sizeof(MPI_Request));
	MPI_Status *status = (MPI_Status *) alloca(nr_nodes * sizeof(MPI_Status));
	
	int i = 0;
	for (auto &node : toNodes) {
		int MPI_to = node->getCommIndex();
		assert(MPI_to != wrank);
		
		int ret = MPI_Isend((void *)delv,
				sizeof(delv->header) + delv->header.size,
				MPI_BYTE, MPI_to, tag, INTRA_COMM,
				&requests[i]);
		if (ret != MPI_SUCCESS) {
			MPI_Abort(INTRA_COMM, ret);
		}
		
		++i;
	}
	
	//! Check that all sends were fine or print some information;
	int ret = MPI_Waitall(nr_nodes, requests, status);
	if(ret == MPI_ERR_IN_STATUS) {
		for(i = 0; i < nr_nodes; ++i){
			if(status[i].MPI_ERROR != MPI_SUCCESS){
				std::cerr << "MPI_Error "
					<< status[i].MPI_ERROR
					<< " sending collective message "
					<< status[i].MPI_TAG
					<< " to rank:"
					<< status[i].MPI_SOURCE
					<< std::endl;
			}
		}
		
		MPI_Abort(INTRA_COMM, ret);
	}
}

void MPIMessenger::sendData(const DataAccessRegion &region, const ClusterNode *to)
{
	int ret;
	const int MPI_to = to->getCommIndex();
	void *address = region.getStartAddress();
	size_t size = region.getSize();
	
	assert(MPI_to < wsize && MPI_to != wrank);
	
	ret = MPI_Send(address, size, MPI_BYTE, MPI_to, DATA_SEND, INTRA_COMM);
       	if (ret != MPI_SUCCESS)	{
		MPI_Abort(INTRA_COMM, ret);
	}
}

void MPIMessenger::fetchData(const DataAccessRegion &region, const ClusterNode *from)
{
	int ret;
	const int MPI_from = from->getCommIndex();
	void *address = region.getStartAddress();
	size_t size = region.getSize();
	
	assert(MPI_from < wsize && MPI_from != wrank);
	
	ret = MPI_Recv(address, size, MPI_BYTE, MPI_from, DATA_SEND, INTRA_COMM, MPI_STATUS_IGNORE);
	if (ret != MPI_SUCCESS) {
		MPI_Abort(INTRA_COMM, ret);
	}
}

void MPIMessenger::synchronizeAll(void)
{
	int ret = MPI_Barrier(INTRA_COMM);
	if (ret != MPI_SUCCESS) {
		MPI_Abort(INTRA_COMM, ret);
	}
}

Message *MPIMessenger::checkMail(void)
{
	int ret, flag, count, type;
	MPI_Status status;
	Message::Deliverable *msg;
	
	ret = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, INTRA_COMM, &flag, &status);
	if (ret != MPI_SUCCESS) {
		MPI_Abort(INTRA_COMM, ret);
	}
	
	if (!flag) {
		return nullptr;
	}
	
	type = status.MPI_TAG;
	if (type == DATA_SEND) {
		return nullptr;
	}
	
	ret = MPI_Get_count(&status, MPI_BYTE, &count);
	if (ret != MPI_SUCCESS) {
		std::cerr << "Error while trying to determing size of message\n" << std::endl;
		MPI_Abort(INTRA_COMM, ret);
	}
	
	msg = (Message::Deliverable *)malloc(count);
	if (!msg) {
		perror("malloc for message");
		MPI_Abort(INTRA_COMM, 1);
	}
	
	assert(count != 0);
	ret = MPI_Recv((void *)msg, count, MPI_BYTE, status.MPI_SOURCE,
			status.MPI_TAG, INTRA_COMM, MPI_STATUS_IGNORE);
	if (ret != MPI_SUCCESS) {
		std::cerr << "Error receiving incoming message" << std::endl;
		MPI_Abort(INTRA_COMM, ret);
	}
	
	return GenericFactory<int, Message*, Message::Deliverable*>::getInstance().create(status.MPI_TAG, msg);
}
