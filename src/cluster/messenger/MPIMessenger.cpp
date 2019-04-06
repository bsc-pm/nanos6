#include <cstdlib>
#include <vector>

#include "MPIMessenger.hpp"
#include "cluster/messages/Message.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

#include <ClusterNode.hpp>
#include <MemoryAllocator.hpp>

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

void MPIMessenger::sendMessage(Message *msg, ClusterNode const *toNode, bool block)
{
	int ret;
	Message::Deliverable *delv = msg->getDeliverable();
	const int mpiDst = toNode->getCommIndex();
	size_t msgSize = sizeof(delv->header) + delv->header.size;
	
	//! At the moment we use the Message type as the MPI
	//! tag of the communication
	int tag = delv->header.type;

	assert(mpiDst < wsize && mpiDst != wrank);
	assert(delv->header.size != 0);
	
	if (block) {
		ret = MPI_Send((void *)delv, msgSize, MPI_BYTE, mpiDst,
				tag, INTRA_COMM);
		return;
	}
	
	MPI_Request *request =
		(MPI_Request *)MemoryAllocator::alloc(
				sizeof(MPI_Request));
	FatalErrorHandler::failIf(
		request == nullptr,
		"Could not allocate memory for MPI_Request"
	);
	
	ret = MPI_Isend((void *)delv, msgSize, MPI_BYTE, mpiDst,
			tag, INTRA_COMM, request);
	if (ret != MPI_SUCCESS) {
		MPI_Abort(INTRA_COMM, ret);
	}
	
	msg->setMessengerData((void *)request);
}

void MPIMessenger::sendData(const DataAccessRegion &region, const ClusterNode *to)
{
	int ret;
	const int mpiDst = to->getCommIndex();
	void *address = region.getStartAddress();
	size_t size = region.getSize();
	
	assert(mpiDst < wsize && mpiDst != wrank);
	
	ret = MPI_Send(address, size, MPI_BYTE, mpiDst, DATA_SEND, INTRA_COMM);
       	if (ret != MPI_SUCCESS)	{
		MPI_Abort(INTRA_COMM, ret);
	}
}

void MPIMessenger::fetchData(const DataAccessRegion &region, const ClusterNode *from)
{
	int ret;
	const int mpiSrc = from->getCommIndex();
	void *address = region.getStartAddress();
	size_t size = region.getSize();
	
	assert(mpiSrc < wsize && mpiSrc != wrank);
	
	ret = MPI_Recv(address, size, MPI_BYTE, mpiSrc, DATA_SEND, INTRA_COMM, MPI_STATUS_IGNORE);
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

void MPIMessenger::testMessageCompletion(
	std::deque<Message *> &messages,
	std::deque<Message *> &completed
) {
	int msgCount = messages.size();
	MPI_Request requests[msgCount];
	
	for (int i = 0; i < msgCount; ++i) {
		Message *msg = messages[i];
		MPI_Request *req =
			(MPI_Request *)msg->getMessengerData();
		requests[i] = *req;
	}
	
	for (int i = 0; i < msgCount; ++i) {
		int index, ret, flag;
		MPI_Status status;
		
		ret = MPI_Testany(msgCount, requests, &index, &flag, &status);
		FatalErrorHandler::failIf(
			ret != MPI_SUCCESS,
			"Error during MPI_Testany"
		);
		
		//! None finished request
		if (!flag) {
			break;
		}
		
		//! Message at position 'index' has been delivered.
		//! Remove it from 'messages' and add it to completed
		completed.push_back(messages[index]);
		messages.erase(messages.begin() + index);
		
		//! Deallocate the MPI_Request object of the
		//! completed Message
		MemoryAllocator::free(
			requests[index],
			sizeof(MPI_Request)
		);
	}
}
