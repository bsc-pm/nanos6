/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef MPI_MESSENGER_HPP
#define MPI_MESSENGER_HPP

#include <sstream>
#include <vector>

#pragma GCC visibility push(default)
#include <mpi.h>
#pragma GCC visibility pop

#include "Messenger.hpp"

class ClusterPlace;
class DataTransfer;
class Message;

class MPIMessenger : public Messenger {
private:
	int wrank, wsize;
	MPI_Comm INTRA_COMM, PARENT_COMM;
	
public:
	MPIMessenger();
	~MPIMessenger();
	
	void sendMessage(Message *msg, ClusterNode const *toNode, bool block = false);
	void synchronizeAll(void);
	DataTransfer *sendData(const DataAccessRegion &region, const ClusterNode *toNode, int messageId);
	DataTransfer *fetchData(const DataAccessRegion &region, const ClusterNode *fromNode, int messageId);
	Message *checkMail();
	void testMessageCompletion(std::vector<Message *> &messages);
	void testDataTransferCompletion(std::vector<DataTransfer *> &transfers);
	
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
	
	//! \brief Check the return value of an MPI operation for success
	//!
	//! \param[in] retval is a value returned by an MPI operation
	//! \param[in] comm is the MPI communicator on which the operation that
	//!		returned retval operated
	inline void checkSuccess(int retval, MPI_Comm comm)
	{
		char errorString[MPI_MAX_ERROR_STRING];
		int stringLength;
		int errorClass;
		std::stringstream ss;
		
		if (retval == MPI_SUCCESS) {
			return;
		}
		
		ss << "[MPI RANK " << wrank << "]: ";
		
		MPI_Error_class(retval, &errorClass);
		MPI_Error_string(errorClass, errorString,
			&stringLength);
		ss << errorString << " ";
		
		MPI_Error_string(retval, errorString,
			&stringLength);
		ss << errorString;
		std::cerr << ss.str() << std::endl;
		MPI_Abort(comm, retval);
	}
	
	//! \brief Check the status of a number of MPI operations for success
	//!
	//! Some MPI functions such as `MPI_Testsome` return error values in
	//! the MPI_Status objects related with the operation. We use this
	//! method to check for these errors
	//!
	//! \param[in] retval is a value returned by an MPI operation
	//! \param[in] status is an array of MPI_Status objects in which we
	//!		check for MPI errors
	//! \param[in] nr_status is the size of the 'status' array
	//! \param[in] comm is the MPI communicator on which the operation that
	//!		returned retval operated
	inline void checkSuccess(int retval, MPI_Status *status, int nr_status,
			MPI_Comm comm)
	{
		if (retval == MPI_SUCCESS) {
			return;
		}
		
		for (int i = 0; i < nr_status; ++i) {
			checkSuccess(status[i].MPI_ERROR, comm);
		}
	}
};

//! Register MPIMessenger with the object factory
namespace
{
	Messenger *createMPImsn() { return new MPIMessenger; }
	
	const bool __attribute__((unused))_registered_MPI_msn =
		REGISTER_MSN_CLASS("mpi-2sided", createMPImsn);
}

#endif /* MPI_MESSENGER_HPP */
