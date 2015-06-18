#ifndef LEADER_THREAD_HPP
#define LEADER_THREAD_HPP


#include <atomic>


//! \brief This class contains the code of the leader thread that consists in performing maintenance duties
class LeaderThread {
	static int _returnCode;
	static std::atomic<bool> _mustExit;
	
public:
	//! \brief A loop that takes care of maintenance duties
	//! The loop ends after notifyMainExit is called
	static void maintenanceLoop();
	
	//! \brief Signal the leather thread that the "main" function has finished and thus that the execution must end
	//!
	//! \param in returnCode the exit code
	static void notifyMainExit(int returnCode);
	
	//! \brief Returns the exit code of the "main" function
	//!
	//! \returns the exit code of the "main" function
	static int getMainReturnCode();
	
};


#endif // LEADER_THREAD_HPP
