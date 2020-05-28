/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>

#include "lowlevel/FatalErrorHandler.hpp"

#include "CTFTrace.hpp"
#include "CTFAPI.hpp"

void CTFAPI::CTFTrace::createTraceDirectories(std::string &userPath, std::string &kernelPath)
{
	int ret;

	_userPath   = _tracePath;
	_kernelPath = _tracePath;

	ret = mkdir(_tracePath.c_str(), 0766);
	FatalErrorHandler::failIf(ret, "Instrument: ctf: failed to create trace directories");

	_kernelPath += "/kernel";
	ret = mkdir(_kernelPath.c_str(), 0766);
	FatalErrorHandler::failIf(ret, "Instrument: ctf: failed to create trace directories");

	// TODO add timestamp?
	// TODO get folder name & path form env var?
	// TODO 1042 is the user id, get the real one

	_userPath += "/ust";
	ret = mkdir(_userPath.c_str(), 0766);
	FatalErrorHandler::failIf(ret, "Instrument: ctf: failed to create trace directories");
	_userPath += "/uid";
	ret = mkdir(_userPath.c_str(), 0766);
	FatalErrorHandler::failIf(ret, "Instrument: ctf: failed to create trace directories");
	_userPath += "/1042";
	ret = mkdir(_userPath.c_str(), 0766);
	FatalErrorHandler::failIf(ret, "Instrument: ctf: failed to create trace directories");
	_userPath += "/64-bit";
	ret = mkdir(_userPath.c_str(), 0766);
	FatalErrorHandler::failIf(ret, "Instrument: ctf: failed to create trace directories");

	userPath   = _userPath;
	kernelPath = _kernelPath;
}

void CTFAPI::CTFTrace::initializeTraceTimer(void)
{
	_absoluteStartTime = CTFAPI::getTimestamp();
}

void CTFAPI::CTFTrace::clean(void)
{
	delete _metadata;
}

