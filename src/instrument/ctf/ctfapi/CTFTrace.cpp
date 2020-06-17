/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 500
#endif

#include <fstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <libgen.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <ftw.h>

#include <MemoryAllocator.hpp>
#include "lowlevel/FatalErrorHandler.hpp"
#include "CTFTrace.hpp"
#include "CTFAPI.hpp"


EnvironmentVariable<std::string> CTFAPI::CTFTrace::_defaultTemporalPath("TMPDIR", "/tmp");

static bool copyFile(std::string &src, std::string &dst)
{
	std::ifstream srcStream(src, std::ios::binary);
	std::ofstream dstStream(dst, std::ios::binary);
	dstStream << srcStream.rdbuf();
	if (srcStream.bad() || dstStream.bad()) {
		std::cerr << "Warning: ctf: Cannot copy file " << src << std::endl;
		return false;
	}
	return true;
}

static bool copyDir(std::string &src, std::string &dst)
{
	DIR *dir;
	bool status = true;
	std::string tmpsrc;
	std::string tmpdst;
	struct dirent *dentry;

	// open source directory
	dir = opendir(src.c_str());
	if (dir == NULL) {
		std::cerr << "Warning: ctf: Couldn't open temporal trace directory" << std::endl;
		return false;
	}

	// create destination directory
	if (mkdir(dst.c_str(), 0700) != 0) {
		std::cerr << "Warning: ctf: Couldn't create final trace directory " << dst << std::endl;
		return false;
	}

	// iterate entries in directory and copy them
	while (status && ((dentry = readdir(dir)) != NULL)) {
		struct stat st;

		if ((strcmp(dentry->d_name, ".")  == 0) ||
		    (strcmp(dentry->d_name, "..") == 0))
		{
			continue;
		}

		tmpsrc = src + "/" + dentry->d_name;
		tmpdst = dst + "/" + dentry->d_name;

		if (stat(tmpsrc.c_str(), &st) != 0) {
			std::cerr << "Warning: ctf: Source file " << tmpsrc << " does not exist" << std::endl;
			status = false;
			break;
		}

		if (S_ISREG(st.st_mode)) {
			status = copyFile(tmpsrc, tmpdst);
		} else if (S_ISDIR(st.st_mode)) {
			status = copyDir(tmpsrc, tmpdst);
		}
	}

	if (closedir(dir))
		std::cerr << "Warning: ctf: Failed to close source directory" << std::endl;

	return status;
}

static int remove_cb(const char *fpath,
		     __attribute__((unused)) const struct stat *sb,
		     __attribute__((unused)) int typeflag,
		     __attribute__((unused)) struct FTW *ftwbuf
) {
	int rv;
	rv = remove(fpath);
	if (rv)
		std::cerr << "Warning: ctf: Failed to remove " << fpath << std::endl;
	return rv;
}

static bool removeDir(std::string &path)
{
	int rv;
	rv = nftw(path.c_str(), remove_cb, 64, FTW_DEPTH | FTW_PHYS);
	return (rv == 0);
}

static std::string mkTraceDirectoryName(std::string finalTracePath,
					std::string &binaryName, uint64_t pid)
{
	int cnt;
	struct stat st;
	std::string traceDir, tmp;

	// build a name the trace directory name for the specified tracePath,
	// ensuring that no other directory with this name exist
	traceDir = finalTracePath;
	traceDir += "/trace_" + binaryName + "_" + std::to_string(pid);
	tmp = traceDir;
	cnt = 1;
	while (stat(tmp.c_str(), &st) == 0) {
		tmp = traceDir + "_" + std::to_string(cnt);
		cnt++;
	}

	return tmp;
}

static std::string getFullBinaryName()
{
	const char defaultName[] = "nanos6";
	std::string name;
	struct stat sb;
	int ret;
	size_t pathSize = 512;
	std::vector<char> fullPath(pathSize);
	bool found = false;

	// it is not possible to know the path length of a file pointed by a
	// proc symlink, (see man lstat). Hence, we can only try incrementally.

	do {
		ret = readlink("/proc/self/exe", fullPath.data(), pathSize);

		if (ret == -1)
			break;

		if (fullPath.size() == ret) {
			pathSize += 512;
			fullPath.resize(pathSize);
		} else {
			found = true;
			fullPath[ret] = 0;
		}
	} while (!found);

	if (found) {
		name = std::string(basename(fullPath.data()));
	} else {
		std::cerr << "Warning: ctf: Cannot get binary name, using default" << std::endl;
		name = std::string(defaultName);
	}

	return name;
}

CTFAPI::CTFTrace::CTFTrace()
{
	// get process PID
	_pid = (uint64_t) getpid();
	// get process full binary name
	_binaryName = getFullBinaryName();
}

void CTFAPI::CTFTrace::setTracePath(const char* tracePath)
{
	void *ret;
	char templateName[] = "/nanos6_trace_XXXXXX";
	char *defaultTemporalPath;
	const size_t len = _defaultTemporalPath.getValue().size() +
			   sizeof(templateName) + 1;

	// create a temporal directory under $TMPDIR or /tmp if not set
	defaultTemporalPath = (char *) MemoryAllocator::alloc(len);
	defaultTemporalPath[0] = 0;
	strcat(defaultTemporalPath, _defaultTemporalPath.getValue().c_str());
	strcat(defaultTemporalPath, templateName);
	ret = mkdtemp(defaultTemporalPath);
	FatalErrorHandler::failIf(
		ret == NULL,
		"ctf: failed to create temporal trace directory: ",
		strerror(errno)
	);

	_tmpTracePath = std::string(defaultTemporalPath);
	_finalTraceBasePath = std::string(tracePath);
	MemoryAllocator::free(defaultTemporalPath, len);
}

void CTFAPI::CTFTrace::createTraceDirectories(std::string &userPath, std::string &kernelPath)
{
	int ret;
	std::string tracePath = _tmpTracePath;

	tracePath += "/ctf";
	ret = mkdir(tracePath.c_str(), 0766);
	FatalErrorHandler::failIf(ret, "ctf: failed to create trace directories");

	_userPath   = tracePath;
	_kernelPath = tracePath;

	_kernelPath += "/kernel";
	ret = mkdir(_kernelPath.c_str(), 0766);
	FatalErrorHandler::failIf(ret, "ctf: failed to create trace directories");

	_userPath += "/ust";
	ret = mkdir(_userPath.c_str(), 0766);
	FatalErrorHandler::failIf(ret, "ctf: failed to create trace directories");
	_userPath += "/uid";
	ret = mkdir(_userPath.c_str(), 0766);
	FatalErrorHandler::failIf(ret, "ctf: failed to create trace directories");
	_userPath += "/1000";
	ret = mkdir(_userPath.c_str(), 0766);
	FatalErrorHandler::failIf(ret, "ctf: failed to create trace directories");
	_userPath += "/64-bit";
	ret = mkdir(_userPath.c_str(), 0766);
	FatalErrorHandler::failIf(ret, "ctf: failed to create trace directories");

	userPath   = _userPath;
	kernelPath = _kernelPath;
}

void CTFAPI::CTFTrace::moveTemporalTraceToFinalDirectory()
{
	// TODO do not copy the trace if it's located in the same filesystem,
	// just rename it

	// create final trace name
	std::string finalTracePath = mkTraceDirectoryName(_finalTraceBasePath,
							  _binaryName, _pid);
	// copy temporal trace into final destination
	if (!copyDir(_tmpTracePath, finalTracePath)) {
		std::cerr << "Warning: ctf: The trace could not be moved to the current directory. Please copy the trace manually"
			 << std::endl << "trace location: " << _tmpTracePath <<
			 std::endl;
		return;
	}
	// remove temporal trace
	if (!removeDir(_tmpTracePath)) {
		std::cerr << "Warning: ctf: The temporal trace directory " <<
			_tmpTracePath << "could not be removed. Please, remove it manually"
			<< std::endl;
	}
}

void CTFAPI::CTFTrace::initializeTraceTimer()
{
	// get absolute timestamp used to calculate relative timestamps of all
	// tracepoints. On Linux, this timestamp is actually relative to boot
	// time.
	_absoluteStartTime = CTFAPI::getTimestamp();
}

void CTFAPI::CTFTrace::clean()
{
	delete _metadata;
}
