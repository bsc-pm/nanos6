/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 500
#endif


#include <dirent.h>
#include <errno.h>
#include <fstream>
#include <ftw.h>
#include <iostream>
#include <libgen.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "CTFAPI.hpp"
#include "CTFTrace.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

#include <MemoryAllocator.hpp>


ConfigVariable<std::string> CTFAPI::CTFTrace::_defaultTemporalPath("instrument.ctf.tmpdir");
ConfigVariable<std::string> CTFAPI::CTFTrace::_ctf2prvWrapper("instrument.ctf.conversor.location");
ConfigVariable<bool> CTFAPI::CTFTrace::_ctf2prvEnabled("instrument.ctf.conversor.enabled", true);
EnvironmentVariable<std::string> CTFAPI::CTFTrace::_systemPATH("PATH");

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
	ssize_t ret;
	size_t pathSize = 512;
	std::vector<char> fullPath(pathSize);
	bool found = false;

	// it is not possible to know the path length of a file pointed by a
	// proc symlink, (see man lstat). Hence, we can only try incrementally.

	do {
		ret = readlink("/proc/self/exe", fullPath.data(), pathSize);

		if (ret == -1)
			break;

		if (fullPath.size() == (size_t) ret) {
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
	const char *defaultTemporalPath;

	if (_defaultTemporalPath.isPresent()) {
		defaultTemporalPath = _defaultTemporalPath.getValue().c_str();
	} else {
		const char *tmpDir = getenv("TMPDIR");
		if (tmpDir != NULL)
			defaultTemporalPath = tmpDir;
		else
			defaultTemporalPath = "/tmp";
	}

	size_t len = strlen(defaultTemporalPath) + sizeof(templateName);
	char * templatePath = (char *) MemoryAllocator::alloc(len);
	templatePath[0] = 0;

	strcat(templatePath, defaultTemporalPath);
	strcat(templatePath, templateName);

	ret = mkdtemp(templatePath);
	FatalErrorHandler::failIf(
		ret == NULL,
		"ctf: failed to create temporal trace directory: ",
		strerror(errno)
	);

	_tmpTracePath = std::string(templatePath);
	_finalTraceBasePath = std::string(tracePath);
	MemoryAllocator::free(templatePath, len);
}

void CTFAPI::CTFTrace::createTraceDirectories(std::string &basePath, std::string &userPath, std::string &kernelPath)
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

	basePath   = _tmpTracePath;
	userPath   = _userPath;
	kernelPath = _kernelPath;
}

void CTFAPI::CTFTrace::moveTemporalTraceToFinalDirectory()
{
	// TODO do not copy the trace if it's located in the same filesystem,
	// just rename it
	//
	std::cout << "Moving trace to current directory, please wait " << std::flush;

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

	std::cout << "[DONE] " << std::endl;
}

static bool isExecutable(const char *file)
{
	struct stat sb;

	if ((stat(file, &sb) == 0)) {
		if (sb.st_mode & S_IXUSR ||
		    sb.st_mode & S_IXGRP ||
		    sb.st_mode & S_IXOTH) {
			return true;
		}
	}

	return false;
}

static bool findInDir(const char *targetCommand, const char *dirPath)
{
	DIR *dir;

	if (!(dir = opendir(dirPath)))
		return false;

	dirent *dirEntry;
	while ((dirEntry = readdir(dir))) {
		if (strcmp(targetCommand, dirEntry->d_name) == 0) {
			std::string file = std::string(dirPath) + "/" + dirEntry->d_name;
			if (isExecutable(file.c_str()))
				return true;
		}
	}

	return false;
}

static bool findCommand(const char *targetCommand, std::string path)
{
	size_t start = 0;
	size_t end;

	while ((end = path.find(":", start)) != std::string::npos) {
		std::string currentPath = path.substr(start, end - start);

		if (findInDir(targetCommand, currentPath.c_str()))
			return true;

		start = end + 1;
	}

	std::string currentPath = path.substr(start, path.size() - end);
	if (findInDir(targetCommand, currentPath.c_str()))
		return true;

	return false;
}

void CTFAPI::CTFTrace::convertToParaver()
{
	int ret;
	const char defaultConverter[] = "ctf2prv";
	std::string converter;
	std::string command;

	// is conversion enabled?
	if (!_ctf2prvEnabled.getValue())
		return;

	// should we use a machine-specific wrapper?
	if (_ctf2prvWrapper.isPresent()) {
		converter = _ctf2prvWrapper.getValue();
	} else {
		const char *envConverter = getenv("CTF2PRV");

		if (envConverter != NULL) {
			converter = std::string(envConverter);
		} else {
			// if not, is the default converter in the system path?
			if (!findCommand(defaultConverter, _systemPATH.getValue())) {
				FatalErrorHandler::warn("The ", defaultConverter, " tool is not in the system PATH. Automatic ctf to prv conversion is not possible.");
				return;
			}

			converter = std::string(defaultConverter);
		}
	}

	std::cout << "Converting CTF trace to Paraver, please wait " << std::flush;

	// perform the conversion!
	command = converter + " " + _tmpTracePath;
	ret = system(command.c_str());
	FatalErrorHandler::warnIf(
		ret == -1,
		"ctf: automatic ctf to prv conversion failed: ",
		strerror(errno)
	);

	std::cout << "[DONE]" << std::endl;
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
	delete _userMetadata;
	delete _kernelMetadata;
}
