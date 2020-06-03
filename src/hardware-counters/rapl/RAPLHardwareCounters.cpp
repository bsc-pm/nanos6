/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/utsname.h>

#include "RAPLHardwareCounters.hpp"
#include "executors/threads/CPUManager.hpp"
#include "hardware/HardwareInfo.hpp"


RAPLHardwareCounters::RAPLHardwareCounters(bool verbose, const std::string &verboseFile)
{
	_verbose = verbose;
	_verboseFile = verboseFile;

	// Detect CPU and Package capabilities
	raplDetectCPU();
	raplDetectPackages();

	// Initialize the library
	raplInitialize();
}

RAPLHardwareCounters::~RAPLHardwareCounters()
{
	// Shutdown the library and print statistics if verbose is enabled
	raplShutdown();
}

void RAPLHardwareCounters::raplDetectCPU()
{
	FILE *file = fopen("/proc/cpuinfo", "r");
	FatalErrorHandler::failIf(file == nullptr, "Could not initialize the RAPL Power library");

	int family;
	char buffer[BUFSIZ]; // BUFSIZ == Max input stream size (cstdio)
	char vendor[BUFSIZ];

	// Read 'BUFSIZE' chunks from 'file' until it is empty
	char *line = fgets(buffer, BUFSIZ, file);
	while (line != nullptr) {
		// Line == vendor_id
		if (!strncmp(line, "vendor_id", 9)) {
			__attribute__((unused)) int ret = sscanf(line, "%*s%*s%s", vendor);
			assert(ret != EOF);

			if (strncmp(vendor, "GenuineIntel", 12)) {
				// Not an Intel chip, can't read power counters from this arch
				FatalErrorHandler::fail("Current architecture not supported by the RAPL Power library");
			}
		}

		// Line == cpu family
		if (!strncmp(line, "cpu family", 10)) {
			__attribute__((unused)) int ret = sscanf(line, "%*s%*s%*s%d", &family);
			assert(ret != EOF);

			if (family != 6) {
				// Can't read power counters if CPU family != 6
				FatalErrorHandler::fail("Current architecture not supported by the RAPL Power library");
			}
		}

		// Read the next line
		line = fgets(buffer, BUFSIZ, file);
	}
}

void RAPLHardwareCounters::raplDetectPackages()
{
	_numCPUs = CPUManager::getTotalCPUs();
	_numPackages = HardwareInfo::getNumPhysicalPackages();
	assert(_numPackages > 0);
}

void RAPLHardwareCounters::raplInitialize()
{
	size_t j;
	FILE *file;
	char tempFileName[RAPL_BUFFER_SIZE];
	char baseNames[RAPL_MAX_PACKAGES][RAPL_BUFFER_SIZE];

	// Iterate all the power metrics and save the current values for each package
	for (size_t i = 0; i < _numPackages; ++i) {
		j = 0;

		// Save the base name of the current package
		__attribute__((unused)) int ret = snprintf(baseNames[i], RAPL_BUFFER_SIZE,  "/sys/class/powercap/intel-rapl/intel-rapl:%zu", i);
		assert(ret >= 0);

		// Use a temporary string for the complete file name
		ret = snprintf(tempFileName, RAPL_BUFFER_SIZE, "%s/name", baseNames[i]);
		assert(ret >= 0);

		// Obtain all the available counter (event) names for this package
		file = fopen(tempFileName, "r");
		FatalErrorHandler::failIf(file == nullptr, "Could not open the needed files for the RAPL Power library");
		ret = fscanf(file, "%s", _eventNames[i][j]);
		assert(ret != EOF);

		_validEvents[i][j] = true;
		fclose(file);

		// Save the interested file name for later usage
		ret = snprintf(_fileNames[i][j], RAPL_BUFFER_SIZE, "%s/energy_uj", baseNames[i]);
		assert(ret >= 0);

		// Iterate each subdomain
		for (j = 1; j < RAPL_NUM_DOMAINS; ++j) {
			ret = snprintf(tempFileName, RAPL_BUFFER_SIZE, "%s/intel-rapl:%zu:%zu/name", baseNames[i], i, j - 1);
			assert(ret >= 0);

			file = fopen(tempFileName, "r");
			if (file == nullptr) {
				_validEvents[i][j] = false;
			} else {
				_validEvents[i][j] = true;
				ret = fscanf(file, "%s", _eventNames[i][j]);
				assert(ret != EOF);

				fclose(file);
				ret = snprintf(_fileNames[i][j], RAPL_BUFFER_SIZE, "%s/intel-rapl:%zu:%zu/energy_uj", baseNames[i], i, j - 1);
				assert(ret >= 0);
			}
		}
	}

	// Gather the initial events
	raplReadCounters(_startValues);
}

void RAPLHardwareCounters::raplReadCounters(size_t values[RAPL_MAX_PACKAGES][RAPL_NUM_DOMAINS])
{
	FILE *file;
	for (size_t i = 0; i < _numPackages; ++i) {
		for (size_t j = 0; j < RAPL_NUM_DOMAINS; ++j) {
			if (_validEvents[i][j]) {
				file = fopen(_fileNames[i][j], "r");
				FatalErrorHandler::failIf(file == nullptr, "Could not read power counters with RAPL");
				__attribute__((unused)) int ret = fscanf(file, "%zu", &(values[i][j]));
				assert(ret != EOF);

				fclose(file);
			}
		}
	}
}

void RAPLHardwareCounters::raplShutdown()
{
	// Gather the finishing values
	raplReadCounters(_finishValues);

	if (_verbose) {
		displayStatistics();
	}
}

void RAPLHardwareCounters::displayStatistics() const
{
	// 	Try opening the output file
	std::ios_base::openmode openMode = std::ios::out;
	std::ofstream output(_verboseFile, openMode);
	FatalErrorHandler::warnIf(
		!output.is_open(),
		"Could not create or open the verbose file: ", _verboseFile, ". ",
		"Using standard output."
	);

	// Retrieve statistics
	std::stringstream outputStream;
	outputStream << std::left << std::fixed << std::setprecision(5);
	outputStream << "-------------------------------\n";
	outputStream <<
		std::setw(7)  << "STATS"              << " " <<
		std::setw(6)  << "RAPL"               << " " <<
		std::setw(20) << "PACKAGE IDENTIFIER" << " " <<
		std::setw(9)  << "JOULES"             << "\n";

	// For each package and domain, print power statistics for the whole execution
	for (size_t i = 0; i < _numPackages; ++i) {
		for (size_t j = 0; j < RAPL_NUM_DOMAINS; ++j) {
			if (_validEvents[i][j]) {
				double joules = (((double)_finishValues[i][j] - (double)_startValues[i][j]) / 1000000.0);
				outputStream <<
					std::setw(7)  << "STATS"           << " " <<
					std::setw(6)  << "RAPL"            << " " <<
					std::setw(20) << _eventNames[i][j] << " " <<
					std::setw(9)  << joules            << "\n";
			}
		}
	}

	outputStream << "-------------------------------\n";

	if (output.is_open()) {
		// Output into the file and close it
		output << outputStream.str();
		output.close();
	} else {
		std::cout << outputStream.str();
	}
}
