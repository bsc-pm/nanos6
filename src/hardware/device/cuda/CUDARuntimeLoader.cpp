/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#include "CUDARuntimeLoader.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>

#if defined(USE_CUDA_CL)
#include <CL/cl.hpp>
#endif

#include <dirent.h>
#include <elf.h>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>

#include "lowlevel/FatalErrorHandler.hpp"
#include "support/config/ConfigCentral.hpp"
#include "support/config/ConfigVariable.hpp"

const std::string CUDA_BINARY_STRING =
	"__cuda__binary__ompss_2_nanos6_internal_error";

CUfunction CUDARuntimeLoader::loadFunction(const char *str)
{
	int currentDevice;
	cudaGetDevice(&currentDevice);
	auto it = _functionCache[currentDevice].find(str);
	if (it != _functionCache[currentDevice].end())
		return it->second;

	CUfunction fnc;
	for (CUmodule mod : _perGpuModules[currentDevice]) {
		if (cuModuleGetFunction(&fnc, mod, str) == CUDA_SUCCESS) {
			_functionCache[currentDevice][str] = fnc;
			return fnc;
		}
	}

	FatalErrorHandler::fail("Cuda function: [", str, "] was not found, can't continue.\n");
	return (CUfunction)0;
}

static Elf64_Shdr *findElfSection(char *elfFile, std::string search)
{
	// In an ELF file, there is a "header" and then some "section headers" which describe each section
	// In this section header, the name is a number which corresponds to a string in a special table

	// Map exe to the ELF header
	Elf64_Ehdr *ehdr = (Elf64_Ehdr *)elfFile;
	// Then find where the ELF section headers start
	Elf64_Shdr *shdr = (Elf64_Shdr *)(elfFile + ehdr->e_shoff);
	// Now get the number of sections
	Elf32_Half shnum = ehdr->e_shnum;

	// e_shstrndx is the index of the string table section's header
	Elf64_Shdr *shStrtab = &shdr[ehdr->e_shstrndx];
	// Now we use sh_offset which contains the real location of the section in the executable
	const char *const shStrtabExe = elfFile + shStrtab->sh_offset;

	// Now, for each section, check if it matches the name
	for (Elf32_Half i = 0; i < shnum; ++i) {
		std::string sectionName = std::string(&shStrtabExe[shdr[i].sh_name]);
		if (sectionName == search) {
			return &shdr[i];
		}
	}

	return NULL;
}

void CUDARuntimeLoader::getSectionBinaryModules(std::vector<nameData> &cuObjs)
{
	const char *selfExe = "/proc/self/exe";

	struct stat st;
	if (stat(selfExe, &st) != 0) {
		FatalErrorHandler::fail("Cannot stat file: ", selfExe);
	}

	int fd = open(selfExe, O_RDONLY);
	FatalErrorHandler::failIf(fd == -1, "Cannot open executable: ", selfExe);

	char *exe = (char *)mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	FatalErrorHandler::failIf(exe == MAP_FAILED, "Cannot mmap executable in read-only mode");

	Elf64_Shdr *fatbinSegmentHeader = (Elf64_Shdr *)findElfSection(exe, ".nvFatBinSegment");
	Elf64_Shdr *fatbinSectionHeader = (Elf64_Shdr *)findElfSection(exe, ".nv_fatbin");

	// No kernels found
	if (!fatbinSegmentHeader || !fatbinSectionHeader)
		return;

	nvFatbinSegment *fatbinSegment = (nvFatbinSegment *)&exe[fatbinSegmentHeader->sh_offset];
	Elf64_Addr fatbinSectionAddress = fatbinSectionHeader->sh_addr;

	assert(fatbinSegmentHeader->sh_size > 0);
	assert(fatbinSegmentHeader->sh_size % sizeof(nvFatbinSegment) == 0);

	for (size_t i = 0; i < (fatbinSegmentHeader->sh_size / sizeof(nvFatbinSegment)); ++i) {
		nvFatbinSegment *currSegmentInfo = &fatbinSegment[i];
		assert(currSegmentInfo->magic == NV_FATBIN_SEGMENT_MAGIC);

		uint64_t currentFatbinOffset = fatbinSectionHeader->sh_offset +
			(currSegmentInfo->fatbinAddress - fatbinSectionAddress);
		assert(fatbinSectionAddress <= currSegmentInfo->fatbinAddress);
		nvFatbinDesc *desc = (nvFatbinDesc *)&exe[currentFatbinOffset];

		assert(desc->magic == NV_FATBIN_DESC_MAGIC);
		// Copy the full fatbin AND its header, which is not included in the size
		cuObjs.emplace_back(CUDA_BINARY_STRING,
			std::string((char *)desc, desc->size + sizeof(nvFatbinDesc)));
	}

	// We don't care if these calls fail
	munmap(exe, st.st_size);
	close(fd);
}

#if defined(USE_CUDA_CL)
std::vector<std::string>
CUDARuntimeLoader::compileAllClKernels(const std::vector<nameData> &sourcefiles)
{
	if (sourcefiles.size() == 0)
		return {};
	std::vector<cl::Platform> allPlats;
	cl::Platform::get(&allPlats);

	if (allPlats.size() == 0) {
		return {};
	}

	cl::Platform cudaPlatform;

	bool nvidiaPlatFound = false;
	for (cl::Platform plat : allPlats) {
		if (plat.getInfo<CL_PLATFORM_NAME>() == "NVIDIA CUDA") {
			cudaPlatform = plat;
			nvidiaPlatFound = true;
			break;
		}
	}


	if (!nvidiaPlatFound)
		return {};


	std::vector<cl::Device> allDevices;
	cudaPlatform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);

	if (allDevices.size() == 0) {
		FatalErrorHandler::warn("Failed to find  CL-capable devices");
		return {};
	}


	cl::Context context(allDevices);
	cl::Program::Sources sources;

	for (const nameData &src : sourcefiles) {
		sources.push_back({src.data.c_str(), src.data.size()});
	}

	cl::Program program(context, sources);
	if (program.build(allDevices) != CL_SUCCESS) {
		FatalErrorHandler::warn("Failed to compile CL kernels");
		return {};
	}

	std::vector<size_t> binarySizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
	std::vector<char *> compiledPrograms = program.getInfo<CL_PROGRAM_BINARIES>();

	std::vector<std::string> gpuBins(binarySizes.size());
	for (size_t i = 0; i < binarySizes.size(); ++i) {
		gpuBins[i] = std::string(compiledPrograms[i], binarySizes[i]);
	}

	return gpuBins;
}
#endif


std::vector<std::string> CUDARuntimeLoader::compileSourcesIntoPtxForContext(
	const std::vector<nameData> &sourcefiles,
	const std::vector<nameData> &headers, int gpu)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, gpu);
	const int capabilityNumber = deviceProp.major * 10 + deviceProp.minor;

	const size_t numHeaders = headers.size();
	std::vector<const char *> headerFiles(numHeaders);
	std::vector<const char *> headerNames(numHeaders);
	for (size_t i = 0; i < numHeaders; ++i) {
		headerNames[i] = (char *)strrchr(headers[i].filename.c_str(), '/') + 1;
		headerFiles[i] = (char *)headers[i].data.c_str();
	}

	std::vector<std::string> ptxFiles;
	for (const auto &binaryNameAndData : sourcefiles) {
		const std::string &sourcefile = binaryNameAndData.data;
		nvrtcProgram prog;
		nvrtcResult rc;
		rc = nvrtcCreateProgram(&prog, sourcefile.c_str(), NULL, numHeaders,
			headerFiles.data(), headerNames.data());
		if (rc != NVRTC_SUCCESS) {
			std::string errorMsg = "Failed to create CUDA program: ";
			errorMsg += nvrtcGetErrorString(rc);
			FatalErrorHandler::warn(errorMsg);
			continue;
		}

		const char *includeOption = "-I.";
		std::string archirectureOption = "--gpu-architecture=compute_" + std::to_string(capabilityNumber);
		const char *options[] = {includeOption, archirectureOption.c_str(), NULL};
		rc = nvrtcCompileProgram(prog, 2, options);

		if (rc != NVRTC_SUCCESS) {
			std::string errorMsg = "Failed to compile CUDA program: ";
			errorMsg += " for source file: ";
			errorMsg += binaryNameAndData.filename;
			errorMsg +=
				"\nThis usually happens when there are c++ headers in the source "
				"code that are not in the folder (this applies to standard headers)";
			errorMsg += nvrtcGetErrorString(rc);
			errorMsg += "\n";
			FatalErrorHandler::warn(errorMsg);
			continue;
		}

		size_t ptxImageLen;
		rc = nvrtcGetPTXSize(prog, &ptxImageLen);
		if (rc != NVRTC_SUCCESS) {
			FatalErrorHandler::warn("Failed to get PTX size: ",
				nvrtcGetErrorString(rc));
		}
		std::string ptxImage(ptxImageLen + 1, '\0');
		rc = nvrtcGetPTX(prog, &ptxImage[0]);
		ptxFiles.emplace_back(ptxImage);
	}
	return ptxFiles;
}

std::vector<std::string>
CUDARuntimeLoader::getAllCudaUserBinariesPath(const std::string &path)
{
	std::vector<std::string> paths;

	DIR *dir = opendir(path.c_str());
	if (dir) {
		// add all files path in the directory to paths vector
		struct dirent *ent;
		while ((ent = readdir(dir)) != nullptr) {
			if (ent->d_type == DT_REG) {
				paths.push_back(path + "/" + ent->d_name);
			}
		}
		closedir(dir);
	}
	return paths;
}

CUDARuntimeLoader::CUDARuntimeLoader()
{
	int devNum;
	cudaError_t err = cudaGetDeviceCount(&devNum);
	FatalErrorHandler::failIf(err != cudaSuccess, "Failed to get device count");

	std::vector<CUdevice> devices(devNum);
	std::vector<CUcontext> contexts(devNum);

	// Initialize the primary context, this context is special and shared with the runtime api
	for (int i = 0; i < devNum; ++i) {
		FatalErrorHandler::failIf(cuDeviceGet(&devices[i], i) != CUDA_SUCCESS,
			"Failed to get device " + std::to_string(i));
		FatalErrorHandler::failIf(cuDevicePrimaryCtxRetain(&contexts[i], devices[i]) != CUDA_SUCCESS,
			"Failed to retain primary context for device " + std::to_string(i));
	}

	_functionCache.resize(devNum);
	_perGpuModules.resize(devNum);

	// load kernels from the kernels folder
	std::string kernels{
		ConfigVariable<std::string>("devices.cuda.kernels_folder")};

	std::vector<nameData> cuObjs;
	std::vector<nameData> cuSrcs;
	std::vector<nameData> cuHeaders;
#if defined(USE_CUDA_CL)
	std::vector<nameData> clSrcs;
#endif

	for (const std::string &dirEntry : getAllCudaUserBinariesPath(kernels)) {
		const auto entryEndsWith = [&](const std::string &ext) -> bool {
			return dirEntry.size() > ext.size() && dirEntry.substr(dirEntry.size() - ext.size()) == ext;
		};

		if (entryEndsWith(".c") || entryEndsWith(".cpp"))
			continue;
#if defined(USE_CUDA_CL)
		else if (entryEndsWith(".cl")) {
			clSrcs.emplace_back(dirEntry, getFileContent(dirEntry));
		}
#endif
		else if (entryEndsWith(".hpp") || entryEndsWith(".h")) {
			cuHeaders.emplace_back(dirEntry, getFileContent(dirEntry));
		} else if (entryEndsWith(".cu")) {
			cuSrcs.emplace_back(dirEntry, getFileContent(dirEntry));
		} else if (entryEndsWith(".co") || entryEndsWith(".ptx") || entryEndsWith(".o") || entryEndsWith(".cuo")) {
			cuObjs.emplace_back(dirEntry, getFileContent(dirEntry));
		} else {
			FatalErrorHandler::warn("Unknown file type in nanos6 kernels directory: ", dirEntry);
		}
	}

	// When this is emplaced after the folder elements, the kernels in the folder
	// are overriding implemented functions
	getSectionBinaryModules(cuObjs);

#if defined(USE_CUDA_CL)
	std::vector<std::string> clSourcePtx = compileAllClKernels(clSrcs);
#endif

	bool warningEnabled =
		ConfigVariable<bool>("devices.cuda.warning_on_incompatible_binary");
	for (size_t i = 0; i < contexts.size(); ++i) {
		cuCtxSetCurrent(contexts[i]);

		std::vector<std::string> cuSourcePtx =
			compileSourcesIntoPtxForContext(cuSrcs, cuHeaders, i);
		for (std::string &currentPtx : cuSourcePtx) {
			CUmodule tmp;
			if (cuModuleLoadData(&tmp, &currentPtx[0]) == CUDA_SUCCESS)
				_perGpuModules[i].push_back(tmp);
			else
				FatalErrorHandler::warn("Could not load compiled ptx");
		}

#if defined(USE_CUDA_CL)
		for (std::string &currentPtx : clSourcePtx) {
			CUmodule tmp;
			if (cuModuleLoadData(&tmp, &currentPtx[0]) == CUDA_SUCCESS)
				_perGpuModules[i].push_back(tmp);
			else
				FatalErrorHandler::warn("Could not load Cl compiled ptx");
		}
#endif


		for (nameData &cuObj : cuObjs) {
			CUmodule tmp;
			if (cuModuleLoadData(&tmp, &cuObj.data[0]) == CUDA_SUCCESS)
				_perGpuModules[i].push_back(tmp);
			else {
				if (cuObj.filename == CUDA_BINARY_STRING) {
					if (!warningEnabled)
						continue;

					const char *green = "\033[1;32m";
					const char *reset = "\033[0m";
					const char *red = "\033[1;31m";

					cudaDeviceProp deviceProp;
					cudaGetDeviceProperties(&deviceProp, i);
					const int capabilityNumber = deviceProp.major * 10 + deviceProp.minor;

					fprintf(stderr, "\n");

					FatalErrorHandler::warn(
						"The embedded fatbin couldn't be loaded for GPU: ", i);

					FatalErrorHandler::warn(
						"This usually means that the compute capabilities of the "
						"embedded binary Doesn't match the one of the current device.");

					FatalErrorHandler::warn(
						"To fix this you can add an implementation of the cuda functions "
						"as a ptx/fatbin to the folder: ",
						kernels);

					FatalErrorHandler::warn(
						"Please, make sure that those implementations are compiled using "
						"the correct capabilities of your CUDA-Capable device.");

					fprintf(stderr, "%s", green);
					FatalErrorHandler::warn("On CLANG you probably want to add the "
											"compilation flag --cuda-gpu-arch=sm_",
						capabilityNumber);

					const std::string nvcc_string =
						"On NVCC you probably want to add the compilation flag "
						"--generate-code arch=compute_"
						+ std::to_string(capabilityNumber) + ",code=sm_" + std::to_string(capabilityNumber);

					FatalErrorHandler::warn(nvcc_string);
					fprintf(stderr, "%s", reset);

					FatalErrorHandler::warn(
						"If you have more than one GPUs with different capabilities, try "
						"to compile the cuda code for all the needed capabilities.");
					fprintf(stderr, "%s", red);
					FatalErrorHandler::warn(
						"If you want to silence this warning, go to the nanos6 config "
						"file and set warning_on_incompatible_binary to false");
					fprintf(stderr, "%s", reset);

				} else
					FatalErrorHandler::warn("Could not load ptx file: ", cuObj.filename);
			}
		}
	}
}

std::string CUDARuntimeLoader::getFileContent(const std::string &filename)
{
	std::ifstream file(filename, std::ios::binary | std::ios::ate);
	std::streamsize size = file.tellg();
	file.seekg(0, std::ios::beg);
	std::string buffer;
	buffer.resize(size);
	file.read((char *)&buffer[0], size);
	return buffer;
}
