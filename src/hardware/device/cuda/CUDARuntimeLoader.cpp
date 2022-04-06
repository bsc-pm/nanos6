/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#include "CUDARuntimeLoader.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>

#include <iostream>
#include <dirent.h>
#include <elf.h>
#include <fcntl.h>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/mman.h>

#include "lowlevel/FatalErrorHandler.hpp"
#include "support/config/ConfigCentral.hpp"
#include "support/config/ConfigVariable.hpp"

const std::string CUDA_BINARY_STRING = "__cuda__binary__ompss_2_nanos6_internal_error";

CUfunction CUDARuntimeLoader::loadFunction(const char *str)
{
	int currentDevice;
	cudaGetDevice(&currentDevice);
	auto it = _functionCache[currentDevice].find(str);
	if (it != _functionCache[currentDevice].end())
		return it->second;

	CUfunction fnc;
	for (CUmodule mod : _perGpuModles[currentDevice]) {
		if (cuModuleGetFunction(&fnc, mod, str) == CUDA_SUCCESS) {
			_functionCache[currentDevice][str] = fnc;
			return fnc;
		}
	}

	FatalErrorHandler::fail("Cuda function: [", str, "] was not found, can't continue.\n");
	return (CUfunction)0;
}


void CUDARuntimeLoader::getSectionBinaryModules(std::vector<name_data> &cuObjs)
{
	const char *selfExe = "/proc/self/exe";

	struct stat st;
	if (stat(selfExe, &st) != 0) {
		FatalErrorHandler::fail("Cannot stat file: ", selfExe, "\n");
	}

	int fd = open(selfExe, O_RDONLY);
	char *exe = (char *)mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

	Elf64_Ehdr *ehdr = (Elf64_Ehdr *)exe;
	Elf64_Shdr *shdr = (Elf64_Shdr *)(exe + ehdr->e_shoff);
	Elf32_Half shnum = ehdr->e_shnum;

	Elf64_Shdr *shStrtab = &shdr[ehdr->e_shstrndx];
	const char *const shStrtabExe = exe + shStrtab->sh_offset;

	for (Elf32_Half i = 0; i < shnum; ++i) {
		if (std::string(shStrtabExe + shdr[i].sh_name) == ".nv_fatbin") {
			nvFatbinDesc *desc = (nvFatbinDesc *)&exe[shdr[i].sh_offset];

			while (desc->magic == NV_FATBIN_DESC_MAGIC) {
				cuObjs.emplace_back(CUDA_BINARY_STRING, std::string((char *)desc, desc->size));

				desc = (nvFatbinDesc *)((char *)desc + desc->size + 16); // 16 = 4 magic + 4 version + 8 size
				desc = (nvFatbinDesc *)((((uintptr_t)desc) + 15) & ~15); // align 16 bytes
			}
			break;
		}
	}

	munmap(exe, st.st_size);
	close(fd);
}

std::vector<std::string> CUDARuntimeLoader::compileSourcesIntoPtxForContext(const std::vector<name_data> &sourcefiles, const std::vector<name_data> &headers, int gpu)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, gpu);
	const int capability_number = deviceProp.major * 10 + deviceProp.minor;

	const size_t num_headers = headers.size();
	const char *header_files[num_headers];
	const char *header_names[num_headers];
	for (size_t i = 0; i < num_headers; ++i) {
		header_names[i] = (char *)strrchr(headers[i].filename.c_str(), '/') + 1;
		header_files[i] = (char *)headers[i].data.c_str();
	}

	std::vector<std::string> ptx_files;
	for (const auto &n_d : sourcefiles) {
		const std::string &sourcefile = n_d.data;
		nvrtcProgram prog;
		nvrtcResult rc;
		rc = nvrtcCreateProgram(&prog, sourcefile.c_str(), NULL, num_headers, header_files, header_names);
		if (rc != NVRTC_SUCCESS) {
			std::string error_msg = "Failed to create CUDA program: ";
			error_msg += nvrtcGetErrorString(rc);
			FatalErrorHandler::warn(error_msg);
			continue;
		}

		char *include_option = (char *)"-I.";
		char architecture_option[32];
		snprintf(architecture_option, 32, "--gpu-architecture=compute_%d", capability_number);
		char *options[] = {include_option, architecture_option, NULL};
		rc = nvrtcCompileProgram(prog, 2, options);

		if (rc != NVRTC_SUCCESS) {
			std::string error_msg = "Failed to compile CUDA program: ";
			error_msg += " for source file: ";
			error_msg += n_d.filename;
			error_msg += "\nThis usually happens when there are c++ headers in the source code that are not in the folder (this applies to standard headers)";
			error_msg += nvrtcGetErrorString(rc);
			error_msg += "\n";
			FatalErrorHandler::warn(error_msg);
			continue;
		}

		size_t ptx_image_len;
		rc = nvrtcGetPTXSize(prog, &ptx_image_len);
		if (rc != NVRTC_SUCCESS) {
			FatalErrorHandler::warn("Failed to get PTX size: ", nvrtcGetErrorString(rc));
		}
		std::string ptx_image(ptx_image_len + 1, '\0');
		rc = nvrtcGetPTX(prog, &ptx_image[0]);
		ptx_files.emplace_back(ptx_image);
	}
	return ptx_files;
}


// To maintain compatibility with older versions of C++, and to avoid adding runtime dependencies we do it the old way.
std::vector<std::string> CUDARuntimeLoader::getAllCudaUserBinariesPath(const std::string &path)
{
	std::vector<std::string> paths;
	char cwd[1024];
	bool current_dir_ok = getcwd(cwd, sizeof(cwd)) != nullptr;
	std::string current_dir = current_dir_ok ? std::string(cwd) + "/" + path : "";

	if (path[0] == '/')
		current_dir = path;

	DIR *dir = opendir(current_dir.c_str());
	if (dir) {
		// add all files path in the directory to paths vector
		struct dirent *ent;
		while ((ent = readdir(dir)) != nullptr) {
			if (ent->d_type == DT_REG) {
				paths.push_back(current_dir + "/" + ent->d_name);
			}
		}
		closedir(dir);
	}
	return paths;
}

CUDARuntimeLoader::CUDARuntimeLoader(const std::vector<CUcontext> &contexts)
{
	_functionCache.resize(contexts.size());
	_perGpuModles.resize(contexts.size());
	// load kernels from the kernels folder
	std::string kernels{ConfigVariable<std::string>("devices.cuda.kernels_folder")};

	std::vector<name_data> cuObjs;
	std::vector<name_data> cuSrcs;
	std::vector<name_data> cuHeaders;


	for (auto const &dirEntry : getAllCudaUserBinariesPath(kernels)) {
		if (dirEntry.size() > 2 && dirEntry.substr(dirEntry.size() - 2) == ".c")
			continue;
		if (dirEntry.size() > 4 && dirEntry.substr(dirEntry.size() - 4) == ".cpp")
			continue;

		if ((dirEntry.size() > 4 && dirEntry.substr(dirEntry.size() - 4) == ".hpp") || (dirEntry.size() > 2 && dirEntry.substr(dirEntry.size() - 2) == ".h"))
			cuHeaders.emplace_back(dirEntry, getFileContent(dirEntry));
		else if (dirEntry.size() > 3 && dirEntry.substr(dirEntry.size() - 3) == ".cu")
			cuSrcs.emplace_back(dirEntry, getFileContent(dirEntry));
		else
			cuObjs.emplace_back(dirEntry, getFileContent(dirEntry));
	}

	// When this is emplaced after the folder elements, the kernels in the folder are overriding implemented functions
	getSectionBinaryModules(cuObjs);

	bool warningEnabled = ConfigVariable<bool>("devices.cuda.warning_on_incompatible_binary");
	for (size_t i = 0; i < contexts.size(); ++i) {
		cuCtxSetCurrent(contexts[i]);

		std::vector<std::string> cu_src_ptx_vec = compileSourcesIntoPtxForContext(cuSrcs, cuHeaders, i);
		for (std::string &cu_src_ptx : cu_src_ptx_vec) {
			CUmodule tmp;
			if (cuModuleLoadData(&tmp, &cu_src_ptx[0]) == CUDA_SUCCESS)
				_perGpuModles[i].push_back(tmp);
			else
				FatalErrorHandler::warn("Could not load compiled ptx");
		}

		for (name_data &cuObj : cuObjs) {
			CUmodule tmp;
			if (cuModuleLoadData(&tmp, &cuObj.data[0]) == CUDA_SUCCESS)
				_perGpuModles[i].push_back(tmp);
			else {
				if (cuObj.filename == CUDA_BINARY_STRING) {
					if (!warningEnabled)
						continue;
					const char *green = "\033[1;32m";
					const char *reset = "\033[0m";
					const char *red = "\033[1;31m";
					cudaDeviceProp deviceProp;
					cudaGetDeviceProperties(&deviceProp, i);
					const int capability_number = deviceProp.major * 10 + deviceProp.minor;
					fprintf(stderr, "\n");
					FatalErrorHandler::warn("The embedded fatbin couldn't be loaded for GPU: ", i);
					FatalErrorHandler::warn("This usually means that the compute capabilities of the embedded binary Doesn't match the one of the current device.");

					FatalErrorHandler::warn("To fix this you can add an implementation of the cuda functions as a ptx/fatbin to the folder: ", kernels);
					FatalErrorHandler::warn("Please, make sure that those implementations are compiled using the correct capabilities of your CUDA-Capable device.");

					fprintf(stderr, green);
					FatalErrorHandler::warn("On CLANG you probably want to add the compilation flag --cuda-gpu-arch=sm_", capability_number);
					const std::string nvcc_string = "On NVCC you probably want to add the compilation flag --generate-code arch=compute_" + std::to_string(capability_number) + ",code=sm_" + std::to_string(capability_number);
					FatalErrorHandler::warn(nvcc_string);
					fprintf(stderr, reset);

					FatalErrorHandler::warn("If you have more than one GPUs with different capabilities, try to compile the cuda code for all the needed capabilities.");
					fprintf(stderr, red);
					FatalErrorHandler::warn("If you want to silence this warning, go to the nanos6 config file and set warning_on_incompatible_binary to false");
					fprintf(stderr, reset);

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
