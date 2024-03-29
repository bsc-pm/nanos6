/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_RUNTIME_LOADER_HPP
#define CUDA_RUNTIME_LOADER_HPP

#include <config.h>

#include <cuda.h>

#include <cassert>
#include <iostream>
#include <unistd.h>
#include <unordered_map>
#include <vector>

/*
 * This file interprets and parses CUDA binaries to load kernels
 * For PTX that is incrusted in the binary, there is a very small amount of information available:
 * 	- https://zenodo.org/record/2339027/files/decoding-cuda-binary-file-format.pdf
 *  - And whatever the LLVM implementation does to build the binaries
 */

class CUDARuntimeLoader {
	static const uint32_t NV_FATBIN_DESC_MAGIC = 0xba55ed50; // 50ed55ba
	static const uint32_t NV_FATBIN_SEGMENT_MAGIC = 0x466243b1; // 50ed55ba

	struct nvFatbinDetailed {
		uint16_t deviceType; // 2 FOR GPU
		uint16_t embeddedType;
		uint32_t offsetFromHeader;
		uint64_t sizeEmbeddedFile;
		uint64_t zeros;
		uint32_t version;
		uint32_t arch; // 20, 35 etc...
	};

	struct nvFatbinDesc {
		uint32_t magic;
		uint32_t version;
		uint64_t size;
		// nvFatbinDetailed detailed_header;
	};

	struct nvFatbinSegment {
		uint32_t magic;
		uint32_t version;
		uint64_t fatbinAddress;
		uint8_t _dontCare[8];
	};

	struct nameData {
		std::string filename;
		std::string data;
		nameData(const std::string &_filename, const std::string &_data) :
			filename(_filename), data(_data)
		{
		}
	};

	static void getSectionBinaryModules(std::vector<nameData> &cu_objs);

	static std::vector<std::string>
	compileSourcesIntoPtxForContext(const std::vector<nameData> &sourcefiles,
		const std::vector<nameData> &headers,
		int gpu);

#if defined(USE_CUDA_CL)
	static std::vector<std::string>
	compileAllClKernels(const std::vector<nameData> &sourcefiles);

#endif

	static std::vector<std::string>
	getAllCudaUserBinariesPath(const std::string &path);

	static std::string getFileContent(const std::string &filename);

public:
	CUfunction loadFunction(const char *str);

	CUDARuntimeLoader();

private:
	std::vector<std::vector<CUmodule>> _perGpuModules;
	std::vector<std::unordered_map<const char *, CUfunction>> _functionCache;
};

#endif
