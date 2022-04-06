/*
        This file is part of Nanos6 and is licensed under the terms contained in
   the COPYING file.

        Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_RUNTIME_LOADER_HPP
#define CUDA_RUNTIME_LOADER_HPP

#include <cuda.h>

#include <cassert>
#include <iostream>
#include <unistd.h>
#include <unordered_map>
#include <vector>

class CUDARuntimeLoader {
  static const uint32_t NV_FATBIN_DESC_MAGIC = 0xba55ed50; // 50ed55ba

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
    nvFatbinDetailed detailed_header;
  };

  struct nvCudaFatbinDesc {
    unsigned int type; // 0 = elf, 1 = ptx
    unsigned int binaryOffset;
    unsigned long long int size;
    //... missing parameters
  };

  struct nameData {
    std::string filename;
    std::string data;
    nameData(const std::string &_filename, const std::string &_data)
        : filename(_filename), data(_data) {}
  };

  static void getSectionBinaryModules(std::vector<nameData> &cu_objs);

  static std::vector<std::string>
  compileSourcesIntoPtxForContext(const std::vector<nameData> &sourcefiles,
                                  const std::vector<nameData> &headers,
                                  int gpu);

  static std::vector<std::string>
  getAllCudaUserBinariesPath(const std::string &path);

  static std::string getFileContent(const std::string &filename);

public:
  CUfunction loadFunction(const char *str);

  CUDARuntimeLoader(const std::vector<CUcontext> &contexts);

private:
  std::vector<std::vector<CUmodule>> _perGpuModles;
  std::vector<std::unordered_map<const char *, CUfunction>> _functionCache;
};

#endif