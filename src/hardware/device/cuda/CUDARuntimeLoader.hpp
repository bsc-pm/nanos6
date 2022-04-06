/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_RUNTIME_LOADER_HPP
#define CUDA_RUNTIME_LOADER_HPP

#include <cuda.h>

#include <iostream>
#include <unistd.h>
#include <cassert>
#include <vector>
#include <unordered_map>

class CUDARuntimeLoader
{
    static const uint32_t NV_FATBIN_DESC_MAGIC = 0xba55ed50; // 50ed55ba     

    struct nv_fatbin_detailed
    {
        uint16_t device_type; //2 FOR GPU
        uint16_t embedded_type;
        uint32_t offset_from_header;
        uint64_t size_embedded_file;
        uint64_t zeros;
        uint32_t version;
        uint32_t arch;// 20, 35 etc...
    };

    struct nv_fatbin_desc
    {
        uint32_t magic;
        uint32_t version;
        uint64_t size;
        nv_fatbin_detailed detailed_header;
    };

    struct nv_cuda_fatbin_desc { 
        unsigned int           type;//0 = elf, 1 = ptx
        unsigned int           binary_offset;
        unsigned long long int size;
        //... missing parameters
    };

    struct name_data
    {
        std::string filename;
        std::string data;
        name_data(const std::string& _filename, const std::string& _data) : filename(_filename), data(_data) {}
    };


    static void getSectionBinaryModules(std::vector<name_data>& cu_objs);

    static std::vector<std::string> compileSourcesIntoPtxForContext(const std::vector<name_data>& sourcefiles,const std::vector<name_data>& headers, int gpu);

    static std::vector<std::string> getAllCudaUserBinariesPath(const std::string& path);
    
    static std::string getFileContent(const std::string& filename);

    public:

    CUfunction loadFunction(const char* str);

    CUDARuntimeLoader(const std::vector<CUcontext>& contexts);

    private:

    std::vector<std::vector<CUmodule>> _per_gpu_modules;
    std::vector<std::unordered_map<const char*, CUfunction>> _function_cache;
};

#endif