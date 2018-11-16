#include "MetaObject/logging/profiling.hpp"
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/thread/ThreadRegistry.hpp"
#include <sstream>
#include <string>

#ifdef HAVE_CUDA
#include "cuda.h"
#ifdef HAVE_OPENCV
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif
#else
typedef struct CUstream_st* CUstream;
#endif
#if WIN32
#include "Windows.h"
#else
#include "dlfcn.h"
#endif

using namespace mo;

typedef int (*push_f)(const char*);
typedef int (*pop_f)();
typedef void (*nvtx_name_thread_f)(uint32_t, const char*);
typedef void (*nvtx_name_stream_f)(CUstream, const char*);

typedef void (*rmt_push_cpu_f)(const char*, unsigned int*);
typedef void (*rmt_pop_cpu_f)();
typedef void (*rmt_push_cuda_f)(const char*, unsigned int*, void*);
typedef void (*rmt_pop_cuda_f)(void*);
typedef void (*rmt_set_thread_name_f)(const char*);

#ifndef PROFILING_NONE
push_f nvtx_push = nullptr;
pop_f nvtx_pop = nullptr;
nvtx_name_thread_f nvtx_name_thread = nullptr;
nvtx_name_stream_f nvtx_name_stream = nullptr;

// Remotery* rmt = nullptr;

rmt_push_cpu_f rmt_push_cpu = nullptr;
rmt_pop_cpu_f rmt_pop_cpu = nullptr;
rmt_push_cuda_f rmt_push_gpu = nullptr;
rmt_pop_cuda_f rmt_pop_gpu = nullptr;
rmt_set_thread_name_f rmt_set_thread = nullptr;
#endif

void mo::setThreadName(const char* name)
{
    if (rmt_set_thread)
    {
        rmt_set_thread(name);
    }
    if (nvtx_name_thread)
    {
        nvtx_name_thread(static_cast<uint32_t>(mo::getThisThread()), name);
    }
}

void initNvtx()
{
    if (nvtx_push && nvtx_pop)
        return;
#ifdef _MSC_VER
    HMODULE nvtx_handle = LoadLibrary("nvToolsExt64_1.dll");
    if (nvtx_handle)
    {
        MO_LOG(info) << "Loaded nvtx module";
        nvtx_push = (push_f)GetProcAddress(nvtx_handle, "nvtxRangePushA");
        nvtx_pop = (pop_f)GetProcAddress(nvtx_handle, "nvtxRangePop");
    }
    else
    {
        MO_LOG(info) << "No nvtx library loaded";
    }
#else
    void* nvtx_handle = dlopen("libnvToolsExt.so", RTLD_NOW);
    if (nvtx_handle)
    {
        getDefaultLogger().info("Loaded nvtx profiling module");
        nvtx_push = (push_f)dlsym(nvtx_handle, "nvtxRangePushA");
        nvtx_pop = (pop_f)dlsym(nvtx_handle, "nvtxRangePop");
        nvtx_name_thread = (nvtx_name_thread_f)dlsym(nvtx_handle, "nvtxNameOsThreadA");
        nvtx_name_stream = (nvtx_name_stream_f)dlsym(nvtx_handle, "nvtxNameCuStreamA");
    }
    else
    {
        getDefaultLogger().info("no nvtx profiling module loaded");
    }
#endif
}

void mo::initProfiling()
{
    initNvtx();
}

void mo::pushCpu(const char* name)
{
    if (nvtx_push)
    {
        (*nvtx_push)(name);
    }
}

void mo::popCpu()
{
    if (nvtx_pop)
    {
        (*nvtx_pop)();
    }
}

void mo::setStreamName(const char* name, const cudaStream_t stream)
{
    if (nvtx_name_stream)
    {
        nvtx_name_stream(stream, name);
    }
}

scoped_profile::scoped_profile(const std::string& name)
    : scoped_profile(name.c_str())
{
}

scoped_profile::scoped_profile(const char* name)
{
    if (nvtx_push)
    {
        (*nvtx_push)(name);
    }
}

scoped_profile::scoped_profile(const char* name, const char* func)
{
    std::stringstream ss;
    ss << name;
    ss << "[";
    ss << func;
    ss << "]";
    const char* str = ss.str().c_str();
    if (nvtx_push)
    {
        (*nvtx_push)(str);
    }
}

scoped_profile::~scoped_profile()
{
    if (nvtx_pop)
    {
        (*nvtx_pop)();
    }
}
