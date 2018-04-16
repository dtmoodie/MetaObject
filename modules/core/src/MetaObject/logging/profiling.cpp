#include "MetaObject/logging/profiling.hpp"
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/thread/ThreadRegistry.hpp"
#include <sstream>
#include <string>
#ifdef HAVE_NVTX
#include "nvToolsExt.h"
#include "nvToolsExtCuda.h"
#endif
#ifdef HAVE_CUDA
#include "cuda.h"
#ifdef HAVE_OPENCV
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif
#else
typedef struct CUstream_st *CUstream;
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

#ifdef RMT_BUILTIN
void initRemotery()
{
    /*if(rmt)
        return;
    rmt_CreateGlobalInstance(&rmt);
    CUcontext ctx;
    cuCtxGetCurrent(&ctx);
    rmtCUDABind bind;
    bind.context = ctx;
    bind.CtxSetCurrent = (void*)&cuCtxSetCurrent;
    bind.CtxGetCurrent = (void*)&cuCtxGetCurrent;
    bind.EventCreate = (void*)&cuEventCreate;
    bind.EventDestroy = (void*)&cuEventDestroy;
    bind.EventRecord = (void*)&cuEventRecord;
    bind.EventQuery = (void*)&cuEventQuery;
    bind.EventElapsedTime = (void*)&cuEventElapsedTime;
    rmt_BindCUDA(&bind);
    rmt_push_cpu = &_rmt_BeginCPUSample;
    rmt_pop_cpu = &_rmt_EndCPUSample;
    rmt_push_gpu = &_rmt_BeginCUDASample;
    rmt_pop_gpu = &_rmt_EndCUDASample;
    rmt_set_thread = &_rmt_SetCurrentThreadName;*/
}

#else
void initRemotery()
{
    if (rmt_push_cpu && rmt_pop_cpu)
        return;
#ifdef _DEBUG
    HMODULE handle = LoadLibrary("remoteryd.dll");
#else
    HMODULE handle = LoadLibrary("remotery.dll");
#endif
    if (handle)
    {
        MO_LOG(info) << "Loaded remotery library for profiling";
        typedef void (*rmt_init)(Remotery**);
        rmt_init init = (rmt_init)GetProcAddress(handle, "_rmt_CreateGlobalInstance");
        if (init)
        {
            init(&rmt);
#ifdef HAVE_CUDA
            typedef void (*rmt_cuda_init)(const rmtCUDABind*);
            rmt_cuda_init cuda_init = (rmt_cuda_init)(GetProcAddress(handle, "_rmt_BindCUDA"));
            if (cuda_init)
            {
                CUcontext ctx;
                cuCtxGetCurrent(&ctx);
                rmtCUDABind bind;
                bind.context = ctx;
                bind.CtxSetCurrent = (void*)&cuCtxSetCurrent;
                bind.CtxGetCurrent = (void*)&cuCtxGetCurrent;
                bind.EventCreate = (void*)&cuEventCreate;
                bind.EventDestroy = (void*)&cuEventDestroy;
                bind.EventRecord = (void*)&cuEventRecord;
                bind.EventQuery = (void*)&cuEventQuery;
                bind.EventElapsedTime = (void*)&cuEventElapsedTime;
                cuda_init(&bind);
            }
            rmt_push_cpu = (rmt_push_cpu_f)GetProcAddress(handle, "_rmt_BeginCPUSample");
            rmt_pop_cpu = (rmt_pop_cpu_f)GetProcAddress(handle, "_rmt_EndCPUSample");
            rmt_push_gpu = (rmt_push_cuda_f)GetProcAddress(handle, "_rmt_BeginCUDASample");
            rmt_pop_gpu = (rmt_pop_cuda_f)GetProcAddress(handle, "_rmt_EndCUDASample");
            rmt_set_thread = (rmt_set_thread_name_f)GetProcAddress(handle, "_rmt_SetCurrentThreadName");
#endif
        }
    }
    else
    {
        MO_LOG(info) << "No remotery library found";
    }
}
#endif

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
        MO_LOG(info) << "Loaded nvtx module";
        nvtx_push = (push_f)dlsym(nvtx_handle, "nvtxRangePushA");
        nvtx_pop = (pop_f)dlsym(nvtx_handle, "nvtxRangePop");
        nvtx_name_thread = (nvtx_name_thread_f)dlsym(nvtx_handle, "nvtxNameOsThreadA");
        nvtx_name_stream = (nvtx_name_stream_f)dlsym(nvtx_handle, "nvtxNameCuStreamA");
    }
    else
    {
        MO_LOG(info) << "No nvtx library loaded";
    }
#endif
}

void mo::initProfiling()
{
#ifndef PROFILING_NONE
    initNvtx();
    initRemotery();
#endif
}
void mo::pushCpu(const char* name, unsigned int* rmt_hash)
{
    if (nvtx_push)
        (*nvtx_push)(name);
    /*if (rmt && rmt_push_cpu)
    {
        rmt_push_cpu(name, rmt_hash);
    }*/
}

void mo::popCpu()
{
    if (nvtx_pop)
    {
        (*nvtx_pop)();
    }
    /*if (rmt && rmt_pop_cpu)
    {
        rmt_pop_cpu();
    }*/
}
void mo::setStreamName(const char* name, const cudaStream_t stream)
{
    if (nvtx_name_stream)
    {
        nvtx_name_stream(stream, name);
    }
}

scoped_profile::scoped_profile(std::string name, unsigned int* obj_hash, unsigned int* cuda_hash, cudaStream_t stream)
    : scoped_profile(name.c_str(), obj_hash, cuda_hash, stream)
{
}

scoped_profile::scoped_profile(const char* name, unsigned int* rmt_hash, unsigned int* rmt_cuda, cudaStream_t stream)
{
#ifndef PROFILING_NONE
    if (nvtx_push)
        (*nvtx_push)(name);
/*if (rmt && rmt_push_cpu)
    {
        rmt_push_cpu(name, rmt_hash);
    }*/
#endif
}

scoped_profile::scoped_profile(
    const char* name, const char* func, unsigned int* rmt_hash, unsigned int* rmt_cuda, cudaStream_t stream)
{
#ifndef PROFILING_NONE
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
/*if (rmt && rmt_push_cpu)
    {
        rmt_push_cpu(str, rmt_hash);
        if(stream && rmt_push_gpu)
        {
            rmt_push_gpu(str, rmt_cuda, cv::cuda::StreamAccessor::getStream(*stream));
            this->stream = stream;
        }
    }*/
#endif
}

scoped_profile::~scoped_profile()
{
#ifndef PROFILING_NONE
    if (nvtx_pop)
    {
        (*nvtx_pop)();
    }
    /*if (rmt && rmt_pop_cpu)
    {
        rmt_pop_cpu();
    }*/
    if (stream && rmt_pop_gpu)
    {
        rmt_pop_gpu(stream);
    }
#endif
}
