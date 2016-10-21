#include "MetaObject/Logging/Profiling.hpp"
#include "MetaObject/Logging/Log.hpp"
#include <string>
#include <sstream>
#define RMT_USE_CUDA
#include "dependencies/Remotery/lib/Remotery.h"

#ifdef HAVE_CUDA
#include "cuda.h"
#ifdef HAVE_OPENCV
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif
#endif
#if WIN32
#include "Windows.h"
#else
#include "dlfcn.h"
#endif

using namespace mo;

typedef int(*push_f)(const char*);
typedef int(*pop_f)();

typedef void(*rmt_push_cpu_f)(const char*, unsigned int*);
typedef void(*rmt_pop_cpu_f)();
typedef void(*rmt_push_cuda_f)(const char*, unsigned int*, void*);
typedef void(*rmt_pop_cuda_f)(void*);
typedef void(*rmt_set_thread_name_f)(const char*);

push_f nvtx_push = NULL;
pop_f nvtx_pop = NULL;

Remotery* rmt = nullptr;
rmt_push_cpu_f rmt_push_cpu = nullptr;
rmt_pop_cpu_f rmt_pop_cpu = nullptr;
rmt_push_cuda_f rmt_push_gpu = nullptr;
rmt_pop_cuda_f rmt_pop_gpu = nullptr;
rmt_set_thread_name_f rmt_set_thread = nullptr;

void mo::SetThreadName(const char* name)
{
    if(rmt_set_thread)
    {
        rmt_set_thread(name);
    }
}
void mo::InitProfiling()
{
#if WIN32
	
	HMODULE nvtx_handle = LoadLibrary("nvToolsExt64_1.dll");
	if (nvtx_handle)
	{
        LOG(info) << "Loaded nvtx module";
		nvtx_push = (push_f)GetProcAddress(nvtx_handle, "nvtxRangePushA");
		nvtx_pop = (pop_f)GetProcAddress(nvtx_handle, "nvtxRangePop");
	}else
    {
        LOG(info) << "No nvtx library loaded";
    }
	
#else


#endif
#ifdef _DEBUG
    HMODULE handle = LoadLibrary("remoteryd.dll");
#else
    HMODULE handle = LoadLibrary("remotery.dll");
#endif
    if(handle)
    {
        LOG(info) << "Loaded remotery library for profiling";
        typedef void(*rmt_init)(Remotery**);    
        rmt_init init = (rmt_init)GetProcAddress(handle, "_rmt_CreateGlobalInstance");
        if(init)
        {
            init(&rmt);
#ifdef HAVE_CUDA
            typedef void(*rmt_cuda_init)(const rmtCUDABind*);
            rmt_cuda_init cuda_init = (rmt_cuda_init)(GetProcAddress(handle, "_rmt_BindCUDA"));
            if(cuda_init)
            {
                CUcontext ctx;
                cuCtxGetCurrent(&ctx);
                rmtCUDABind bind;
                bind.context = ctx;
                bind.CtxSetCurrent = &cuCtxSetCurrent;
                bind.CtxGetCurrent = &cuCtxGetCurrent;
                bind.EventCreate = &cuEventCreate;
                bind.EventDestroy = &cuEventDestroy;
                bind.EventRecord = &cuEventRecord;
                bind.EventQuery = &cuEventQuery;
                bind.EventElapsedTime = &cuEventElapsedTime;
                cuda_init(&bind);
            }
            rmt_push_cpu = (rmt_push_cpu_f)GetProcAddress(handle, "_rmt_BeginCPUSample");
            rmt_pop_cpu = (rmt_pop_cpu_f)GetProcAddress(handle, "_rmt_EndCPUSample");
            rmt_push_gpu = (rmt_push_cuda_f)GetProcAddress(handle, "_rmt_BeginCUDASample");
            rmt_pop_gpu = (rmt_pop_cuda_f)GetProcAddress(handle, "_rmt_EndCUDASample");
            rmt_set_thread = (rmt_set_thread_name_f)GetProcAddress(handle, "_rmt_SetCurrentThreadName");
#endif
        }    
	}else
    {
        LOG(info) << "No remotery library found";
    }
}



scoped_profile::scoped_profile(const char* name, unsigned int* rmt_hash, unsigned int* rmt_cuda, cv::cuda::Stream* stream)
{
    if(nvtx_push)
        (*nvtx_push)(name);
	if (rmt && rmt_push_cpu)
	{
		if(rmt_hash)
            rmt_push_cpu(name, rmt_hash);
		else
            rmt_push_cpu(name, nullptr);
	}
}

scoped_profile::scoped_profile(const char* name, const char* func, unsigned int* rmt_hash, unsigned int* rmt_cuda, cv::cuda::Stream* stream)
{
	std::stringstream ss;
	ss << name;
	ss << "[";
	ss << func;
	ss << "]";
    const char* str = ss.str().c_str();
    if(nvtx_push)
    {
        (*nvtx_push)(str);
    }
	if (rmt && rmt_push_cpu)
	{
        rmt_push_cpu(str, rmt_hash);
        if(stream && rmt_push_gpu)
        {
            rmt_push_gpu(str, rmt_cuda, cv::cuda::StreamAccessor::getStream(*stream));
            this->stream = stream;
        }
	}
}

scoped_profile::~scoped_profile()
{
    if(nvtx_pop)
    {
        (*nvtx_pop)();
    }
	if (rmt && rmt_pop_cpu)
	{
        rmt_pop_cpu();
	}
    if(stream && rmt_pop_gpu)
    {
        rmt_pop_gpu(cv::cuda::StreamAccessor::getStream(*stream));
    }
}
