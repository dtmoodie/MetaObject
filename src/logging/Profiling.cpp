#include "MetaObject/Logging/Profiling.hpp"
#include <string>
#include <sstream>
#include "dependencies/Remotery/lib/Remotery.h"
#ifdef HAVE_CUDA
#include "cuda.h"
#endif
#if WIN32
#include "Windows.h"
#else
#include "dlfcn.h"
#endif

using namespace mo;

typedef int(*push_f)(const char*);
typedef int(*pop_f)();


push_f nvtx_push = NULL;
pop_f nvtx_pop = NULL;
Remotery* rmt = nullptr;

void EagleLib::InitProfiling(bool use_nvtx, bool use_remotery)
{
#if WIN32
	if (use_nvtx)
	{
		HMODULE handle = LoadLibrary("nvToolsExt64_1.lib");
		if (handle)
		{
			nvtx_push = (push_f)GetProcAddress(handle, "nvtxRangePushA");
			nvtx_pop = (pop_f)GetProcAddress(handle, "nvtxRangePop");
		}
	}
#else


#endif
	if (use_remotery)
	{
		rmt_CreateGlobalInstance(&rmt);
#ifdef HAVE_CUDA
		rmtCUDABind bind;
		bind.context = m_Context;
		bind.CtxSetCurrent = &cuCtxSetCurrent;
		bind.CtxGetCurrent = &cuCtxGetCurrent;
		bind.EventCreate = &cuEventCreate;
		bind.EventDestroy = &cuEventDestroy;
		bind.EventRecord = &cuEventRecord;
		bind.EventQuery = &cuEventQuery;
		bind.EventElapsedTime = &cuEventElapsedTime;
		rmt_BindCUDA(&bind);
#endif
	}
}



scoped_profile::scoped_profile(const char* name, unsigned int* rmt_hash, unsigned int* rmt_cuda)
{
    if(nvtx_push)
        (*nvtx_push)(name);
	if (rmt)
	{
		if(rmt_hash)
			_rmt_BeginCPUSample(name, rmt_hash);
		else
			_rmt_BeginCPUSample(name, nullptr);
	}
}

scoped_profile::scoped_profile(const char* name, const char* func, unsigned int* rmt_hash, unsigned int* rmt_cuda)
{
	std::stringstream ss;
	ss << name;
	ss << "[";
	ss << func;
	ss << "]";
    if(nvtx_push)
    {
        (*nvtx_push)(ss.str().c_str());
    }
	if (rmt)
	{
		if (rmt_hash)
			_rmt_BeginCPUSample(ss.str().c_str(), rmt_hash);
		else
			_rmt_BeginCPUSample(ss.str().c_str(), nullptr);
	}
}

scoped_profile::~scoped_profile()
{
    if(nvtx_pop)
    {
        (*nvtx_pop)();
    }
	if (rmt)
	{
		rmt_EndCPUSample();
	}
}
