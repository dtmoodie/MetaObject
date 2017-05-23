#pragma once
#include "MetaObject/detail/Export.hpp"
#include <string>
typedef struct CUstream_st *cudaStream_t;
namespace cv {
namespace cuda {
class Stream;
}
}

namespace mo {
MO_EXPORTS void setThreadName(const char* name);
class IMetaObject;
MO_EXPORTS void initProfiling();
MO_EXPORTS void pushCpu(const char* name, unsigned int* rmt_hash = nullptr);
MO_EXPORTS void popCpu();
MO_EXPORTS void setStreamName(const char* name, const cudaStream_t stream);
struct MO_EXPORTS scoped_profile {
    scoped_profile(std::string name, unsigned int* obj_hash = nullptr, unsigned int* cuda_hash = nullptr, cudaStream_t stream = nullptr);
    scoped_profile(const char* name, unsigned int* obj_hash = nullptr, unsigned int* cuda_hash = nullptr, cudaStream_t stream = nullptr);
    scoped_profile(const char* name, const char* func, unsigned int* obj_hash = nullptr, unsigned int* cuda_hash = nullptr, cudaStream_t stream = nullptr);
    ~scoped_profile();
private:
    cudaStream_t* stream = nullptr;
};
}


#define PROFILE_OBJ(name) \
mo::scoped_profile profile_object(name, __FUNCTION__, &_rmt_hash, &_rmt_cuda_hash, _ctx->stream)

#define PROFILE_RANGE(name) \
mo::scoped_profile profile_scope_##name(#name)

#define PROFILE_FUNCTION \
mo::scoped_profile profile_function(__FUNCTION__);
