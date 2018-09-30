#pragma once
#include "MetaObject/detail/Export.hpp"
#include <string>
typedef struct CUstream_st* cudaStream_t;


namespace mo
{

class IMetaObject;
MO_EXPORTS void setThreadName(const char* name);

MO_EXPORTS void initProfiling();
MO_EXPORTS void pushCpu(const char* name);
MO_EXPORTS void popCpu();
MO_EXPORTS void setStreamName(const char* name, const cudaStream_t stream);

struct MO_EXPORTS scoped_profile
{
    scoped_profile(const std::string& name);

    scoped_profile(const char* name);

    scoped_profile(const char* name,
                   const char* func);

    ~scoped_profile();
};

}

#define PROFILE_OBJ(name) mo::scoped_profile profile_object(name, __FUNCTION__)

#define PROFILE_RANGE(name) mo::scoped_profile profile_scope_##name(#name)

#define PROFILE_FUNCTION mo::scoped_profile profile_function(__FUNCTION__);
