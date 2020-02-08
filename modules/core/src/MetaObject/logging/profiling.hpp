#pragma once
#include "MetaObject/detail/Export.hpp"
#include <string>
typedef struct CUstream_st* cudaStream_t;

namespace mo
{

    class IMetaObject;
    MO_EXPORTS void setThisThreadName(const char* name);

    MO_EXPORTS void initProfiling();
    MO_EXPORTS void pushCpu(const char* name);
    MO_EXPORTS void popCpu();
    MO_EXPORTS void setStreamName(const char* name, const cudaStream_t stream);

    struct MO_EXPORTS ScopedProfile
    {
        ScopedProfile(const std::string& name);

        ScopedProfile(const char* name);

        ScopedProfile(const char* name, const char* func);

        ~ScopedProfile();
    };
} // namespace mo

#define PROFILE_OBJ(name) mo::ScopedProfile profile_object(name, __FUNCTION__)

#define PROFILE_RANGE(name) mo::ScopedProfile profile_scope_##name(#name)

#define PROFILE_FUNCTION mo::ScopedProfile profile_function(__FUNCTION__);
