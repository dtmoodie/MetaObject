#pragma once

#include "MetaObject/core/detail/Counter.hpp"
#include "MetaObject/object/MetaObject.hpp"
#include "MetaObject/object/detail/MetaObjectMacros.hpp"
#include "MetaObject/params/ParamMacros.hpp"
#include "MetaObject/signals/TSignal.hpp"
#include "MetaObject/signals/detail/SignalMacros.hpp"
#include "MetaObject/signals/detail/SlotMacros.hpp"

#include "MetaObject/params/TInputParam.hpp"
#include "MetaObject/params/TParamPtr.hpp"

#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"
RUNTIME_MODIFIABLE_INCLUDE

#ifdef HAVE_CUDA
RUNTIME_COMPILER_SOURCEDEPENDENCY_EXT(".cu")
#endif

#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined(test_recompile_object_EXPORTS)
#define DLL_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#define DLL_EXPORTS  __attribute__((visibility("default")))
#else
#define DLL_EXPORTS 
#endif

using namespace mo;
struct DLL_EXPORTS test_meta_object_signals : public MetaObject
{
    ~test_meta_object_signals() { std::cout << "Deleting object\n"; }
    MO_BEGIN(test_meta_object_signals)
    MO_SIGNAL(void, test_void)
    MO_SIGNAL(void, test_int, int)
    MO_END
};

struct DLL_EXPORTS test_meta_object_slots : public MetaObject
{
    MO_BEGIN(test_meta_object_slots)
    MO_SLOT(void, test_void)
    MO_SLOT(void, test_int, int)
    STATE(int, call_count, 0)
    MO_END
};

struct DLL_EXPORTS test_meta_object_parameters : public MetaObject
{
    MO_BEGIN(test_meta_object_parameters)
    PARAM(int, test, 5)
    MO_END
};

struct DLL_EXPORTS test_meta_object_output : public MetaObject
{
    MO_BEGIN(test_meta_object_output)
    OUTPUT(int, test_output, 0)
    MO_END
    int param_update_call_count = 0;
    virtual void onParamUpdate(
        IParam*, Context*, OptionalTime, size_t, const std::shared_ptr<ICoordinateSystem>&, UpdateFlags) override
    {
        ++param_update_call_count;
    }
};

struct DLL_EXPORTS test_meta_object_input : public MetaObject
{
    MO_BEGIN(test_meta_object_input)
        INPUT(int, test_input, nullptr)
    MO_END
    int param_update_call_count = 0;
    virtual void onParamUpdate(
        IParam*, Context*, OptionalTime, size_t, const std::shared_ptr<ICoordinateSystem>&, UpdateFlags) override
    {
        ++param_update_call_count;
    }
};

#ifdef HAVE_CUDA
struct DLL_EXPORTS test_cuda_object : public MetaObject
{
    MO_BEGIN(test_cuda_object)
    PARAM(int, test, 0)
    MO_END
    void run_kernel();
};
#endif
