#pragma once
#include "Enums.hpp"
#include "Time.hpp"

namespace std
{
    template <class T>
    class shared_ptr;

    template <class T, class Alloc>
    class vector;
}

// Forward declarations and typedefs
namespace boost
{
    template <typename T>
    class unique_lock;

    namespace fibers
    {
        class mutex;
        class recursive_timed_mutex;
    }
}

namespace mo
{
    struct Allocator;
    struct DeviceAllocator;
    class ISignal;
    class ICallback;
    class ISlot;
    class IMetaObjectInfo;
    class IMetaObject;
    class ISignalRelay;

    struct IAsyncStream;
    struct AsyncStream;
    namespace cuda
    {
        struct AsyncStream;
        struct CvAsyncStream;
    }

    class Thread;

    class IParam;
    struct ICoordinateSystem;
    struct IDataContainer;

    template <class T>
    struct ITInputParam;

    class RelayManager;
    template <class T>
    class TSlot;
    class Connection;
    class TypeInfo;

    class MetaObject;

    class TypeInfo;

    class InputParam;
    template <class T, class Enable = void>
    class TParam;

    struct Header;

    struct ParamInfo;
    struct SignalInfo;
    struct SlotInfo;
    struct CallbackInfo;

    template <class T>
    class TSignal;
    template <class T>
    class TSlot;
    template <class T, class Mutex = boost::fibers::mutex>
    class TSignalRelay;

    using ParamVec_t = std::vector<IParam*>;
    using ParamInfoVec_t = std::vector<ParamInfo*>;
    using SignalInfoVec_t = std::vector<SignalInfo*>;
    using SlotInfoVec_t = std::vector<SlotInfo*>;
    using InputParamVec_t = std::vector<InputParam*>;
    using Mutex_t = boost::fibers::recursive_timed_mutex;
    using Lock_t = boost::unique_lock<Mutex_t>;

    using IAsyncStreamPtr_t = std::shared_ptr<IAsyncStream>;
    using ICoordinateSystemPtr_t = std::shared_ptr<ICoordinateSystem>;
    using ConnectionPtr_t = std::shared_ptr<Connection>;
    using IParamPtr_t = std::shared_ptr<IParam>;

    using Update_s = void(IParam*, Header, UpdateFlags);
    using DataUpdate_s = void(const std::shared_ptr<IDataContainer>&, IParam*, UpdateFlags);
    using UpdateSlot_t = TSlot<Update_s>;
    using IDataContainerPtr_t = std::shared_ptr<IDataContainer>;
    using IDataContainerConstPtr_t = std::shared_ptr<const IDataContainer>;
    using AllocatorPtr_t = std::shared_ptr<Allocator>;
}
