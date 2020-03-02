#ifndef MO_CORE_FORWARD_HPP
#define MO_CORE_FORWARD_HPP
#include "Enums.hpp"
#include "Time.hpp"

namespace std
{
    template <class T>
    class shared_ptr;

    template <class T, class Alloc>
    class vector;

    template <class T>
    class unique_lock;
} // namespace std

namespace cereal
{
    class BinaryInputArchive;
    class BinaryOutputArchive;
} // namespace cereal

namespace mo
{
    struct Allocator;
    struct DeviceAllocator;
    class ISignal;
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
    } // namespace cuda

    class Thread;
    class IParam;
    struct IDataContainer;
    class RelayManager;
    class Connection;
    class TypeInfo;
    class MetaObject;
    struct Header;

    struct ParamInfo;
    struct SignalInfo;
    struct SlotInfo;
    struct CallbackInfo;
    struct TimedMutex;

    template <class T, class E = void>
    class TSignal;

    template <class T>
    class TSlot;

    template <class T>
    class TParam;

    template <class T, class ENABLE = void>
    struct TDataContainer;

    using ParamVec_t = std::vector<IParam*>;
    using ParamInfoVec_t = std::vector<ParamInfo*>;
    using SignalInfoVec_t = std::vector<SignalInfo*>;
    using SlotInfoVec_t = std::vector<SlotInfo*>;
    using Mutex_t = TimedMutex;
    using Lock_t = std::unique_lock<Mutex_t>;

    using IAsyncStreamPtr_t = std::shared_ptr<IAsyncStream>;

    using ConnectionPtr_t = std::shared_ptr<Connection>;
    using IParamPtr_t = std::shared_ptr<IParam>;

    using Update_s = void(IParam*, Header, UpdateFlags, IAsyncStream&);
    using DataUpdate_s = void(const std::shared_ptr<IDataContainer>&, IParam*, UpdateFlags, IAsyncStream&);
    template <class T>
    using TDataUpdate_s = void(const std::shared_ptr<TDataContainer<T>>&, IParam*, UpdateFlags, IAsyncStream&);
    using UpdateSlot_t = TSlot<Update_s>;
    template <class T>
    using TUpdateSlot_t = TSlot<TDataUpdate_s<T>>;
    using IDataContainerPtr_t = std::shared_ptr<IDataContainer>;
    using IDataContainerConstPtr_t = std::shared_ptr<const IDataContainer>;
    using AllocatorPtr_t = std::shared_ptr<Allocator>;

    using BinaryInputVisitor = cereal::BinaryInputArchive;
    using BinaryOutputVisitor = cereal::BinaryOutputArchive;

    template <class T, class Mutex = Mutex_t>
    class TSignalRelay;
} // namespace mo
#endif // MO_CORE_FORWARD_HPP