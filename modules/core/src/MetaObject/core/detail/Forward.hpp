#pragma once
#include "Enums.hpp"
#include "Time.hpp"
#include <vector>
namespace std
{
    template <class T>
    class shared_ptr;
}
// Forward declarations and typedefs
namespace boost
{
    class recursive_timed_mutex;
    template <typename T>
    class unique_lock;
}

namespace mo
{
    class Context;
    class RelayManager;
    class ISignal;
    class ICallback;
    class ISlot;
    template <class T>
    class TSlot;
    class Connection;
    class TypeInfo;
    class IVariableManager;
    class IMetaObjectInfo;
    class IMetaObject;
    class MetaObject;
    class ISignalRelay;
    class TypeInfo;
    class IParam;
    class ICoordinateSystem;
    struct IDataContainer;

    class IParam;
    class InputParam;
    template <class T, class Enable = void>
    class ITParam;
    template <class T>
    struct ITInputParam;
    struct Header;

    struct ParamInfo;
    struct SignalInfo;
    struct SlotInfo;
    struct CallbackInfo;

    using ParamVec_t = std::vector<IParam*>;
    using ParamInfoVec_t = std::vector<ParamInfo*>;
    using SignalInfoVec_t = std::vector<SignalInfo*>;
    using SlotInfoVec_t = std::vector<SlotInfo*>;
    using InputParamVec_t = std::vector<InputParam*>;
    using Mutex_t = boost::recursive_timed_mutex;
    using Lock = boost::unique_lock<mo::Mutex_t>;
    using ContextPtr_t = std::shared_ptr<Context>;
    using CoordinateSystemPtr_t = std::shared_ptr<ICoordinateSystem>;
    using ConnectionPtr_t = std::shared_ptr<Connection>;
    using IParamPtr_t = std::shared_ptr<IParam>;

    template <class T>
    class TSignal;

    template <class T>
    class TSlot;

    template <class T>
    class TSignalRelay;

    using Update_s = void(IParam*, Header, UpdateFlags);
    using DataUpdate_s = void(const std::shared_ptr<IDataContainer>&, IParam*, UpdateFlags);
    using UpdateSlot_t = TSlot<Update_s>;
    using IDataContainerPtr_t = std::shared_ptr<IDataContainer>;
    using IDataContainerConstPtr_t = std::shared_ptr<const IDataContainer>;
}
