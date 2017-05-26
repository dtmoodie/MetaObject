#pragma once
#include <vector>
namespace std{
    template<class T> class shared_ptr;
}
// Forward declarations and typedefs
namespace boost {
class recursive_timed_mutex;
}

namespace mo {
class Context;
class RelayManager;
class ISignal;
class ICallback;
class ISlot;
template<class T> class TSlot;
class Connection;
class TypeInfo;
class IVariableManager;
class IMetaObjectInfo;
class ISignalRelay;
class TypeInfo;
class IParam;
class ICoordinateSystem;

class IParam;
class InputParam;
template<class T> class ITParam;
template<class T> class ITInputParam;

struct ParamInfo;
struct SignalInfo;
struct SlotInfo;
struct CallbackInfo;

typedef std::vector<IParam*> ParamVec_t;
typedef std::vector<ParamInfo*> ParamInfoVec_t;
typedef std::vector<SignalInfo*> SignalInfoVec_t;
typedef std::vector<SlotInfo*> SlotInfoVec_t;
typedef boost::recursive_timed_mutex Mutex_t;
typedef std::shared_ptr<Context> ContextPtr_t;

template<class T> class TSignal;
template<class T> class TSlot;
template<class T> class TSignalRelay;
}