#pragma once

#include "InputParam.hpp"
#include "ITParam.hpp"
#ifdef _MSC_VER
#pragma warning( disable : 4250)
#endif

namespace mo {
template<class T>
class ITInputParam: virtual public ITParam<T>, virtual public InputParam {
public:
    ITInputParam(const std::string& name = "",  Context* ctx = nullptr);
    ~ITInputParam();
    bool setInput(std::shared_ptr<IParam> input);
    bool setInput(IParam* input);

    virtual bool acceptsInput(IParam* param) const;
    virtual bool acceptsType(const TypeInfo& type) const;

    IParam*        getInputParam();
    OptionalTime_t getInputTimestamp();
    size_t         getInputFrameNumber();

    virtual bool   isInputSet() const;
    
    virtual bool getData(Storage_t& data, const OptionalTime_t& ts = OptionalTime_t(),
                 Context* ctx = nullptr, size_t* fn_ = nullptr);

    virtual bool getData(Storage_t& data, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr);
protected:
    bool updateDataImpl(const T& data, OptionalTime_t ts, Context* ctx, boost::optional<size_t> fn, ICoordinateSystem* cs) { return true; }

    virtual void onInputDelete(IParam const* param);
    virtual void onInputUpdate(ConstStorageRef_t, IParam*, Context*, OptionalTime_t, size_t, ICoordinateSystem*, UpdateFlags);

    std::shared_ptr<ITParam<T>> _shared_input;
    ITParam<T>*                 _input;
private:
    UpdateSlot_t               _update_slot;
    TSlot<void(IParam const*)> _delete_slot;
    bool setInputImpl(IParam* param);
};
}
#include "detail/ITInputParamImpl.hpp"