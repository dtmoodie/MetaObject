#pragma once
#include "InputParam.hpp"

namespace mo {
class MO_EXPORTS InputParamAny: public mo::InputParam {
public:
    InputParamAny(const std::string& name = "");
    bool getInput(OptionalTime_t ts, size_t* fn = nullptr);
    bool getInput(size_t fn, OptionalTime_t* ts = nullptr);

    size_t getInputFrameNumber();
    OptionalTime_t getInputTimestamp();

    // This gets a pointer to the variable that feeds into this input
    virtual IParam* getInputParam();
    virtual bool isInputSet() const;
    virtual bool setInput(std::shared_ptr<mo::IParam> param);
    virtual bool setInput(mo::IParam* param = nullptr);

    virtual bool acceptsInput(std::weak_ptr<mo::IParam> param) const;
    virtual bool acceptsInput(mo::IParam* param) const;
    virtual bool acceptsType(mo::TypeInfo type) const;

    const mo::TypeInfo& getTypeInfo() const;
    void on_param_update(mo::Context* ctx, mo::IParam* param);
    void on_param_delete(mo::IParam const *);
protected:
    IParam* input = nullptr;
    static mo::TypeInfo _void_type_info;
    mo::TSlot<void(mo::Context*, mo::IParam*)> _update_slot;
    mo::TSlot<void(mo::IParam const*)> _delete_slot;
};

}
