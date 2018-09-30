#pragma once
#include "InputParam.hpp"

namespace mo
{
    class MO_EXPORTS InputParamAny : public mo::InputParam
    {
      public:
        InputParamAny(const std::string& name = "");
        bool getInput(const OptionalTime_t& ts, size_t* fn = nullptr) override;
        bool getInput(size_t fn, OptionalTime_t* ts = nullptr) override;

        size_t getInputFrameNumber() override;
        OptionalTime_t getInputTimestamp() override;

        // This gets a pointer to the variable that feeds into this input
        IParam* getInputParam() const override;
        bool isInputSet() const override;
        bool setInput(std::shared_ptr<mo::IParam> param) override;
        bool setInput(mo::IParam* param = nullptr) override;

        virtual bool acceptsInput(std::weak_ptr<mo::IParam> param) const;
        bool acceptsInput(mo::IParam* param) const override;
        bool acceptsType(const mo::TypeInfo& type) const override;

        mo::TypeInfo getTypeInfo() const override;

        void on_param_update(mo::Context* ctx, mo::IParam* param);
        void on_param_delete(mo::IParam const*);

      protected:
        IParam* input = nullptr;
        static mo::TypeInfo _void_type_info;
        mo::TSlot<void(mo::Context*, mo::IParam*)> _update_slot;
        mo::TSlot<void(mo::IParam const*)> _delete_slot;
    };
}
