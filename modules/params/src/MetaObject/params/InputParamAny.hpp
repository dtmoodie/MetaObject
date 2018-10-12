#pragma once
#include "InputParam.hpp"

namespace mo
{
    class MO_EXPORTS InputParamAny : public mo::InputParam
    {
      public:
        InputParamAny(const std::string& name = "");

        virtual bool getInputData(const Header& desired, Header* retrieved) override;

        size_t getInputFrameNumber() override;
        OptionalTime getInputTimestamp() override;

        // This gets a pointer to the variable that feeds into this input
        IParam* getInputParam() const override;
        bool isInputSet() const override;
        bool setInput(const std::shared_ptr<mo::IParam>& param) override;
        bool setInput(mo::IParam* param = nullptr) override;

        virtual bool acceptsInput(std::weak_ptr<mo::IParam> param) const;
        bool acceptsInput(mo::IParam* param) const override;
        bool acceptsType(const mo::TypeInfo& type) const override;

        mo::TypeInfo getTypeInfo() const override;

        void on_param_update(mo::Context* ctx, mo::IParam* param);
        void on_param_delete(mo::IParam const*);

      protected:
        IParam* input = nullptr;
        std::shared_ptr<IParam> m_shared_input;
        static mo::TypeInfo _void_type_info;
        mo::TSlot<void(mo::Context*, mo::IParam*)> _update_slot;
        mo::TSlot<void(mo::IParam const*)> _delete_slot;
        IContainerConstPtr_t m_data;
    };
}
