#pragma once
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/params/InputParam.hpp"

namespace mo
{
    class ParamServer;
    class ParamClient;
    class MO_EXPORTS ZeroMQContext
    {
      public:
        static ZeroMQContext* Instance();

      protected:
        friend class ParamServer;
        friend class ParamClient;
        struct impl;
        impl* _pimpl;

      private:
        ZeroMQContext();
        ZeroMQContext(const ZeroMQContext& ctx) = delete;
        ZeroMQContext& operator=(const ZeroMQContext& ctx) = delete;
    };

    class MO_EXPORTS ParamPublisher : public InputParam
    {
      public:
        ParamPublisher();
        virtual ~ParamPublisher();

        virtual bool getInput(const OptionalTime_t& time = OptionalTime_t()) = 0;

        // This gets a pointer to the variable that feeds into this input
        virtual IParam* getInputParam() = 0;

        // Set input and setup callbacks
        virtual bool setInput(std::shared_ptr<IParam> param) = 0;
        virtual bool setInput(IParam* param = nullptr) = 0;

        // Check for correct serialization routines, etc
        virtual bool AcceptsInput(std::weak_ptr<IParam> param) const = 0;
        virtual bool AcceptsInput(IParam* param) const = 0;
        virtual bool AcceptsType(TypeInfo type) const = 0;

      protected:
        void onInputUpdate(Context* ctx, IParam* param);
        struct impl;
        impl* _pimpl;
    };
}