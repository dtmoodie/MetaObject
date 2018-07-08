#pragma once
#include <MetaObject/params/IParam.hpp>

namespace mo
{
    class OutputParam : virtual public IParam
    {
      public:
        virtual ~OutputParam();
        virtual std::vector<TypeInfo> listOutputTypes() const = 0;
        virtual ParamBase* getOutputParam(const TypeInfo type) = 0;
        virtual const ParamBase* getOutputParam(const TypeInfo type) const = 0;
        virtual ParamBase* getOutputParam() = 0;
        virtual const ParamBase* getOutputParam() const = 0;
    };
}
