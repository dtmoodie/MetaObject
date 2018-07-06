#pragma once
#include <MetaObject/params/IParam.hpp>

namespace mo
{
    class OutputParam : virtual public IParam
    {
      public:
        virtual ~OutputParam();
        virtual std::vector<TypeInfo> listOutputTypes() const = 0;
        virtual IParam* getOutputParam(const TypeInfo type) = 0;
    };
}
