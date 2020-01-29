#pragma once
#include "InputParam.hpp"

namespace mo
{
    class MO_EXPORTS InputParamAny : public mo::InputParam
    {
      public:
        InputParamAny(const std::string& name = "");

        bool acceptsInput(mo::IParam* param) const override;
        bool acceptsType(const mo::TypeInfo& type) const override;

      protected:
    };
}
