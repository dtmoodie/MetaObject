#pragma once
#include "ITParam.hpp"

namespace mo
{
    template <typename T>
    class ITAccessibleParam : virtual public ITParam<T>
    {
      public:
        virtual AccessToken<T> access() = 0;
        virtual ConstAccessToken<T> access() const = 0;
    };
}
