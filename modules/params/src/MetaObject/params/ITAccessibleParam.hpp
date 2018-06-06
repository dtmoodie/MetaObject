#pragma once
#include "ITParam.hpp"

namespace mo
{

    template <typename T>
    class ITConstAccessibleParam : virtual public ITParam<T>
    {
      public:
        virtual bool canAccess() const = 0;
        virtual ConstAccessToken<T> access() const = 0;
    };

    template <typename T>
    class ITAccessibleParam : virtual public ITConstAccessibleParam<T>
    {
      public:
        virtual AccessToken<T> access() = 0;
    };
}
