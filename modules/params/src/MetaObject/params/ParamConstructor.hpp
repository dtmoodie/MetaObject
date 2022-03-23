#pragma once
#include "MetaObject/core/detail/Enums.hpp"
#include "ParamFactory.hpp"

namespace mo
{
    template <class T>
    class ParamConstructor
    {
      public:
        ParamConstructor()
        {
            ParamFactory::instance()->registerConstructor(
                TypeInfo(typeid(typename T::ValueType)), std::bind(&ParamConstructor<T>::create), T::Type);

            ParamFactory::instance()->registerConstructor(TypeInfo(typeid(T)), std::bind(&ParamConstructor<T>::create));
        }

        static std::shared_ptr<IParam> create()
        {
            return std::make_shared<T>();
        }
    };
} // namespace mo
