#pragma once
#include "BufferFactory.hpp"
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include "MetaObject/params/ITParam.hpp"

namespace mo
{
    namespace Buffer
    {
        template <typename T>
        class Proxy;

        template <typename T>
        struct BufferConstructor
        {
            BufferConstructor()
            {
                BufferFactory::registerFunction(TypeInfo(typeid(typename T::ValueType)),
                                                std::bind(&BufferConstructor<T>::create_buffer, std::placeholders::_1),
                                                T::BufferType);
            }
            static IParam* create_buffer(IParam* input)
            {
                if (auto T_param = dynamic_cast<ITParam<T>*>(input))
                {
                    return new Proxy<T>(T_param, new T("map for " + input->getTreeName()));
                }
                return nullptr;
            }
        };
    }
}
