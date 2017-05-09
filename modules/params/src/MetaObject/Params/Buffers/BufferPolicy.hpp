#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.hpp"
#include "BufferFactory.hpp"

namespace mo
{
    class IParam;
    namespace Buffer
    {
        template<typename T> class Proxy;
        
        template<typename T> struct BufferConstructor
        {
			BufferConstructor()
            {
                BufferFactory::RegisterFunction(TypeInfo(typeid(typename T::ValueType)),
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
