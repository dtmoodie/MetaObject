#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include "BufferFactory.hpp"

#include <functional>
#include <map>
#include <memory>

namespace mo
{
    class IParameter;
    namespace Buffer
    {
        template<typename T> class Proxy;
        template<typename T> class CircularBuffer;
        template<typename T> class ConstMap;
        template<typename T> class Map;
        
        template<typename T> struct BufferConstructor
        {
			BufferConstructor()
            {
				BufferFactory::Instance()->RegisterFunction(TypeInfo(typeid(T)),
					std::bind(&BufferConstructor<T>::create_cbuffer, std::placeholders::_1),
					ParameterProxyBufferFactory::cbuffer);

				BufferFactory::Instance()->RegisterFunction(TypeInfo(typeid(T)),
					std::bind(&BufferConstructor<T>::create_cmap, std::placeholders::_1),
					ParameterProxyBufferFactory::cmap);

				BufferFactory::Instance()->RegisterFunction(TypeInfo(typeid(T)),
					std::bind(&BufferConstructor<T>::create_map, std::placeholders::_1),
					ParameterProxyBufferFactory::map);
            }
            static Parameter* create_cbuffer(Parameter* input)
            {
                if (auto typed_param = dynamic_cast<ITypedParameter<T>*>(input))
                {
                    return new Proxy<T>(typed_param, new CircularBuffer<T>("buffer for " + input->GetTreeName()));
                }
                return nullptr;
            }
            static Parameter* create_cmap(Parameter* input)
            {
                if (auto typed_param = dynamic_cast<ITypedParameter<T>*>(input))
                {
                    return new Proxy<T>(typed_param, new ConstMap<T>("cmap for " + input->GetTreeName()));
                }
                return nullptr;
            }
            static Parameter* create_map(Parameter* input)
            {
                if (auto typed_param = dynamic_cast<ITypedParameter<T>*>(input))
                {
                    return new Proxy<T>(typed_param, new Map<T>("map for " + input->GetTreeName()));
                }
                return nullptr;
            }
        };

        template<typename T> class ParameterBufferPolicy
        {
            static BufferConstructor<T> _buffer_constructor;
        public:
            ParameterBufferPolicy()
            {
                (void)&_buffer_constructor;
            }
        };
        template<typename T> BufferConstructor<T> ParameterBufferPolicy<T>::_buffer_constructor;
    }
}