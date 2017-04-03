#pragma once

#include "Map.hpp"

namespace mo
{
    namespace Buffer
    {
        template<typename T> class ConstMap: public Map<T>
        {
        public:
            typedef T ValueType;
			static const ParameterTypeFlags Type = ConstMap_e;
            ConstMap(const std::string& name = "",
					 const T& init = T(), 
					 boost::optional<mo::time_t> ts= boost::optional<mo::time_t>(),
					 ParameterType& type = Buffer_e,
					const std::string& tooltip = "") :
                Map<T>(name, init, ts, type, tooltip)
            {
                (void)&_constructor;
                _size = 10;
            }

            void clean()
            {
                while(_data_buffer.size() > _size)
                {
                    _data_buffer.erase(_data_buffer.begin());
                }
            }
            virtual void SetSize(long long size)
            {
                _size = size;
            }
            virtual ParameterTypeFlags GetBufferType() const{ return ConstMap_e;}
		private:
			size_t _size;
        };
    }
#define MO_METAPARAMETER_INSTANCE_CONST_MAP_(N) \
    template<class T> struct MetaParameter<T, N, void>: public MetaParameter<T, N-1, void> \
    { \
        static ParameterConstructor<Buffer::ConstMap<T>> _map_parameter_constructor; \
        static BufferConstructor<Buffer::ConstMap<T>> _map_constructor;  \
        MetaParameter<T, N>(const char* name): \
            MetaParameter<T, N-1>(name) \
        { \
            (void)&_map_parameter_constructor; \
            (void)&_map_constructor; \
        } \
    }; \
    template<class T> ParameterConstructor<Buffer::ConstMap<T>> MetaParameter<T, N, void>::_map_parameter_constructor; \
    template<class T> BufferConstructor<Buffer::ConstMap<T>> MetaParameter<T, N, void>::_map_constructor;

	MO_METAPARAMETER_INSTANCE_CONST_MAP_(__COUNTER__)
}
