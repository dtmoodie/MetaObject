#pragma once

#include "Map.hpp"

namespace mo
{
    namespace Buffer
    {
        template <typename T>
        class ConstMap : public Map<T>
        {
          public:
            typedef T ValueType;
            static const ParamType Type = ConstMap_e;
            ConstMap(const std::string& name = "",
                     const T& init = T(),
                     OptionalTime_t ts = OptionalTime_t(),
                     ParamType& type = ParamFlags::Buffer_e,
                     const std::string& tooltip = "")
                : Map<T>(name, init, ts, type, tooltip)
            {
                (void)&_constructor;
                _size = 10;
            }

            void clean()
            {
                while (_data_buffer.size() > _size)
                {
                    _data_buffer.erase(_data_buffer.begin());
                }
            }
            virtual void SetSize(long long size) { _size = size; }
            virtual ParamType getBufferType() const { return ConstMap_e; }
          private:
            size_t _size;
        };
    }
#define MO_METAParam_INSTANCE_CONST_MAP_(N)                                                                            \
    template <class T>                                                                                                 \
    struct MetaParam<T, N, void> : public MetaParam<T, N - 1, void>                                                    \
    {                                                                                                                  \
        static ParamConstructor<Buffer::ConstMap<T>> _map_Param_constructor;                                           \
        static BufferConstructor<Buffer::ConstMap<T>> _map_constructor;                                                \
        MetaParam<T, N>(const char* name) : MetaParam<T, N - 1>(name)                                                  \
        {                                                                                                              \
            (void)&_map_Param_constructor;                                                                             \
            (void)&_map_constructor;                                                                                   \
        }                                                                                                              \
    };                                                                                                                 \
    template <class T>                                                                                                 \
    ParamConstructor<Buffer::ConstMap<T>> MetaParam<T, N, void>::_map_Param_constructor;                               \
    template <class T>                                                                                                 \
    BufferConstructor<Buffer::ConstMap<T>> MetaParam<T, N, void>::_map_constructor;

    MO_METAParam_INSTANCE_CONST_MAP_(__COUNTER__)
}
