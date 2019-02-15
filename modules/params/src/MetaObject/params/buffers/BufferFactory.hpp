#pragma once
#include "MetaObject/core/detail/Enums.hpp"
#include "MetaObject/detail/Export.hpp"

#include <functional>

namespace std
{
    template <class T>
    class shared_ptr;
}
class SystemTable;
namespace mo
{
    class IParam;
    class InputParam;

    namespace buffer
    {
        class IBuffer;
        class MO_EXPORTS BufferFactory
        {
          public:
            using BufferConstructor = std::function<IBuffer*()>;

            static std::shared_ptr<BufferFactory> instance(SystemTable*);
            static std::shared_ptr<BufferFactory> instance();
            static void registerConstructor(const BufferConstructor& constructor, BufferFlags buffer);
            static IBuffer* createBuffer(IParam* param, BufferFlags flags);
            static IBuffer* createBuffer(const std::shared_ptr<IParam>& param, BufferFlags flags);
        };
    }
}
