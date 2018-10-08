#pragma once
#include "MetaObject/core/detail/Enums.hpp"
#include "MetaObject/detail/Export.hpp"

#include <functional>
#include <map>

namespace mo
{
    class IParam;
    class InputParam;

    namespace buffer
    {
        class MO_EXPORTS BufferFactory
        {
          public:
            using BufferConstructor = std::function<InputParam*()>;
            static void registerConstructor(const BufferConstructor& constructor, BufferFlags buffer);
            static InputParam* createBuffer(IParam* param, BufferFlags flags);
        };
    }
}
