#pragma once
#include "MetaObject/core/detail/Enums.hpp"
#include "MetaObject/detail/Export.hpp"

#include <functional>

namespace std
{
    template <class T>
    class shared_ptr;
}

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
            static InputParam* createBuffer(const std::shared_ptr<IParam>& param, BufferFlags flags);
        };
    }
}
