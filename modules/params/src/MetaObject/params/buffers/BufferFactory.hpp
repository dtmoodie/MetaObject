#pragma once
#include "MetaObject/core/detail/Enums.hpp"
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include <functional>
#include <memory>
namespace mo
{
    class IParam;

    namespace Buffer
    {
        class MO_EXPORTS BufferFactory
        {
          public:
            typedef std::function<IParam*(IParam*)> create_buffer_f;

            static void RegisterFunction(TypeInfo type, const create_buffer_f& func, ParamType buffer_type_);
            static std::shared_ptr<IParam> CreateProxy(IParam* param, ParamType buffer_type_);
        };
    }
}