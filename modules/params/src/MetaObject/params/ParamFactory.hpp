#pragma once
#include "MetaObject/core/detail/Enums.hpp"
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include <functional>
#include <memory>
#include <vector>
namespace mo
{
    class IParam;
    class MO_EXPORTS ParamFactory
    {
      public:
        typedef std::function<std::shared_ptr<IParam>(void)> create_f;
        static ParamFactory* instance();

        ParamFactory();
        ~ParamFactory();
        // Each specialization of a Param must have a unique type
        void registerConstructor(const TypeInfo& data_type, create_f function, BufferFlags Param_type);
        void registerConstructor(const TypeInfo& Param_type, create_f function);

        // Give datatype and Param type enum
        std::shared_ptr<IParam> create(const TypeInfo& data_type, BufferFlags Param_type);

        // Must give exact Param type, such as TParam<int>
        std::shared_ptr<IParam> create(const TypeInfo& Param_type);

        std::vector<TypeInfo> listConstructableDataTypes(BufferFlags type);
        std::vector<std::pair<TypeInfo, BufferFlags>> listConstructableDataTypes();

      private:
        struct impl;
        std::unique_ptr<impl> m_pimpl;
    };
}
