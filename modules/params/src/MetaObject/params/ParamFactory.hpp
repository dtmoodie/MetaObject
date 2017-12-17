#pragma once
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/core/detail/Enums.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include <functional>
#include <memory>
#include <vector>
namespace mo {
class IParam;
class MO_EXPORTS ParamFactory {
public:
    typedef std::function<std::shared_ptr<IParam>(void)> create_f;
    static ParamFactory* instance();

    ParamFactory();
    ~ParamFactory();
    // Each specialization of a Param must have a unique type
    void registerConstructor(const TypeInfo& data_type, create_f function, ParamType Param_type);
    void registerConstructor(const TypeInfo& Param_type, create_f function);

    // Give datatype and Param type enum
    std::shared_ptr<IParam> create(const TypeInfo& data_type, ParamType Param_type);

    // Must give exact Param type, such as TParam<int>
    std::shared_ptr<IParam> create(const TypeInfo& Param_type);

    std::vector<TypeInfo> listConstructableDataTypes(ParamType type);
    std::vector<std::pair<TypeInfo, ParamType>> listConstructableDataTypes();
private:
    struct impl;
    std::unique_ptr<impl> m_pimpl;
};
}