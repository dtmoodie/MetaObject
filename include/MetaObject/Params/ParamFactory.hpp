#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/Enums.hpp"
#include "MetaObject/Detail/TypeInfo.hpp"
#include <functional>
#include <memory>
namespace mo {
class IParam;
// Only include types that it makes sense to dynamically construct.
// No reason to create a TParamPtr most of the time because it is used to wrap
// user owned data


class MO_EXPORTS ParamFactory {
public:
    typedef std::function<IParam*(void)> create_f;
    static ParamFactory* instance();

    // Each specialization of a Param must have a unique type
    void RegisterConstructor(TypeInfo data_type, create_f function, ParamType Param_type);
    void RegisterConstructor(TypeInfo Param_type, create_f function);

    // Give datatype and Param type enum
    std::shared_ptr<IParam> create(TypeInfo data_type, ParamType Param_type);
    // Must give exact Param type, such as TParam<int>
    std::shared_ptr<IParam> create(TypeInfo Param_type);
private:
    struct impl;
    std::shared_ptr<impl> pimpl;
};
}