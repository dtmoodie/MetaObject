#include "MetaObject/Buffers/BufferProxy.hpp"
#include "parameters/Parameter.hpp"
using namespace mo;
using namespace mo::Buffer;

ParameterProxyBufferFactory* ParameterProxyBufferFactory::Instance()
{
    static ParameterProxyBufferFactory inst;
    return &inst;
}
void ParameterProxyBufferFactory::RegisterFunction(TypeInfo type, const create_buffer_f& func, buffer_type buffer_type_)
{
    _registered_buffer_factories[type][buffer_type_] = func;
}
Parameter*  ParameterProxyBufferFactory::CreateProxy(Parameter* param, buffer_type buffer_type_)
{
    auto factory_func = _registered_buffer_factories.find(param->GetTypeInfo());
    if (factory_func != _registered_buffer_factories.end())
    {
        if(factory_func->second[buffer_type_])
            return factory_func->second[buffer_type_](param);
    }
    return nullptr;
}
std::shared_ptr<Parameter>  ParameterProxyBufferFactory::CreateProxy(std::shared_ptr<Parameter> param, buffer_type buffer_type_)
{
    return std::shared_ptr<Parameter>(CreateProxy(param.get(), buffer_type_));
}