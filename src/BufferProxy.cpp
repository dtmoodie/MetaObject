#include "MetaObject/Buffers/BufferFactory.hpp"
#include "MetaObject/Parameters/IParameter.hpp"

using namespace mo;
using namespace mo::Buffer;

BufferFactory* BufferFactory::Instance()
{
    static BufferFactory inst;
    return &inst;
}
void BufferFactory::RegisterFunction(TypeInfo type, const create_buffer_f& func, buffer_type buffer_type_)
{
    _registered_buffer_factories[type][buffer_type_] = func;
}
std::shared_ptr<IParameter>  BufferFactory::CreateProxy(std::weak_ptr<IParameter> param_, buffer_type buffer_type_)
{
    std::shared_ptr<IParameter> param(param_);
    auto factory_func = _registered_buffer_factories.find(param->GetTypeInfo());
    if (factory_func != _registered_buffer_factories.end())
    {
        if(factory_func->second[buffer_type_])
            return std::shared_ptr<IParameter>(factory_func->second[buffer_type_](param.get()));
    }
    return nullptr;
}