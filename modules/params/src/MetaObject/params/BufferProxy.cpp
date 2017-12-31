#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/buffers/BufferFactory.hpp"
#include <map>

using namespace mo;
using namespace mo::Buffer;

static std::map<TypeInfo, std::map<ParamType, BufferFactory::create_buffer_f>>& registry()
{
    static std::map<TypeInfo, std::map<ParamType, BufferFactory::create_buffer_f>>* g_inst = nullptr;
    if (g_inst == nullptr)
    {
        g_inst = new std::map<TypeInfo, std::map<ParamType, BufferFactory::create_buffer_f>>();
    }
    return *g_inst;
}

void BufferFactory::RegisterFunction(TypeInfo type, const create_buffer_f& func, ParamType buffer_type_)
{
    auto& reg = registry();
    auto itr1 = reg.find(type);
    if (itr1 != reg.end())
    {
        auto itr2 = itr1->second.find(buffer_type_);
        if (itr2 != itr1->second.end())
            return;
    }
    registry()[type][buffer_type_] = func;
}
std::shared_ptr<IParam> BufferFactory::CreateProxy(IParam* param, ParamType buffer_type_)
{
    auto factory_func = registry().find(param->getTypeInfo());
    if (factory_func != registry().end())
    {
        if (factory_func->second[buffer_type_])
            return std::shared_ptr<IParam>(factory_func->second[buffer_type_](param));
    }
    return nullptr;
}