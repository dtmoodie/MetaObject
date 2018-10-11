#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/InputParam.hpp"
#include "MetaObject/params/buffers/BufferFactory.hpp"

#include <map>

using namespace mo;
using namespace mo::buffer;

namespace
{
    static std::map<mo::BufferFlags, BufferFactory::BufferConstructor>& registry()
    {
        static std::map<mo::BufferFlags, BufferFactory::BufferConstructor>* g_inst = nullptr;
        if (g_inst == nullptr)
        {
            g_inst = new std::map<mo::BufferFlags, BufferFactory::BufferConstructor>();
        }
        return *g_inst;
    }
}

void BufferFactory::registerConstructor(const BufferConstructor& func, BufferFlags buffer_type_)
{
    auto& map = registry();
    map[buffer_type_] = func;
}

InputParam* BufferFactory::createBuffer(IParam* param, mo::BufferFlags buffer_type_)
{
    auto& map = registry();
    auto itr = map.find(buffer_type_);
    if (itr == map.end())
    {
        return nullptr;
    }
    InputParam* buffer = itr->second();
    if (buffer)
    {
        if (buffer->setInput(param))
        {
            return buffer;
        }
    }
    return nullptr;
}

InputParam* BufferFactory::createBuffer(const std::shared_ptr<IParam>& param, mo::BufferFlags buffer_type_)
{
    auto& map = registry();
    auto itr = map.find(buffer_type_);
    if (itr == map.end())
    {
        return nullptr;
    }
    InputParam* buffer = itr->second();
    if (buffer)
    {
        if (buffer->setInput(param))
        {
            return buffer;
        }
    }
    return nullptr;
}
