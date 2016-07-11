#include "parameters/ParameterFactory.hpp"
#include <signals/logging.hpp>
using namespace Parameters;

ParameterFactory* ParameterFactory::instance()
{
    static ParameterFactory* inst = nullptr;
    if(inst == nullptr)
        inst = new ParameterFactory();
    return inst;
}

void ParameterFactory::RegisterConstructor(Loki::TypeInfo data_type, create_f function, int parameter_type)
{
    _registered_constructors[data_type][parameter_type] = function;
}
void ParameterFactory::RegisterConstructor(Loki::TypeInfo parameter_type, create_f function)
{
    _registered_constructors_exact[parameter_type] = function;
}

Parameter* ParameterFactory::create(Loki::TypeInfo data_type, int parameter_type)
{
    auto itr = _registered_constructors.find(data_type);
    if(itr != _registered_constructors.end())
    {
        auto itr2 = itr->second.find(parameter_type);
        if(itr2 != itr->second.end())
        {
            return itr2->second();
        }
        LOG(debug) << "Requested datatype (" << data_type.name() << ") exists but the specified parameter type : " << parameter_type << " does not.";
    }
    LOG(debug) << "Requested datatype (" << data_type.name() << ") does not exist";
    return nullptr;
}
Parameter* ParameterFactory::create(Loki::TypeInfo parameter_type)
{
    auto itr = _registered_constructors_exact.find(parameter_type);
    if(itr != _registered_constructors_exact.end())
    {
        return itr->second();
    }
    return nullptr;
}