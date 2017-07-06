#include "MetaObject/params/ParamFactory.hpp"
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/params/IParam.hpp"
#include <map>

using namespace mo;
struct ParamFactory::impl
{
    std::map<TypeInfo, std::map<int, create_f>> _registered_constructors;
    std::map<TypeInfo, create_f> _registered_constructors_exact;
};
ParamFactory* ParamFactory::instance()
{
    static ParamFactory* inst = nullptr;
    if(inst == nullptr)
        inst = new ParamFactory();
    if(inst->pimpl == nullptr)
        inst->pimpl.reset(new ParamFactory::impl());
    return inst;
}

void ParamFactory::RegisterConstructor(TypeInfo data_type, create_f function, ParamType Param_type)
{
    pimpl->_registered_constructors[data_type][Param_type] = function;
}
void ParamFactory::RegisterConstructor(TypeInfo Param_type, create_f function)
{
    pimpl->_registered_constructors_exact[Param_type] = function;
}

std::shared_ptr<IParam> ParamFactory::create(TypeInfo data_type, ParamType Param_type)
{
    auto itr = pimpl->_registered_constructors.find(data_type);
    if(itr != pimpl->_registered_constructors.end())
    {
        auto itr2 = itr->second.find(Param_type);
        if(itr2 != itr->second.end())
        {
            return std::shared_ptr<IParam>(itr2->second());
        }
        MO_LOG(debug) << "Requested datatype (" << data_type.name() << ") exists but the specified Param type : " << Param_type << " does not.";
    }
    MO_LOG(debug) << "Requested datatype (" << data_type.name() << ") does not exist";
    return nullptr;
}

std::shared_ptr<IParam> ParamFactory::create(TypeInfo Param_type)
{
    auto itr = pimpl->_registered_constructors_exact.find(Param_type);
    if(itr != pimpl->_registered_constructors_exact.end())
    {
        return std::shared_ptr<IParam>(itr->second());
    }
    return nullptr;
}