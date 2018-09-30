#include "MetaObject/params/ParamFactory.hpp"
#include "MetaObject/core/detail/singleton.hpp"
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/params/IParam.hpp"
#include <map>

using namespace mo;

struct ParamFactory::impl
{
    std::map<TypeInfo, std::map<ParamType, create_f>> _registered_constructors;
    std::map<TypeInfo, create_f> _registered_constructors_exact;
};

ParamFactory::ParamFactory() : m_pimpl(std::unique_ptr<ParamFactory::impl>(new ParamFactory::impl()))
{
}

ParamFactory::~ParamFactory()
{
    m_pimpl.release();
}

ParamFactory* ParamFactory::instance()
{
    return uniqueSingleton<ParamFactory>();
}

void ParamFactory::registerConstructor(const TypeInfo& data_type, create_f function, ParamType param_type)
{
    m_pimpl->_registered_constructors[data_type][param_type] = function;
}

void ParamFactory::registerConstructor(const TypeInfo& param_type, create_f function)
{
    m_pimpl->_registered_constructors_exact[param_type] = function;
}

std::shared_ptr<IParam> ParamFactory::create(const TypeInfo& data_type, ParamType param_type)
{
    auto itr = m_pimpl->_registered_constructors.find(data_type);
    if (itr != m_pimpl->_registered_constructors.end())
    {
        auto itr2 = itr->second.find(param_type);
        if (itr2 != itr->second.end())
        {
            return std::shared_ptr<IParam>(itr2->second());
        }
        MO_LOG(debug) << "Requested datatype (" << data_type.name()
                      << ") exists but the specified Param type : " << param_type << " does not.";
    }
    MO_LOG(debug) << "Requested datatype (" << data_type.name() << ") does not exist";
    return nullptr;
}

std::shared_ptr<IParam> ParamFactory::create(const TypeInfo& param_type)
{
    auto itr = m_pimpl->_registered_constructors_exact.find(param_type);
    if (itr != m_pimpl->_registered_constructors_exact.end())
    {
        return std::shared_ptr<IParam>(itr->second());
    }
    return nullptr;
}

std::vector<TypeInfo> ParamFactory::listConstructableDataTypes(ParamType type)
{
    std::vector<TypeInfo> output;
    for (auto itr1 = m_pimpl->_registered_constructors.begin(); itr1 != m_pimpl->_registered_constructors.end(); ++itr1)
    {
        for (auto itr2 = itr1->second.begin(); itr2 != itr1->second.end(); ++itr2)
        {
            if (itr2->first == type)
            {
                output.push_back(itr1->first);
            }
        }
    }
    return output;
}

std::vector<std::pair<TypeInfo, ParamType>> ParamFactory::listConstructableDataTypes()
{
    std::vector<std::pair<TypeInfo, ParamType>> output;
    for (auto itr1 = m_pimpl->_registered_constructors.begin(); itr1 != m_pimpl->_registered_constructors.end(); ++itr1)
    {
        for (auto itr2 = itr1->second.begin(); itr2 != itr1->second.end(); ++itr2)
        {
            output.emplace_back(itr1->first, itr2->first);
        }
    }
    return output;
}
