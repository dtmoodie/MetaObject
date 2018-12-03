#include "MetaObject/params/ParamFactory.hpp"
#include "MetaObject/core/detail/singleton.hpp"
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/params/IParam.hpp"
#include <map>

using namespace mo;

struct ParamFactory::impl
{
    std::map<TypeInfo, std::map<BufferFlags, create_f>> _registered_constructors;
    std::map<TypeInfo, create_f> _registered_constructors_exact;
};

ParamFactory::ParamFactory()
    : m_pimpl(std::unique_ptr<ParamFactory::impl>(new ParamFactory::impl()))
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

void ParamFactory::registerConstructor(const TypeInfo& data_type, create_f function, BufferFlags param_type)
{
    m_pimpl->_registered_constructors[data_type][param_type] = function;
}

void ParamFactory::registerConstructor(const TypeInfo& param_type, create_f function)
{
    m_pimpl->_registered_constructors_exact[param_type] = function;
}

std::shared_ptr<IParam> ParamFactory::create(const TypeInfo& data_type, BufferFlags param_type)
{
    auto itr = m_pimpl->_registered_constructors.find(data_type);
    if (itr != m_pimpl->_registered_constructors.end())
    {
        auto itr2 = itr->second.find(param_type);
        if (itr2 != itr->second.end())
        {
            return std::shared_ptr<IParam>(itr2->second());
        }
        MO_LOG(debug,
               "Requsted datatype ({}) exists but the specified param type ({}) does not",
               data_type.name(),
               param_type);
    }
    MO_LOG(debug, "Requsted datatype ({}) does not exist", data_type.name());
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

std::vector<TypeInfo> ParamFactory::listConstructableDataTypes(BufferFlags type)
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

std::vector<std::pair<TypeInfo, BufferFlags>> ParamFactory::listConstructableDataTypes()
{
    std::vector<std::pair<TypeInfo, BufferFlags>> output;
    for (auto itr1 = m_pimpl->_registered_constructors.begin(); itr1 != m_pimpl->_registered_constructors.end(); ++itr1)
    {
        for (auto itr2 = itr1->second.begin(); itr2 != itr1->second.end(); ++itr2)
        {
            output.emplace_back(itr1->first, itr2->first);
        }
    }
    return output;
}
