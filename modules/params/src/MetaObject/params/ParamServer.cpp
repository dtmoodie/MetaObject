#include <MetaObject/logging/logging.hpp>
#include <MetaObject/params/IParam.hpp>
#include <MetaObject/params/IPublisher.hpp>
#include <MetaObject/params/ISubscriber.hpp>
#include <MetaObject/params/ParamServer.hpp>
#include <MetaObject/signals/TSignalRelay.hpp>

using namespace mo;

IParamServer::~IParamServer()
{
}

ParamServer::ParamServer()
{
    m_delete_slot.bind(&ParamServer::removeParam, this);
    m_obj_delete_slot.bind(&ParamServer::removeParams, this);
}

ParamServer::~ParamServer()
{
}

void ParamServer::addParam(IMetaObject& obj, IParam& param)
{
    m_params[param.getTreeName()] = &param;
    m_obj_params[&obj].push_back(param.getTreeName());
    param.registerDeleteNotifier(m_delete_slot);
}

void ParamServer::removeParam(const IParam& param)
{
    auto itr = m_params.find(param.getTreeName());
    if (itr == m_params.end())
    {
        for (itr = m_params.begin(); itr != m_params.end(); ++itr)
        {
            if (itr->second == &param)
            {
                m_params.erase(itr);
                return;
            }
        }
    }
    else
    {
        m_params.erase(itr);
    }
    // log error?
}

void ParamServer::removeParams(const IMetaObject& obj)
{
    auto itr = m_obj_params.find(&obj);
    if (itr != m_obj_params.end())
    {
        for (const auto& name : itr->second)
        {
            m_params.erase(name);
        }
    }
}

std::vector<IPublisher*> ParamServer::getPublishers(TypeInfo type)
{
    std::vector<IPublisher*> valid_outputs;
    const auto void_type = TypeInfo::Void();
    for (auto itr = m_params.begin(); itr != m_params.end(); ++itr)
    {
        IParam* ptr = itr->second;
        if (ptr->checkFlags(ParamFlags::kOUTPUT))
        {
            auto publisher = dynamic_cast<IPublisher*>(ptr);
            if (publisher)
            {
                const auto output_types = publisher->getOutputTypes();
                if (type == void_type ||
                    std::find(output_types.begin(), output_types.end(), type) != output_types.end())
                {
                    valid_outputs.push_back(publisher);
                }
            }
        }
    }
    return valid_outputs;
}

std::vector<IParam*> ParamServer::getAllParms()
{
    std::vector<IParam*> output;
    for (auto& itr : m_params)
    {
        output.push_back(itr.second);
    }
    return output;
}

IParam* ParamServer::getParam(std::string name)
{
    auto itr = m_params.find(name);
    if (itr != m_params.end())
    {
        return itr->second;
    }
    return nullptr;
}

IPublisher* ParamServer::getPublisher(std::string name)
{
    auto itr = m_params.find(name);
    if (itr != m_params.end())
    {
        return dynamic_cast<IPublisher*>(itr->second);
    }
    // Check if the passed in value is the item specific name
    std::vector<IParam*> potentials;
    for (auto& itr : m_params)
    {
        if (itr.first.find(name) != std::string::npos)
        {
            potentials.push_back(itr.second);
        }
    }
    if (potentials.size())
    {
        if (potentials.size() > 1)
        {
            std::stringstream ss;
            for (auto potential : potentials)
                ss << potential->getTreeName() << "\n";
            MO_LOG(debug, "Warning ambiguous name '{}' passed in, multiple potential matches", ss.str());
        }
        return dynamic_cast<IPublisher*>(potentials[0]);
    }
    MO_LOG(debug, "Unable to find param with name name {}", name);
    return nullptr;
}
void ParamServer::linkParams(IPublisher& output, ISubscriber& input)
{
    input.setInput(&output);
}
