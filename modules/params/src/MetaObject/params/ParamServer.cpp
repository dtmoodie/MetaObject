#include "MetaObject/params/ParamServer.hpp"
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/InputParam.hpp"
#include "MetaObject/signals/TSignalRelay.hpp"

using namespace mo;

IParamServer::~IParamServer()
{
}

ParamServer::ParamServer()
{
    delete_slot = std::bind(static_cast<void (ParamServer::*)(IMetaObject*, IParam*)>(&ParamServer::removeParam),
                            this,
                            std::placeholders::_1,
                            std::placeholders::_2);
}

ParamServer::~ParamServer()
{
}

void ParamServer::addParam(IMetaObject* obj, IParam* param)
{
    _params[param->getTreeName()] = param;
    _obj_params[obj].push_back(param->getTreeName());
    param->registerDeleteNotifier(&delete_slot);
}

void ParamServer::removeParam(IMetaObject* /*obj*/, IParam* param)
{
    _params.erase(param->getTreeName());
}

void ParamServer::removeParam(const IMetaObject* obj)
{
    auto itr = _obj_params.find(obj);
    if (itr != _obj_params.end())
    {
        for (const auto& name : itr->second)
        {
            _params.erase(name);
        }
    }
}

std::vector<IParam*> ParamServer::getOutputParams(TypeInfo type)
{
    std::vector<IParam*> valid_outputs;
    for (auto itr = _params.begin(); itr != _params.end(); ++itr)
    {
        if (itr->second->getTypeInfo() == type && itr->second->checkFlags(ParamFlags::Output_e))
        {
            valid_outputs.push_back(itr->second);
        }
    }
    return valid_outputs;
}

std::vector<IParam*> ParamServer::getAllParms()
{
    std::vector<IParam*> output;
    for (auto& itr : _params)
    {
        output.push_back(itr.second);
    }
    return output;
}

std::vector<IParam*> ParamServer::getAllOutputParams()
{
    std::vector<IParam*> output;
    for (auto& itr : _params)
    {
        if (itr.second->checkFlags(ParamFlags::Output_e))
        {
            output.push_back(itr.second);
        }
    }
    return output;
}

IParam* ParamServer::getParam(std::string name)
{
    auto itr = _params.find(name);
    if (itr != _params.end())
    {
        return itr->second;
    }
    return nullptr;
}

IParam* ParamServer::getOutputParam(std::string name)
{
    auto itr = _params.find(name);
    if (itr != _params.end())
    {
        return itr->second;
    }
    // Check if the passed in value is the item specific name
    std::vector<IParam*> potentials;
    for (auto& itr : _params)
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
        return potentials[0];
    }
    MO_LOG(debug, "Unable to find param with name name {}", name);
    return nullptr;
}
void ParamServer::linkParams(IParam* output, IParam* input)
{
    if (auto input_param = dynamic_cast<InputParam*>(input))
        input_param->setInput(output);
}
