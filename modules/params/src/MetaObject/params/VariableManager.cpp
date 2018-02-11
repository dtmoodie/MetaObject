#include "MetaObject/params/VariableManager.hpp"
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/InputParam.hpp"
#include "MetaObject/signals/TSignalRelay.hpp"
#include "MetaObject/signals/TSlot.hpp"
#include <map>
using namespace mo;
struct VariableManager::impl
{
    std::map<std::string, IParam*> _params;
    TSlot<void(IMetaObject*, IParam*)> delete_slot;
    std::map<const IMetaObject*, std::vector<std::string>> _obj_params;
};

VariableManager::VariableManager()
{
    pimpl = new impl();
    pimpl->delete_slot = std::bind(static_cast<void(VariableManager::*)(IMetaObject*, IParam*)>(&VariableManager::removeParam), this, std::placeholders::_1, std::placeholders::_2);
}

VariableManager::~VariableManager()
{
    delete pimpl;
}

void VariableManager::addParam(IMetaObject* obj, IParam* param)
{
    pimpl->_params[param->getTreeName()] = param;
    pimpl->_obj_params[obj].push_back(param->getTreeName());
    param->registerDeleteNotifier(&pimpl->delete_slot);
}

void VariableManager::removeParam(IMetaObject* obj, IParam* param)
{
    pimpl->_params.erase(param->getTreeName());
}

void VariableManager::removeParam(const IMetaObject* obj)
{
    auto itr = pimpl->_obj_params.find(obj);
    if(itr != pimpl->_obj_params.end())
    {
        for(const auto& name : itr->second)
        {
            pimpl->_params.erase(name);
        }
    }
}

std::vector<IParam*> VariableManager::getOutputParams(TypeInfo type)
{
    std::vector<IParam*> valid_outputs;
    for (auto itr = pimpl->_params.begin(); itr != pimpl->_params.end(); ++itr)
    {
        if (itr->second->getTypeInfo() == type && itr->second->checkFlags(ParamFlags::Output_e))
        {
            valid_outputs.push_back(itr->second);
        }
    }
    return valid_outputs;
}

std::vector<IParam*> VariableManager::getAllParms()
{
    std::vector<IParam*> output;
    for (auto& itr : pimpl->_params)
    {
        output.push_back(itr.second);
    }
    return output;
}

std::vector<IParam*> VariableManager::getAllOutputParams()
{
    std::vector<IParam*> output;
    for (auto& itr : pimpl->_params)
    {
        if (itr.second->checkFlags(ParamFlags::Output_e))
        {
            output.push_back(itr.second);
        }
    }
    return output;
}

IParam* VariableManager::getParam(std::string name)
{
    auto itr = pimpl->_params.find(name);
    if (itr != pimpl->_params.end())
    {
        return itr->second;
    }
    return nullptr;
}

IParam* VariableManager::getOutputParam(std::string name)
{
    auto itr = pimpl->_params.find(name);
    if (itr != pimpl->_params.end())
    {
        return itr->second;
    }
    // Check if the passed in value is the item specific name
    std::vector<IParam*> potentials;
    for (auto& itr : pimpl->_params)
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
            MO_LOG(debug) << "Warning ambiguous name \"" << name << "\" passed in, multiple potential matches\n "
                          << ss.str();
        }
        return potentials[0];
    }
    MO_LOG(debug) << "Unable to find Param named " << name;
    return nullptr;
}
void VariableManager::linkParams(IParam* output, IParam* input)
{
    if (auto input_param = dynamic_cast<InputParam*>(input))
        input_param->setInput(output);
}
