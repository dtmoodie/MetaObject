#include "MetaObject/params/VariableManager.hpp"
#include "MetaObject/params/IParam.hpp"
#include "MetaObject/params/InputParam.hpp"
#include "MetaObject/logging/logging.hpp"
#include "MetaObject/signals/TSlot.hpp"
#include "MetaObject/signals/TSignalRelay.hpp"
#include <map>
using namespace mo;
struct VariableManager::impl
{
    std::map<std::string, IParam*> _Params;
    //std::map<std::string, std::shared_ptr<Connection>> _delete_Connections;
    TSlot<void(IParam*)> delete_slot;
};
VariableManager::VariableManager()
{
    pimpl = new impl();
    pimpl->delete_slot = std::bind(&VariableManager::removeParam, this, std::placeholders::_1);
}
VariableManager::~VariableManager()
{
    delete pimpl;
}
void VariableManager::addParam(IParam* param)
{
    pimpl->_Params[param->getTreeName()] = param;
    param->registerDeleteNotifier(&pimpl->delete_slot);
}
void VariableManager::removeParam(IParam* param)
{
    pimpl->_Params.erase(param->getTreeName());
}
std::vector<IParam*> VariableManager::getOutputParams(TypeInfo type)
{
    std::vector<IParam*> valid_outputs;
    for(auto itr = pimpl->_Params.begin(); itr != pimpl->_Params.end(); ++itr)
    {
        if(itr->second->getTypeInfo() == type && itr->second->checkFlags(Output_e))
        {
            valid_outputs.push_back(itr->second);
        }
    }
    return valid_outputs;
}
std::vector<IParam*> VariableManager::getAllParms()
{
    std::vector<IParam*> output;
    for(auto& itr : pimpl->_Params)
    {
        output.push_back(itr.second);
    }
    return output;
}
std::vector<IParam*> VariableManager::getAllOutputParams()
{
    std::vector<IParam*> output;
    for(auto& itr : pimpl->_Params)
    {
        if(itr.second->checkFlags(Output_e))
        {
            output.push_back(itr.second);
        }
    }
    return output;
}
IParam* VariableManager::getParam(std::string name)
{
    auto itr = pimpl->_Params.find(name);
    if(itr != pimpl->_Params.end())
    {
        return itr->second;
    }
    return nullptr;
}

IParam* VariableManager::getOutputParam(std::string name)
{
    auto itr = pimpl->_Params.find(name);
    if(itr != pimpl->_Params.end())
    {
        return itr->second;
    }
    // Check if the passed in value is the item specific name
    std::vector<IParam*> potentials;
    for(auto& itr : pimpl->_Params)
    {
        if(itr.first.find(name) != std::string::npos)
        {
            potentials.push_back(itr.second);
        }
    }
    if(potentials.size())
    {
        if(potentials.size() > 1)
        {
            std::stringstream ss;
            for(auto potential : potentials)
                ss << potential->getTreeName() << "\n";
            MO_LOG(debug) << "Warning ambiguous name \"" << name << "\" passed in, multiple potential matches\n " << ss.str();
        }
        return potentials[0];
    }
    MO_LOG(debug) << "Unable to find Param named " << name;
    return nullptr;
}
void VariableManager::linkParams(IParam* output, IParam* input)
{
    if(auto input_param = dynamic_cast<InputParam*>(input))
        input_param->setInput(output);
}
