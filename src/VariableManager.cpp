#include "MetaObject/Parameters/VariableManager.h"
#include "MetaObject/Parameters/IParameter.hpp"
#include "MetaObject/Parameters/InputParameter.hpp"
#include "MetaObject/Logging/Log.hpp"
#include <map>
using namespace mo;
struct VariableManager::impl
{
    std::map<std::string, std::weak_ptr<IParameter>> _parameters;
    std::map<std::string, std::shared_ptr<Connection>> _delete_connections;
};
VariableManager::VariableManager()
{
    pimpl = new impl();
}
VariableManager::~VariableManager()
{
    delete pimpl;
}
void VariableManager::AddParameter(std::shared_ptr<IParameter> param)
{
    pimpl->_parameters[param->GetTreeName()] = param;
    pimpl->_delete_connections[param->GetTreeName()] = param->RegisterDeleteNotifier(std::bind(&VariableManager::RemoveParameter, this, std::placeholders::_1));
}
void VariableManager::RemoveParameter(IParameter* param)
{
    pimpl->_parameters.erase(param->GetTreeName());
    pimpl->_delete_connections.erase(param->GetTreeName());
}
std::vector<std::shared_ptr<IParameter>> VariableManager::GetOutputParameters(TypeInfo type)
{
    std::vector<std::shared_ptr<IParameter>> valid_outputs;
    for(auto itr = pimpl->_parameters.begin(); itr != pimpl->_parameters.end(); ++itr)
    {
        IParameter::Ptr param(itr->second);
        if(param)
        {
            if(param->GetTypeInfo() == type && param->CheckFlags(Output_e))
            {
                valid_outputs.push_back(param);
            }
        }
    }
    return valid_outputs;
}
std::vector<std::shared_ptr<IParameter>> VariableManager::GetAllParmaeters()
{
    std::vector<std::shared_ptr<IParameter>> output;
    for(auto& itr : pimpl->_parameters)
    {
        output.push_back(IParameter::Ptr(itr.second));
    }
    return output;
}
std::vector<std::shared_ptr<IParameter>> VariableManager::GetAllOutputParameters()
{
    std::vector<std::shared_ptr<IParameter>> output;
    for(auto& itr : pimpl->_parameters)
    {
        IParameter::Ptr param(itr.second);
        if(param && param->CheckFlags(Output_e))
        {
            output.push_back(param);
        }
        
    }
    return output;    
}
std::shared_ptr<IParameter> VariableManager::GetParameter(std::string name)
{
    return GetOutputParameter(name);
}

std::shared_ptr<IParameter> VariableManager::GetOutputParameter(std::string name)
{
    auto itr = pimpl->_parameters.find(name);
    if(itr != pimpl->_parameters.end())
    {
        return std::shared_ptr<IParameter>(itr->second);
    }
    // Check if the passed in value is the item specific name
    std::vector<std::shared_ptr<IParameter>> potentials;
    for(auto& itr : pimpl->_parameters)
    {
        if(auto pos = itr.first.find(name) != std::string::npos)
        {
            potentials.push_back(std::shared_ptr<IParameter>(itr.second));
        }
    }
    if(potentials.size())
    {
        if(potentials.size() > 1)
        {
            std::stringstream ss;
            for(auto potential : potentials)
                ss << potential->GetTreeName() << "\n";
            LOG(debug) << "Warning ambiguous name \"" << name << "\" passed in, multiple potential matches\n " << ss.str();
        }
        return potentials[0];
    }
    LOG(debug) << "Unable to find parameter named " << name;
    return nullptr;
}
void VariableManager::LinkParameters(std::weak_ptr<IParameter> output, std::weak_ptr<IParameter> input)
{
    if(auto input_param = std::dynamic_pointer_cast<InputParameter>(std::shared_ptr<IParameter>(input)))
        input_param->SetInput(IParameter::Ptr(output));
}