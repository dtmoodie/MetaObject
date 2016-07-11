#include "parameters/VariableManager.h"
#include "parameters/Parameter.hpp"
#include "parameters/InputParameter.hpp"

#include <signals/logging.hpp>
using namespace Parameters;

void VariableManager::AddParameter(Parameters::Parameter* param)
{
    _parameters[param->GetTreeName()] = param;
    _delete_connections[param->GetTreeName()] = param->RegisterDeleteNotifier(std::bind(&VariableManager::RemoveParameter, this, std::placeholders::_1));
}
void VariableManager::RemoveParameter(Parameters::Parameter* param)
{
    _parameters.erase(param->GetTreeName());
    _delete_connections.erase(param->GetTreeName());
}
std::vector<Parameters::Parameter*> VariableManager::GetOutputParameters(Loki::TypeInfo type)
{
    std::vector<Parameters::Parameter*> valid_outputs;
    for(auto itr = _parameters.begin(); itr != _parameters.end(); ++itr)
    {
        if(itr->second->GetTypeInfo() == type && itr->second->flags & kOutput)
        {
            valid_outputs.push_back(itr->second);
        }
    }
    return valid_outputs;
}
std::vector<Parameters::Parameter*> VariableManager::GetAllParmaeters()
{
    std::vector<Parameters::Parameter*> output;
    for(auto& itr : _parameters)
    {
        output.push_back(itr.second);
    }
    return output;
}
std::vector<Parameters::Parameter*> VariableManager::GetAllOutputParmaeters()
{
    std::vector<Parameters::Parameter*> output;
    for(auto& itr : _parameters)
    {
        if(itr.second->flags & kOutput)
        {
            output.push_back(itr.second);
        }
        
    }
    return output;    
}
Parameters::Parameter* VariableManager::GetParameter(std::string name)
{
    return GetOutputParameter(name);
}

Parameters::Parameter* VariableManager::GetOutputParameter(std::string name)
{
    auto itr = _parameters.find(name);
    if(itr != _parameters.end())
    {
        return itr->second;
    }
    // Check if the passed in value is the item specific name
    std::vector<Parameter*> potentials;
    for(auto& itr : _parameters)
    {
        if(auto pos = itr.first.find(name) != std::string::npos)
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
                ss << potential->GetTreeName() << "\n";
            LOG(debug) << "Warning ambiguous name \"" << name << "\" passed in, multiple potential matches\n " << ss.str();
        }
        return potentials[0];
    }
    LOG(debug) << "Unable to find parameter named " << name;
    return nullptr;
}
void VariableManager::LinkParameters(Parameters::Parameter* output, Parameters::Parameter* input)
{
    if(auto input_param = dynamic_cast<Parameters::InputParameter*>(input))
        input_param->SetInput(output);
}