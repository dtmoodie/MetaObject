#include "parameters/ParameteredObject.h"
#include "parameters/ParameteredObjectImpl.hpp"
#include "parameters/IVariableManager.h"

#include "signals/signal_manager.h"

#include <signals/logging.hpp>

using namespace Parameters;
CEREAL_REGISTER_TYPE(ParameteredObject);


ParameteredObject::ParameteredObject()
{
    //_sig_parameter_updated = nullptr;
    //_sig_parameter_added = nullptr;
    _variable_manager = nullptr;
}

ParameteredObject::~ParameteredObject()
{
    _callback_connections.clear();
    if(_variable_manager)
    {
        for(int i = 0; i < _parameters.size(); ++i)
        {
            _variable_manager->RemoveParameter(_parameters[i]);
        }
    }
    _parameters.clear();
}

void ParameteredObject::SetupVariableManager(std::shared_ptr<IVariableManager> manager)
{
    _variable_manager = manager;
    if (_variable_manager)
    {
        for (auto param : _parameters)
        {
            _variable_manager->AddParameter(param);
        }
    }
}

std::shared_ptr<Parameters::IVariableManager> ParameteredObject::GetVariableManager()
{
    return _variable_manager;
}

Parameter* ParameteredObject::addParameter(Parameter::Ptr param)
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    //DOIF_LOG_FAIL(_sig_parameter_added, (*_sig_parameter_updated)(this), warning);
    sig_parameter_added(this);
    DOIF_LOG_FAIL(_variable_manager, _variable_manager->AddParameter(param.get()), debug);
    _callback_connections.push_back(param->RegisterUpdateNotifier(std::bind(&ParameteredObject::onUpdate, this, param.get(), std::placeholders::_1)));
    _parameters.push_back(param.get());
    _implicit_parameters.push_back(param);
    return param.get();
}

Parameter* ParameteredObject::addParameter(Parameter* param)
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    // Check if it already exists
    auto itr = std::find(_explicit_parameters.begin(), _explicit_parameters.end(), param);
    if (itr != _explicit_parameters.end())
    {
        // an implicit parameter already exists with this exact pointer, do nothing
        return param;
    }
    auto existing_param = getParameterOptional(param->GetName());
    if (existing_param != nullptr)
    {
        // Parameter has already been added, either implicitly or explicitly, do nothing
        LOG(debug) << "Parameter with name " << param->GetName() << " already exists but not as an explicitly defined parameter";
        return param;
    }
    //DOIF_LOG_FAIL(_sig_parameter_added, (*_sig_parameter_updated)(this), debug);
    sig_parameter_added(this);
    DOIF_LOG_FAIL(_variable_manager, _variable_manager->AddParameter(param), debug);
    
    _callback_connections.push_back(param->RegisterUpdateNotifier(std::bind(&ParameteredObject::onUpdate, this, param, std::placeholders::_1)));
    _parameters.push_back(param);
    _explicit_parameters.push_back(param);
    return param;
}


Parameter* updateParameter(std::shared_ptr<Parameter> parameter)
{

    return parameter.get();
}
void ParameteredObject::RemoveParameter(std::string name)
{
    for(auto itr = _parameters.begin(); itr != _parameters.end(); ++itr)
    {
        if((*itr)->GetName() == name)
        {
            DOIF_LOG_FAIL(_variable_manager, _variable_manager->RemoveParameter(*itr), debug);
            itr = _parameters.erase(itr);
        }
    }
}

void ParameteredObject::RemoveParameter(size_t index)
{
    if(index < _parameters.size())
    {
        DOIF_LOG_FAIL(_variable_manager, _variable_manager->RemoveParameter(_parameters[index]), debug);
        _parameters.erase(_parameters.begin() + index);
    }
}

Parameter* ParameteredObject::getParameter(int idx)
{
    CV_Assert(idx >= 0 && idx < _parameters.size());
    return _parameters[idx];
}

Parameter* ParameteredObject::getParameter(const std::string& name)
{
    for (auto& itr : _parameters)
    {
        if (itr->GetName() == name)
        {
            return itr;
        }
    }    
    throw std::string("Unable to find parameter by name: " + name);
}

Parameter* ParameteredObject::getParameterOptional(int idx)
{
    if (idx < 0 || idx >= _parameters.size())
    {
        LOG(debug) << "Requested index " << idx << " out of bounds " << _parameters.size();
        return nullptr;
    }
    return _parameters[idx];
}

Parameter* ParameteredObject::getParameterOptional(const std::string& name)
{
    for (auto& itr : _parameters)
    {
        if (itr->GetName() == name)
        {
            return itr;
        }
    }
    for(auto& itr : _explicit_parameters)
    {
        if(itr->GetName() == name)
        {
            return itr;
        }
    }
    for(auto& itr : _implicit_parameters)
    {
        if(itr->GetName() == name)
        {
            return itr.get();
        }
    }
    LOG(trace) << "Unable to find parameter by name: " << name;
    return nullptr;
}

void ParameteredObject::InitializeExplicitParams()
{
    InitializeExplicitParamsToDefault();
    WrapExplicitParams();
}

void ParameteredObject::InitializeExplicitParamsToDefault()
{
}

void ParameteredObject::WrapExplicitParams()
{
}

std::vector<Parameter*> ParameteredObject::getParameters()
{
    return _parameters;
}

std::vector<Parameter*> ParameteredObject::getDisplayParameters()
{
    return getParameters();
}

void ParameteredObject::RegisterParameterCallback(int idx, const Parameter::update_f& callback, bool lock_param, bool lock_object)
{
    RegisterParameterCallback(getParameter(idx), callback, lock_param, lock_object);
}

void ParameteredObject::RegisterParameterCallback(const std::string& name, const Parameter::update_f& callback, bool lock_param, bool lock_object)
{
    RegisterParameterCallback(getParameter(name), callback, lock_param, lock_object);
}

void ParameteredObject::RegisterParameterCallback(Parameter* param, const Parameter::update_f& callback, bool lock_param, bool lock_object)
{
    if (lock_param && !lock_object)
    {
        _callback_connections.push_back(param->RegisterUpdateNotifier(std::bind(&ParameteredObject::RunCallbackLockParameter, this, std::placeholders::_1, callback, &param->mtx())));
        return;
    }
    if (lock_object && !lock_param)
    {
        _callback_connections.push_back(param->RegisterUpdateNotifier(std::bind(&ParameteredObject::RunCallbackLockObject, this, std::placeholders::_1, callback)));
        return;
    }
    if (lock_object && lock_param)
    {

        _callback_connections.push_back(param->RegisterUpdateNotifier(std::bind(&ParameteredObject::RunCallbackLockBoth, this, std::placeholders::_1, callback, &param->mtx())));
        return;
    }
    _callback_connections.push_back(param->RegisterUpdateNotifier(callback));    
}

void ParameteredObject::onUpdate(Parameters::Parameter* param, cv::cuda::Stream* stream)
{
    if(!(param->flags & kOutput))
        sig_parameter_updated(this);
}

void ParameteredObject::RunCallbackLockObject(Signals::context* ctx, const Parameter::update_f& callback)
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    callback(ctx);
}

void ParameteredObject::RunCallbackLockParameter(Signals::context* ctx, const Parameter::update_f& callback, std::recursive_mutex* paramMtx)
{
    std::lock_guard<std::recursive_mutex> lock(*paramMtx);
    callback(ctx);
}

void ParameteredObject::RunCallbackLockBoth(Signals::context* ctx, const Parameter::update_f& callback, std::recursive_mutex* paramMtx)
{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    std::lock_guard<std::recursive_mutex> lock_(*paramMtx);
    callback(ctx);
}

bool ParameteredObject::exists(const std::string& name)
{
    return getParameterOptional(name) != nullptr;
}

bool ParameteredObject::exists(size_t index)
{
    return index < _parameters.size();
}
std::vector<ParameterInfo*> ParameteredObject::getParameterInfo() const
{
    return std::vector<ParameterInfo*>();
}