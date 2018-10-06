#include "MetaObject/params/InputParamAny.hpp"
#include "IDataContainer.hpp"
#include <boost/thread/locks.hpp>
#include <boost/thread/recursive_mutex.hpp>
using namespace mo;

InputParamAny::InputParamAny(const std::string& name)
    : _update_slot(std::bind(&InputParamAny::on_param_update, this, std::placeholders::_1, std::placeholders::_2))
    , _delete_slot(std::bind(&InputParamAny::on_param_delete, this, std::placeholders::_1))
{
    this->setName(name);
    _void_type_info = mo::TypeInfo(typeid(void));
    this->appendFlags(mo::ParamFlags::Input_e);
}

bool InputParamAny::getInputData(const Header& desired, Header* retrieved)
{
    Lock lock(this->mtx());
    if (input)
    {
        auto data = input->getData(desired);
        if (data)
        {
            if (retrieved)
            {
                *retrieved = data->getHeader();
            }
            m_data = data;
            return true;
        }
    }
    return false;
}

size_t InputParamAny::getInputFrameNumber()
{
    if (input)
    {
        return input->getFrameNumber();
    }
    return 0;
}

OptionalTime_t InputParamAny::getInputTimestamp()
{
    if (input)
    {
        return input->getTimestamp();
    }
    return {};
}

IParam* InputParamAny::getInputParam() const
{
    return input;
}

bool InputParamAny::isInputSet() const
{
    return input != nullptr;
}

bool InputParamAny::setInput(const std::shared_ptr<mo::IParam>& param)
{
    if (setInput(param.get()))
    {
        m_shared_input = param;
        return true;
    }
    return false;
}

bool InputParamAny::setInput(mo::IParam* param)
{
    input = param;
    param->registerDeleteNotifier(&_delete_slot);
    param->registerUpdateNotifier(&_update_slot);
    emitUpdate();
    return true;
}

bool InputParamAny::acceptsInput(std::weak_ptr<mo::IParam> param) const
{
    return true;
}

bool InputParamAny::acceptsInput(mo::IParam* param) const
{
    return true;
}

bool InputParamAny::acceptsType(const mo::TypeInfo& type) const
{
    return true;
}

mo::TypeInfo InputParamAny::getTypeInfo() const
{
    if (input)
    {
        return input->getTypeInfo();
    }
    return _void_type_info;
}

void InputParamAny::on_param_update(mo::Context* ctx, mo::IParam* param)
{
}

void InputParamAny::on_param_delete(mo::IParam const*)
{
    input = nullptr;
}

TypeInfo InputParamAny::_void_type_info;
