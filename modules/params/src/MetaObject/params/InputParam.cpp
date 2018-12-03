#include "MetaObject/params/InputParam.hpp"
#include <boost/fiber/recursive_timed_mutex.hpp>
#include <boost/thread/locks.hpp>
using namespace mo;

InputParam::InputParam()
    : IParam("", mo::ParamFlags::Input_e)
    , m_input_param(nullptr)
{
    m_delete_slot = std::bind(&InputParam::onInputDelete, this, std::placeholders::_1);
    m_update_slot = std::bind(
        &InputParam::onInputUpdate, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
}

InputParam::~InputParam()
{
    if (m_input_param)
    {
        m_input_param->unsubscribe();
    }
}

InputParam::IContainerPtr_t InputParam::getData(const Header& desired)
{
    InputParam::IContainerPtr_t data;
    if (m_input_param)
    {
        data = m_input_param->getData(desired);
    }
    m_current_data = data;
    return data;
}

InputParam::IContainerConstPtr_t InputParam::getData(const Header& desired) const
{
    InputParam::IContainerPtr_t data;
    if (m_current_data)
    {
        if (m_current_data->getHeader() == desired)
        {
            data = m_current_data;
        }
    }

    if (data == nullptr && m_input_param)
    {
        data = m_input_param->getData(desired);
    }
    m_current_data = data;
    return data;
}

std::ostream& InputParam::print(std::ostream& os) const
{
    IParam* input = nullptr;
    {
        Lock_t lock(mtx());
        IParam::print(os);
        input = getInputParam();
    }

    if (input)
    {
        os << "\n";
        input->print(os);
    }
    return os;
}

OptionalTime InputParam::getInputTimestamp()
{
    Lock_t lock(mtx());
    if (m_input_param)
    {
        return m_input_param->getTimestamp();
    }
    return {};
}

uint64_t InputParam::getInputFrameNumber()
{
    Lock_t lock(mtx());
    if (m_input_param)
    {
        return m_input_param->getFrameNumber();
    }
    return std::numeric_limits<uint64_t>::max();
}

bool InputParam::acceptsInput(IParam*) const
{
    return true;
}

bool InputParam::acceptsType(const TypeInfo&) const
{
    return true;
}

void InputParam::setQualifier(const Qualifier_f& f)
{
    Lock_t lock(mtx());
    qualifier = f;
}

void InputParam::onInputDelete(const IParam*)
{
    Lock_t lock(mtx());
    m_input_param = nullptr;
}

bool InputParam::setInput(const std::shared_ptr<IParam>& param)
{
    Lock_t lock(mtx());
    if (setInput(param.get()))
    {
        m_shared_input = param;
        return true;
    }
    return false;
}

bool InputParam::setInput(IParam* param)
{
    {
        Lock_t lock(mtx());
        if (qualifier && !qualifier(param))
        {
            return false;
        }
        if (m_input_param)
        {
            m_input_param->unsubscribe();
            m_update_slot.clear();
            m_delete_slot.clear();
        }
        m_input_param = param;
    }
    param->subscribe();
    param->registerUpdateNotifier(&m_update_slot);
    param->registerDeleteNotifier(&m_delete_slot);
    emitUpdate(Header(), UpdateFlags::InputSet_e);
    return true;
}

bool InputParam::isInputSet() const
{
    return m_input_param != nullptr;
}

IParam* InputParam::getInputParam() const
{
    Lock_t lock(IParam::mtx());
    return m_input_param;
}

TypeInfo InputParam::getTypeInfo() const
{
    Lock_t lock(IParam::mtx());
    if (m_input_param)
    {
        return m_input_param->getTypeInfo();
    }
    return TypeInfo::Void();
}

void InputParam::visit(IReadVisitor& visitor)
{
    IDataContainerPtr_t data;
    {
        Lock_t lock(mtx());
        data = m_current_data;
    }
    if (data)
    {
        data->visit(visitor);
    }
}

void InputParam::visit(IWriteVisitor& visitor) const
{
    IDataContainerPtr_t data;
    {
        Lock_t lock(mtx());
        data = m_current_data;
    }
    if (data)
    {
        data->visit(visitor);
    }
}

void InputParam::visit(BinaryInputVisitor& ar)
{
    IDataContainerPtr_t data;
    {
        Lock_t lock(mtx());
        data = m_current_data;
    }
    if (data)
    {
        data->visit(ar);
    }
}

void InputParam::visit(BinaryOutputVisitor& ar) const
{
    IDataContainerPtr_t data;
    {
        Lock_t lock(mtx());
        data = m_current_data;
    }
    if (data)
    {
        data->visit(ar);
    }
}

void InputParam::onInputUpdate(const IDataContainerPtr_t& data, IParam* param, UpdateFlags)
{
    if (data->getHeader().stream == getStream())
    {
        m_current_data = data;
        emitUpdate(data, InputUpdated_e);
    }
    else
    {
        // mer figure out what do
        if (param->checkFlags(ParamFlags::Buffer_e))
        {
            emitUpdate(data, BufferUpdated_e);
        }
    }
}
