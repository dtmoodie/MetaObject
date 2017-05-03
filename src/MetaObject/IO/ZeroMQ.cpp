#include "MetaObject/IO/ZeroMQ.hpp"
using namespace mo;

#ifdef HAVE_ZEROMQ
#include "zmq.hpp"
#include "MetaObject/IO/Message.hpp"
#include "MetaObject/IO/ZeroMQImpl.hpp"


ZeroMQContext::ZeroMQContext()
{
    _pimpl = new ZeroMQContext::impl();
}
ZeroMQContext* ZeroMQContext::Instance()
{
    static ZeroMQContext g_inst;
    
    return &g_inst;
}

struct ParamPublisher::impl
{
    std::shared_ptr<IParam> shared_input;
    IParam* input;
};

ParamPublisher::ParamPublisher()
{
    _pimpl = new impl();
}

ParamPublisher::~ParamPublisher()
{
    delete _pimpl;
}

bool ParamPublisher::getInput(mo::Time_t ts)
{
    return false;
}

// This gets a pointer to the variable that feeds into this input
IParam* ParamPublisher::getInputParam()
{
    return nullptr;
}

// Set input and setup callbacks
bool ParamPublisher::setInput(std::shared_ptr<IParam> param)
{
    return false;
}

bool ParamPublisher::setInput(IParam* param)
{
    return false;
}

// Check for correct serialization routines, etc
bool ParamPublisher::AcceptsInput(std::weak_ptr<IParam> param) const
{
    return false;
}
bool ParamPublisher::AcceptsInput(IParam* param) const
{
    return false;
}
bool ParamPublisher::AcceptsType(TypeInfo type) const
{
    return false;
}

#else

ZeroMQContext::ZeroMQContext()
{

}

ZeroMQContext* ZeroMQContext::Instance()
{
    return nullptr;
}

struct ParamPublisher::impl
{
    
};

ParamPublisher::ParamPublisher()
{
    
}

ParamPublisher::~ParamPublisher()
{
    delete _pimpl;
}

bool ParamPublisher::getInput(const OptionalTime_t& time)
{
    return false;
}

// This gets a pointer to the variable that feeds into this input
IParam* ParamPublisher::getInputParam()
{
    return nullptr;
}

// Set input and setup callbacks
bool ParamPublisher::setInput(std::shared_ptr<IParam> param)
{
    return false;
}

bool ParamPublisher::setInput(IParam* param)
{
    return false;
}

// Check for correct serialization routines, etc
bool ParamPublisher::AcceptsInput(std::weak_ptr<IParam> param) const
{
    return false;
}
bool ParamPublisher::AcceptsInput(IParam* param) const
{
    return false;
}
bool ParamPublisher::AcceptsType(TypeInfo type) const
{
    return false;
}



#endif