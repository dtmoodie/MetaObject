#include <MetaObject/detail/TypeInfo.hpp>
#include <MetaObject/signals/ISlot.hpp>
#include <MetaObject/signals/Serialization.hpp>
#include <map>

using namespace mo;

struct RegisteredFunctions
{
    SignalSerializationFactory::call_function_f call;
    SignalSerializationFactory::signal_caller_constructor_f caller_constructor;
    SignalSerializationFactory::signal_sink_constructor_f sink_constructor;
};
struct SignalSerializationFactory::impl
{
    std::map<mo::TypeInfo, RegisteredFunctions> _registry;
};

SignalSerializationFactory::SignalSerializationFactory()
{
    _pimpl = new impl();
}

SignalSerializationFactory* SignalSerializationFactory::instance()
{
    static SignalSerializationFactory inst;
    return &inst;
}

SignalSerializationFactory::call_function_f SignalSerializationFactory::getTextFunction(ISlot* slot)
{
    auto itr = _pimpl->_registry.find(slot->getSignature());
    if (itr != _pimpl->_registry.end())
    {
        return itr->second.call;
    }
    return call_function_f();
}

ISignalCaller* SignalSerializationFactory::getTextFunctor(ISlot* slot)
{
    auto itr = _pimpl->_registry.find(slot->getSignature());
    if (itr != _pimpl->_registry.end())
    {
        return itr->second.caller_constructor(slot);
    }
    return nullptr;
}

void SignalSerializationFactory::setTextFunctions(ISlot* slot,
                                                  call_function_f function,
                                                  signal_caller_constructor_f caller_constructor,
                                                  signal_sink_constructor_f sink_constructor)
{
    _pimpl->_registry[slot->getSignature()] = {function, caller_constructor, sink_constructor};
}

ISignalCaller::~ISignalCaller()
{
}
ISignalSink::~ISignalSink()
{
}
