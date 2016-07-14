#pragma once
#include "MetaObject/Logging/Log.hpp"
namespace mo
{
    class SignalManager;
    template<typename T> class TypedSignal;

    template<typename T> TypedSignal<T>* SignalManager::GetSignalOptional(const std::string& name)
	{
		std::lock_guard<std::mutex> lock(mtx);
		auto signature = TypeInfo(typeid(T));
        if(!exists(name, signature))
            return nullptr;
		auto& sig = GetSignal(name, signature);
		if (!sig)
		{
			LOG(debug) << this << " Creating signal " << name << " <" << signature.name() << ">";
			sig.reset(new TypedSignal<T>());
		}
		return std::dynamic_pointer_cast<TypedSignal<T>>(sig).get();
	}
    template<typename T> TypedSignal<T>* SignalManager::GetSignal(const std::string& name)
    {
        std::lock_guard<std::mutex> lock(mtx);
		auto signature = TypeInfo(typeid(T));
        auto&sig = GetSignal(name, signature);
		if (!sig)
		{
			LOG(debug) << this << " Creating signal " << name << " <" << signature.name() << ">";
			sig.reset(new TypedSignal<T>());
		}
        return std::dynamic_pointer_cast<TypedSignal<T>>(sig).get();
    }
}