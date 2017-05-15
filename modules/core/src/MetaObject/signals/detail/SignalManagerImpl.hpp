#pragma once
#include "MetaObject/logging/Log.hpp"
namespace mo
{
    //class SignalManager;
    /*template<typename T> class TSignal;

    template<typename T> std::weak_ptr<TSignal<T>> SignalManager::getSignalOptional(const std::string& name)
	{
		std::lock_guard<std::mutex> lock(mtx);
		auto signature = TypeInfo(typeid(T));
        if(!exists(name, signature))
            return std::weak_ptr<TSignal<T>>();
		auto& sig = getSignal(name, signature);
		if (!sig)
		{
			LOG(debug) << this << " Creating signal " << name << " <" << signature.name() << ">";
			sig.reset(new TSignal<T>());
		}
		return std::weak_ptr<TSignal<T>>(std::dynamic_pointer_cast<TSignal<T>>(sig));
	}
    template<typename T> std::weak_ptr<TSignal<T>> SignalManager::getSignal(const std::string& name)
    {
        std::lock_guard<std::mutex> lock(mtx);
		auto signature = TypeInfo(typeid(T));
        auto&sig = getSignal(name, signature);
		if (!sig)
		{
			LOG(debug) << this << " Creating signal " << name << " <" << signature.name() << ">";
			sig.reset(new TSignal<T>());
		}
        return std::weak_ptr<TSignal<T>>(std::dynamic_pointer_cast<TSignal<T>>(sig));
    }
	*/
}