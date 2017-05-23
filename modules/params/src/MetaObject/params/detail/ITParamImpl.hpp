#pragma once
#ifndef __CUDACC__
#include <boost/thread/recursive_mutex.hpp>
namespace mo{
    template<typename T> class ITParam;
    template<typename T>
    ITParam<T>::ITParam(const std::string& name,
                        ParamFlags flags,
                        OptionalTime_t ts,
                        Context* ctx,
                        size_t fn):
        IParam(name, flags, ts, ctx, fn){
    }

    template<typename T>
    const TypeInfo& ITParam<T>::getTypeInfo() const{
        return _type_info;
    }

    template<class T> template<class... Args>
    ITParam<T>* ITParam<T>::updateData(const Storage_t& data, const Args&... args) {
        size_t fn;
        const size_t* fnptr = GetKeywordInputOptional<tag::frame_number>(args...);
        if (fnptr)
            fn = *fnptr;
        else fn = this->_fn + 1;
        const mo::Time_t* ts = GetKeywordInputOptional<tag::timestamp>(args...);
        auto ctx = GetKeywordInputDefault<tag::context>(Context::getDefaultThreadContext(), args...);
        auto cs = GetKeywordInputDefault<tag::coordinate_system>(nullptr, args...);
        updateDataImpl(data, ts ? *ts : OptionalTime_t(), ctx, fn, cs);
        return this;
    }

    template<class T>
    std::shared_ptr<Connection> ITParam<T>::registerUpdateNotifier(TUpdateSlot_t* f){
        return f->connect(&_typed_update_signal);
    }

    template<class T>
    std::shared_ptr<Connection> ITParam<T>::registerUpdateNotifier(ISlot* f){
        if(f->getSignature() == TypeInfo(typeid(TUpdateSig_t))){
            auto typed = dynamic_cast<TUpdateSlot_t*>(f);
            if(typed){
                return registerUpdateNotifier(typed);
            }
        }else{
            return IParam::registerUpdateNotifier(f);
        }
        return {};
    }

    template<class T>
    std::shared_ptr<Connection> ITParam<T>::registerUpdateNotifier(std::shared_ptr<TSignalRelay<TUpdateSig_t>>& relay){
        return _typed_update_signal.connect(relay);
    }

    template<class T>
    std::shared_ptr<Connection> ITParam<T>::registerUpdateNotifier(std::shared_ptr<ISignalRelay> relay){
        auto typed = std::dynamic_pointer_cast<TSignalRelay<TUpdateSig_t>>(relay);
        if (typed) {
            return registerUpdateNotifier(typed);
        }else {
            return IParam::registerUpdateNotifier(relay);
        }
    }

    template<typename T>
    const TypeInfo ITParam<T>::_type_info(typeid(T));
}
#endif
