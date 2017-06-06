#pragma once
/*
Copyright (c) 2015 Daniel Moodie.
All rights reserved.

Redistribution and use in source and binary forms are permitted
provided that the above copyright notice and this paragraph are
duplicated in all such forms and that any documentation,
advertising materials, and other materials related to such
distribution and use acknowledge that the software was developed
by the Daniel Moodie. The name of
Daniel Moodie may not be used to endorse or promote products derived
from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

https://github.com/dtmoodie/Params
*/
#include "IParam.hpp"
#include "traits/TypeTraits.hpp"

namespace mo {
template <class T>
struct Stamped {
    Stamped(typename ParamTraits<T>::ConstStorageRef_t data)
        : data(data)
        , fn(0) {
    }

    explicit Stamped(mo::Time_t ts, typename ParamTraits<T>::ConstStorageRef_t data)
        : ts(ts)
        , data(data) {
    }

    explicit Stamped(size_t fn, typename ParamTraits<T>::ConstStorageRef_t data)
        : fn(fn)
        , data(data) {
    }

    explicit Stamped(mo::Time_t ts, size_t fn, typename ParamTraits<T>::ConstStorageRef_t data)
        : ts(ts)
        , fn(fn)
        , data(data) {
    }

    template <class... U>
    Stamped(mo::Time_t ts, size_t frame_number, U... args)
        : data(args...) {
        this->ts = ts;
        this->fn = frame_number;
    }
    Stamped& operator=(const Stamped<T>& other) {
        this->data = other.data;
        this->ts   = other.ts;
        this->fn   = other.fn;
        return *this;
    }
    template <class A>
    void serialize(A& ar) {
        ar(ts);
        ar(fn);
        ar(*static_cast<T*>(this));
    }

    OptionalTime_t                                                                                                     ts;
    size_t                                                                                                             fn;
    typename std::remove_const<typename std::remove_reference<typename ParamTraits<T>::ConstStorageRef_t>::type>::type data;
};

// The state struct is used to save a snapshot of the state of a Param at a point in time
template <class T>
struct State : public Stamped<T> {
    State(mo::Time_t ts, size_t fn, Context* ctx, ICoordinateSystem* cs, typename ParamTraits<T>::ConstStorageRef_t init)
        : Stamped<T>(ts, fn, init)
        , ctx(ctx)
        , cs(cs) {
    }

    State(mo::Time_t ts, size_t fn, Context* ctx, typename ParamTraits<T>::ConstStorageRef_t init)
        : Stamped<T>(ts, fn, init)
        , ctx(ctx)
        , cs(nullptr) {
    }

    State(size_t fn, Context* ctx, ICoordinateSystem* cs, typename ParamTraits<T>::ConstStorageRef_t init)
        : Stamped<T>(fn, init)
        , ctx(ctx)
        , cs(cs) {
    }

    State(mo::Time_t ts, size_t fn, typename ParamTraits<T>::ConstStorageRef_t init)
        : Stamped<T>(ts, fn, init)
        , ctx(nullptr)
        , cs(nullptr) {
    }

    State(mo::Time_t ts, typename ParamTraits<T>::ConstStorageRef_t init)
        : Stamped<T>(ts, init)
        , ctx(nullptr)
        , cs(nullptr) {
    }

    State(typename ParamTraits<T>::ConstStorageRef_t init)
        : Stamped<T>(init)
        , ctx(nullptr)
        , cs(nullptr) {
    }

    State& operator=(const State<T>& other) {
        this->data = other.data;
        this->ts   = other.ts;
        this->fn   = other.fn;
        this->ctx  = other.ctx;
        this->cs   = other.cs;
        return *this;
    }
    Context*           ctx;
    ICoordinateSystem* cs;
};

template <typename T>
class AccessToken;
template <typename T, typename U>
using enable_if_storage_differs_t = typename std::enable_if<!std::is_same<T, U>::value && !std::is_same<T, typename ParamTraits<T>::Storage_t>::value, ITParam<T>*>::type;

template <typename T>
class MO_EXPORTS ITParamImpl : virtual public IParam {
public:
    typedef std::shared_ptr<ITParam<T> >               Ptr;
    typedef typename ParamTraits<T>::Storage_t         Storage_t;
    typedef typename ParamTraits<T>::ConstStorageRef_t ConstStorageRef_t;
    typedef typename ParamTraits<T>::InputStorage_t    InputStorage_t;
    typedef typename ParamTraits<T>::Input_t           Input_t;
    typedef void(TUpdateSig_t)(ConstStorageRef_t, IParam*, Context*, OptionalTime_t, size_t, ICoordinateSystem*, UpdateFlags);
    typedef TSignal<TUpdateSig_t> TUpdateSignal_t;
    typedef TSlot<TUpdateSig_t>   TUpdateSlot_t;

    virtual bool getData(InputStorage_t& data, const OptionalTime_t& ts = OptionalTime_t(),
                         Context* ctx = nullptr, size_t* fn_ = nullptr)
        = 0;

    virtual bool getData(InputStorage_t& data, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr) = 0;

    virtual const TypeInfo&             getTypeInfo() const;
    virtual std::shared_ptr<Connection> registerUpdateNotifier(ISlot* f);
    virtual std::shared_ptr<Connection> registerUpdateNotifier(std::shared_ptr<ISignalRelay> relay);
    std::shared_ptr<Connection> registerUpdateNotifier(TUpdateSlot_t* f);
    std::shared_ptr<Connection> registerUpdateNotifier(std::shared_ptr<TSignalRelay<TUpdateSig_t> >& relay);

    template <class... Args>
    ITParamImpl<T>* updateData(const Storage_t& data, const Args&... args) {
        size_t             fn;
        OptionalTime_t     ts;
        Context*           ctx   = nullptr;
        ICoordinateSystem* cs    = nullptr;
        const IParam*      param = GetKeywordInputOptional<tag::param>(args...);
        if (param) {
            fn  = param->getFrameNumber();
            ctx = param->getContext();
            cs  = param->getCoordinateSystem();
            ts  = param->getTimestamp();
        }
        const size_t* fnptr = GetKeywordInputOptional<tag::frame_number>(args...);
        if (fnptr)
            fn = *fnptr;
        else if (param == nullptr)
            fn = this->_fn + 1;

        if (auto tsptr = GetKeywordInputOptional<tag::timestamp>(args...))
            ts = *tsptr;
        if (auto ctx_ = GetKeywordInputDefault<tag::context>(nullptr, args...))
            ctx = ctx_;
        if (auto cs_ = GetKeywordInputDefault<tag::coordinate_system>(nullptr, args...))
            cs = cs_;
        this->updateDataImpl(data, ts, ctx, fn, cs);
        return this;
    }

protected:
    friend class AccessToken<T>;
    TSignal<TUpdateSig_t> _typed_update_signal;
    virtual bool updateDataImpl(const Storage_t& data, OptionalTime_t ts, Context* ctx, size_t fn, ICoordinateSystem* cs) = 0;

private:
    static const TypeInfo _type_info;
};

template <typename T, bool S>
using storage_update_selector_t = typename std::enable_if<std::is_same<typename ParamTraits<T>::Storage_t, T>::value == S>::type;

template <typename T>
class MO_EXPORTS ITParam<T, storage_update_selector_t<T, false> > : virtual public ITParamImpl<T> {
public:
    typedef T                                          ValueType;
    typedef std::shared_ptr<ITParam<T> >               Ptr;
    typedef typename ParamTraits<T>::Storage_t         Storage_t;
    typedef typename ParamTraits<T>::ConstStorageRef_t ConstStorageRef_t;
    typedef typename ParamTraits<T>::InputStorage_t    InputStorage_t;
    typedef typename ParamTraits<T>::Input_t           Input_t;
    typedef void(TUpdateSig_t)(ConstStorageRef_t, IParam*, Context*, OptionalTime_t, size_t, ICoordinateSystem*, UpdateFlags);
    typedef TSignal<TUpdateSig_t> TUpdateSignal_t;
    typedef TSlot<TUpdateSig_t>   TUpdateSlot_t;
    // brief ITParam default constructor, passes args to IParam
    ITParam(const std::string& name = "", ParamFlags flags = Control_e,
            OptionalTime_t ts = OptionalTime_t(), Context* ctx = nullptr, size_t fn = 0)
        : IParam(name, flags, ts, ctx, fn) {
    }

    template <class... Args>
    ITParam<T>* updateData(const T& data, const Args&... args) {
        size_t    fn;
        Storage_t tmp;
        ParamTraits<T>::reset(tmp, data);
        ITParamImpl<T>::updateData(tmp, args...);
        return this;
    }
};

template <typename T>
class MO_EXPORTS ITParam<T, storage_update_selector_t<T, true> > : virtual public ITParamImpl<T> {
public:
    typedef std::shared_ptr<ITParam<T> >               Ptr;
    typedef typename ParamTraits<T>::Storage_t         Storage_t;
    typedef typename ParamTraits<T>::ConstStorageRef_t ConstStorageRef_t;
    typedef typename ParamTraits<T>::InputStorage_t    InputStorage_t;
    typedef typename ParamTraits<T>::Input_t           Input_t;
    typedef void(TUpdateSig_t)(ConstStorageRef_t, IParam*, Context*, OptionalTime_t, size_t, ICoordinateSystem*, UpdateFlags);
    typedef TSignal<TUpdateSig_t> TUpdateSignal_t;
    typedef TSlot<TUpdateSig_t>   TUpdateSlot_t;
    // brief ITParam default constructor, passes args to IParam
    ITParam(const std::string& name = "", ParamFlags flags = Control_e,
            OptionalTime_t ts = OptionalTime_t(), Context* ctx = nullptr, size_t fn = 0)
        : IParam(name, flags, ts, ctx, fn) {
    }
};
}
#include "detail/ITParamImpl.hpp"
