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

https://github.com/dtmoodie/parameters
*/
#include "IParameter.hpp"

namespace mo
{

    template<class T>
    struct Stamped
    {
        Stamped(const T& data):
            data(data), fn(0){}

        Stamped(mo::time_t ts, const T& data):
            ts(ts), data(data){}

        Stamped(size_t fn, const T& data):
            fn(fn), data(data){}

        Stamped(mo::time_t ts, size_t fn, const T& data):
            ts(ts), fn(fn), data(data){}

        template<class...U>
        Stamped(mo::time_t ts, size_t frame_number, U...args) :
            data(args...)
        {
            this->ts = ts;
            this->fn = frame_number;
        }

        template<class A>
        void serialize(A& ar)
        {
            ar(ts);
            ar(fn);
            ar(*static_cast<T*>(this));
        }

        boost::optional<mo::time_t> ts;
        size_t fn;
        T data;
    };

    // The state struct is used to save a snapshot of the state of a parameter at a point in time
    template<class T>
    struct State: public Stamped<T>
    {

        State(mo::time_t ts, size_t fn, Context* ctx, ICoordinateSystem* cs, const T& init):
            Stamped<T>(ts, fn, init),
            ctx(ctx),
            cs(cs)
        {

        }

        State(mo::time_t ts, size_t fn, Context* ctx, const T& init):
            Stamped<T>(ts, fn, init),
            ctx(ctx),
            cs(nullptr)
        {

        }
        State(size_t fn, Context* ctx, ICoordinateSystem* cs, const T& init):
            Stamped<T>(fn, init),
            ctx(ctx),
            cs(cs)
        {

        }

        State(mo::time_t ts, size_t fn, const T&init):
            Stamped<T>(ts, fn, init),
            ctx(nullptr),
            cs(nullptr)
        {

        }

        State(mo::time_t ts, const T& init):
            Stamped<T>(ts, init),
            ctx(nullptr),
            cs(nullptr)
        {
        }

        State(const T& init):
            Stamped<T>(init),
            ctx(nullptr),
            cs(nullptr)
        {
        }
        Context* ctx;
        ICoordinateSystem* cs;
    };


    template<class T>
    class TUpdateToken;
    template<typename T>
    class MO_EXPORTS ITypedParameter : virtual public IParameter
    {
    public:
        typedef std::shared_ptr<ITypedParameter<T>> Ptr;
        typedef T ValueType;

        /*!
         * \brief ITypedParameter default constructor, passes args to IParameter
         */
        ITypedParameter(const std::string& name = "",
                        ParameterType flags = Control_e,
                        boost::optional<mo::time_t> ts = boost::optional<mo::time_t>(),
                        Context* ctx = nullptr,
                        size_t fn = 0);

        // The call is thread safe but the returned pointer may be modified by a different thread
        // ts is the timestamp for which you are requesting data, -1 indicates newest
        // ctx is the context of the data request, such as the thread of the object requesting the data
        /*!
         * \brief GetDataPtr returns a pointer to data at a requested timestamp. Not thread safe.
         * \param ts requested timestamp, default value returns newest data
         * \param ctx context for data to be located and synchronized
         * \param fn_ optional output frame number of requested data
         * \return pointer to data, nullptr if failed
         */
        virtual T*   GetDataPtr(boost::optional<mo::time_t> ts = boost::optional<mo::time_t>(),
                                Context* ctx = nullptr, size_t* fn_ = nullptr) = 0;
        /*!
         * \brief GetDataPtr returns a pointer to data at a requested frame number. Not thread safe
         * \param fn requested frame number
         * \param ctx context for data to be located, used for event stream synchronization
         *        between parameter's _ctx and input ctx
         * \param ts_ optional output timestamp of requested data
         * \return pointer to data, nullptr if failed
         */
        virtual T*   GetDataPtr(size_t fn, Context* ctx = nullptr, boost::optional<mo::time_t>* ts_ = nullptr) = 0;

        // Copies data into value
        // Time index is the index for which you are requesting data
        // ctx is the context of the data request, such as the thread of the object requesting the data
        /*!
         * \brief GetData returns a copy of the data at the requested timestamp. Thread safe
         * \param ts requested timestamp
         * \param ctx context for synchronization
         * \param fn optional output frame number of requested data
         * \return copy of data, throws exception if access fails
         */
        virtual T    GetData(boost::optional<mo::time_t> ts = boost::optional<mo::time_t>(),
                             Context* ctx = nullptr, size_t* fn = nullptr) = 0;

        /*!
         * \brief GetData returns a copy of the data at requested frame number. Thread safe
         * \param fn requested frame number
         * \param ctx is the context of the destination data used for synchronization
         * \param ts is the optional output timestamp of requested data
         * \return copy of data at requested frame number, throws exception on failure
         */
        virtual T    GetData(size_t fn, Context* ctx = nullptr, boost::optional<mo::time_t>* ts = nullptr) = 0;

        /*!
         * \brief GetData copies data into value. Thread safe
         * \param value reference to location of output
         * \param ts requested timestamp, default value is newest
         * \param ctx context for resulting data for synchronization
         * \param fn optional output frame number of requested data
         * \return true on success, false otherwise
         */
        virtual bool GetData(T& value, boost::optional<mo::time_t> ts = boost::optional<mo::time_t>(),
                             Context* ctx = nullptr, size_t* fn = nullptr) = 0;
        /*!
         * \brief GetData copies data into value. Thread safe
         * \param value return value
         * \param fn requested frame number
         * \param ctx context of returned data, used for synchronization
         * \param ts optional output timestamp of data
         * \return true on success, false otherwise
         */
        virtual bool GetData(T& value, size_t fn, Context* ctx = nullptr, boost::optional<mo::time_t>* ts = nullptr) = 0;

        /*!
         * \brief UpdateData used to update the parameter and emit update signals.
         *        Calls commit with provided values
         * \param data is the new data value
         * \param ts timestamp of new data
         * \param fn frame number of new data
         * \param ctx context of new data
         * \param cs coordinate system of new data
         * \return pointer to this parameter
         */
        template<class... Args>
        ITypedParameter<T>* UpdateData(const T& data, const Args&... args)
        {
            boost::optional<size_t> fn;
            const size_t* fnptr = GetKeywordInputOptional<tag::frame_number>(args...);
            if(fnptr)
                fn = *fnptr;
            const mo::time_t* ts = GetKeywordInputOptional<tag::timestamp>(args...);
            UpdateDataImpl(data, ts ? *ts : boost::optional<mo::time_t>(),
                                 GetKeywordInputDefault<tag::context>(nullptr, args...),
                                 fn, GetKeywordInputDefault<tag::coordinate_system>(nullptr, args...));
            return this;
        }


        //virtual TUpdateToken<T> Update();

        const TypeInfo& GetTypeInfo() const;

        virtual bool Update(IParameter* other);
    protected:
        virtual bool UpdateDataImpl(const T& data, boost::optional<mo::time_t> ts, Context* ctx, boost::optional<size_t> fn, ICoordinateSystem* cs) = 0;
    private:
        static const TypeInfo _type_info;
    };

    template<class T>
    class TUpdateToken
    {
    public:
        TUpdateToken(ITypedParameter<T>& param):
            _param(param),
            _ts(-1 * mo::second),
            _fn(std::numeric_limits<size_t>::max()),
            _cs(nullptr),
            _ctx(nullptr)
        {
        }

        ~TUpdateToken()
        {
            //_param.UpdateData(_data, _ts, _ctx, _fn, _cs);
        }

        TUpdateToken& operator()(T&& data)
        {
            _data = std::forward<T>(data);
        }

        TUpdateToken& operator()(const T& data)
        {
            _data = data;
        }

        TUpdateToken& operator()(const mo::time_t& ts)
        {
            _ts = ts;
            return *this;
        }

        TUpdateToken& operator()(mo::time_t&& ts)
        {
            _ts = std::forward<mo::time_t>(ts);
            return *this;
        }

        TUpdateToken& operator()(size_t fn)
        {
            _fn = fn;
            return *this;
        }
        TUpdateToken& operator()(Context* ctx)
        {
            _ctx = ctx;
            return *this;
        }

        TUpdateToken& operator()(ICoordinateSystem* cs)
        {
            _cs = cs;
            return *this;
        }

    private:
        T _data;
        ITypedParameter<T>& _param;
        size_t _fn;
        mo::time_t _ts;
        ICoordinateSystem* _cs;
        Context* _ctx;
    };
}
#include "detail/ITypedParameterImpl.hpp"
