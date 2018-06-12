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

https://github.com/dtmoodie/MetaObject
*/
#pragma once
#include "ITAccessibleParam.hpp"
#include "MetaObject/params/MetaParam.hpp"
#include "ParamConstructor.hpp"

namespace mo
{
    template <typename T>
    class MO_EXPORTS TParam : virtual public ITAccessibleParam<T>
    {
      public:
        typedef T ValueType;
        typedef typename ParamTraits<T>::Storage_t Storage_t;
        typedef typename ParamTraits<T>::ConstStorageRef_t ConstStorageRef_t;
        typedef typename ParamTraits<T>::InputStorage_t InputStorage_t;
        typedef typename ParamTraits<T>::Input_t Input_t;

        static const ParamType Type = TParam_e;
        TParam(const std::string& name, const T& value) : IParam(name) { ParamTraits<T>::reset(_data, value); }
        TParam(const std::string& name) : IParam(name) {}
        TParam();

        virtual bool getData(InputStorage_t& data,
                             const OptionalTime_t& ts = OptionalTime_t(),
                             Context* ctx = nullptr,
                             size_t* fn_ = nullptr);

        virtual bool getData(InputStorage_t& data, size_t fn, Context* ctx = nullptr, OptionalTime_t* ts_ = nullptr);
        virtual AccessToken<T> access();
        virtual ConstAccessToken<T> read() const;
        bool canAccess() const override{return true;}

      protected:
        virtual bool updateDataImpl(const Storage_t& data,
                                    const OptionalTime_t& ts,
                                    Context* ctx,
                                    size_t fn,
                                    const std::shared_ptr<ICoordinateSystem>& cs);
        Storage_t _data;

      private:
        static ParamConstructor<TParam<T>> _typed_param_constructor;
        static MetaParam<T, 100> _meta_param;
    };
}
#include "detail/TParamImpl.hpp"
