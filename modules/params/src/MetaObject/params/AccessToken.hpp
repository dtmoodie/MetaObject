#pragma once
#include "MetaObject/params/IParam.hpp"
#include "traits/TypeTraits.hpp"

namespace boost
{
    template <typename T>
    class lock_guard;
}

namespace mo
{
    struct MO_EXPORTS AccessTokenLock
    {
        AccessTokenLock();
        AccessTokenLock(AccessTokenLock&& other);
        AccessTokenLock(mo::Mutex_t& mtx);
        ~AccessTokenLock();
        std::unique_ptr<boost::lock_guard<mo::Mutex_t>> lock;
    };

    // Guarantees write safe access to underlying data
    template <typename T>
    class MO_EXPORTS AccessToken
    {
      public:
        AccessToken(AccessToken<T>&& other)
            : lock(std::move(other.lock)), _param(other._param), _data(other._data), fn(other.fn), ts(other.ts),
              _ctx(other._ctx), valid(other.valid)
        {
        }

        AccessToken(ITParam<T>& param, typename ParamTraits<T>::Storage_t& data)
            : lock(param.mtx()), _param(param), _data(ParamTraits<T>::get(data))
        {
        }

        template <class U>
        AccessToken(
            ITParam<T>& param,
            U& data,
            typename std::enable_if<!std::is_same<typename ParamTraits<U>::Storage_t, U>::value>::type* dummy = 0)
            : lock(param.mtx()), _param(param), _data(data)
        {
        }

        ~AccessToken()
        {
            if (valid)
            {
                _param.emitUpdate(ts, _ctx, fn);
            }
        }

        T& operator()()
        {
            valid = true;
            return _data;
        }

        const T& operator()() const { return _data; }

        AccessToken<T>& operator()(const OptionalTime_t& ts_)
        {
            ts(ts_);
            return *this;
        }

        AccessToken<T>& operator()(const boost::optional<size_t>& fn_)
        {
            fn(fn_);
            return *this;
        }

        AccessToken<T>& operator()(Context* ctx)
        {
            _ctx = ctx;
            return *this;
        }

        void setValid(bool value) { valid = value; }

        bool getValid() const { return valid; }
      private:
        AccessTokenLock lock;
        ITParam<T>& _param;
        typename ParamTraits<T>::TypeRef_t _data;
        OptionalTime_t ts;
        boost::optional<size_t> fn;
        Context* _ctx = nullptr;
        bool valid = false;
    };

    template <class T>
    struct ConstAccessToken
    {
        ConstAccessToken(const ITParam<T>& param, typename ParamTraits<T>::ConstTypeRef_t data)
            : lock(param.mtx()), _param(param), _data(ParamTraits<T>::get(data))
        {
        }

        const T& operator()() const { return _data; }

        AccessTokenLock lock;
        const ITParam<T>& _param;
        typename ParamTraits<T>::ConstTypeRef_t _data;
    };
}
