#pragma once
#include "MetaObject/params/IParam.hpp"
#include "traits/TypeTraits.hpp"

namespace mo
{

    struct MO_EXPORTS AccessTokenLock
    {
        AccessTokenLock();
        AccessTokenLock(AccessTokenLock&& other);
        AccessTokenLock(Lock_t&& lock);
        AccessTokenLock(Mutex_t& lock);
        ~AccessTokenLock();

      private:
        std::unique_ptr<Lock_t> lock;
    };

    // Guarantees write safe access to underlying data
    template <typename T>
    class MO_EXPORTS AccessToken
    {
      public:
        AccessToken(AccessToken<T>&& other) = default;
        AccessToken(const AccessToken<T>& other) = delete;

        AccessToken& operator=(const AccessToken&) = delete;
        AccessToken& operator=(AccessToken&&) = default;

        AccessToken(Lock_t&& lock, TParam<T>& param, T& data)
            : _lock(std::move(lock))
            , _param(param)
            , _data(data)
        {
        }

        ~AccessToken()
        {
            if (_modified)
            {
                _param.IParam::emitUpdate(std::move(_header));
            }
        }

        T& operator()()
        {
            _modified = true;
            return _data;
        }

        const T& operator()() const
        {
            return _data;
        }

        AccessToken<T>& operator()(const OptionalTime& ts_)
        {
            _header.timestamp = ts_;
            _modified = true;
            return *this;
        }

        AccessToken<T>& operator()(const uint64_t fn_)
        {
            _header.frame_number = fn_;
            _modified = true;
            return *this;
        }

        AccessToken<T>& operator()(IAsyncStream* ctx)
        {
            _header.stream = ctx;
            return *this;
        }

        void setModified(bool value)
        {
            _modified = value;
        }

        bool isModified() const
        {
            return _modified;
        }

      private:
        AccessTokenLock _lock;
        TParam<T>& _param;
        T& _data;
        Header _header;
        bool _modified = false;
    };

    template <class T>
    struct ConstAccessToken
    {
        ConstAccessToken(ConstAccessToken&& other) = default;
        ConstAccessToken(const ConstAccessToken& other) = default;

        ConstAccessToken(Lock_t&& lock, const TParam<T>& param, const T& data)
            : _lock(std::move(lock))
            , _param(param)
            , _data(data)
        {
        }

        ConstAccessToken& operator=(const ConstAccessToken&) = default;
        ConstAccessToken& operator=(ConstAccessToken&&) = default;

        const T& operator()() const
        {
            return _data;
        }

        AccessTokenLock _lock;
        const TParam<T>& _param;
        const T& _data;
    };
}
