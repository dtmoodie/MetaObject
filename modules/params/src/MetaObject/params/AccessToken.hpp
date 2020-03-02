#pragma once
#include "traits/TypeTraits.hpp"
#include <MetaObject/core/detail/forward.hpp>
#include <MetaObject/params/Header.hpp>

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
    struct MO_EXPORTS AccessToken
    {
        using type = typename TParam<T>::type;
        AccessToken(AccessToken<T>&& other) = default;
        AccessToken(const AccessToken<T>& other) = delete;

        AccessToken& operator=(const AccessToken&) = delete;
        AccessToken& operator=(AccessToken&&) = default;

        AccessToken(Lock_t&& lock, TParam<T>& param, type& data)
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

        type& operator()()
        {
            _modified = true;
            return _data;
        }

        const type& operator()() const
        {
            return _data;
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
        type& _data;
        Header _header;
        bool _modified = false;
    };

    template <class T>
    struct ConstAccessToken
    {
        using type = typename TParam<T>::type;
        ConstAccessToken(ConstAccessToken&& other) = default;
        ConstAccessToken(const ConstAccessToken& other) = default;

        ConstAccessToken(Lock_t&& lock, const TParam<T>& param, const type& data)
            : _lock(std::move(lock))
            , _param(param)
            , _data(data)
        {
        }

        ConstAccessToken& operator=(const ConstAccessToken&) = default;
        ConstAccessToken& operator=(ConstAccessToken&&) = default;

        const type& operator()() const
        {
            return _data;
        }

        AccessTokenLock _lock;
        const TParam<T>& _param;
        const type& _data;
    };
} // namespace mo
