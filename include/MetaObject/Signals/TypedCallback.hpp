#pragma once
#include "ICallback.hpp"
#include <functional>

namespace mo
{
    class ISlot;
    template<class Sig> class TypedCallback: public ICallback{};

    template<class R, class ... T> class TypedCallback<R(T...)>: public ICallback, public std::function<R(T...)>
    {
    public:
        TypedCallback(const std::function<R(T...)& f);
        R operator()(T... args);
        TypeInfo GetSignature() const;
        void Disconnect();
    };
}

#include "detail/TypedCallbackImpl.hpp"