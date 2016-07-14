#pragma once

namespace mo
{
    template<class R, class...T> class TypedCallback;

    template<class R, class...T> R TypedCallback<R(T...)>::operator()(T... args)
    {
        return std::function<R(T...)>::operator()(args...);
    }
    template<class R, class...T> TypeInfo TypedCallback<R(T...)>::GetSignature() const
    {
        return TypeInfo(typeid(R(T...)));
    }
    template<class R, class...T>  void TypedCallback<R(T...)>::Disconnect()
    {
        *this = TypedCallback<R(T...)>();
    }
}