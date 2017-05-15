#pragma once
#include "MetaObject/signals/RelayFactory.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
namespace mo
{
    template<class Sig> class SignalRelayConstructor
    {
    public:
        SignalRelayConstructor()
        {
            RelayFactory::Instance()->RegisterCreator(
                []()->ISignalRelay*
            {
                return new TSignalRelay<Sig>();
            }, TypeInfo(typeid(Sig)));
        }
    };
}