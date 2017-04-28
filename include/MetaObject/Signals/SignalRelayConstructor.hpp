#pragma once
#include "MetaObject/Signals/RelayFactory.hpp"
#include "MetaObject/Detail/TypeInfo.hpp"
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