#pragma once
#include "MetaObject/detail/TypeInfo.hpp"
#include "MetaObject/signals/RelayFactory.hpp"

namespace mo
{
    template <class Sig>
    class SignalRelayConstructor
    {
      public:
        SignalRelayConstructor()
        {
            RelayFactory::instance()->registerCreator([]() -> ISignalRelay* { return new TSignalRelay<Sig>(); },
                                                      TypeInfo(typeid(Sig)));
        }
    };
}
