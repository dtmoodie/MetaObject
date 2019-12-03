#pragma once
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"

#include <memory>
namespace mo
{
    class ISlot;
    class ISignal;
    class Connection;
    template <class Sig>
    class TSlot;
    template <class Sig>
    class TSignal;

    class MO_EXPORTS ISignalRelay
    {
      public:
        using Ptr_t = std::shared_ptr<ISignalRelay>;
        using ConstPtr_t = std::shared_ptr<const ISignalRelay>;

        virtual ~ISignalRelay();
        virtual const TypeInfo& getSignature() const = 0;
        virtual bool hasSlots() const = 0;

      protected:
        friend class ISlot;
        friend class ISignal;
        template <class T>
        friend class TSignal;
        template <class T>
        friend class TSlot;

        virtual bool connect(ISlot* slot) = 0;
        virtual bool connect(ISignal* signal) = 0;
        virtual bool disconnect(ISlot* slot) = 0;
        virtual bool disconnect(ISignal* signal) = 0;
    };
}
