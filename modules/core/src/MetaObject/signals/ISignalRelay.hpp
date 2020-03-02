#ifndef MO_SIGNALS_ISIGNAL_RELAY_HPP
#define MO_SIGNALS_ISIGNAL_RELAY_HPP
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include <MetaObject/core/detail/forward.hpp>

#include <memory>
namespace mo
{
    class MO_EXPORTS ISignalRelay
    {
      public:
        using Ptr_t = std::shared_ptr<ISignalRelay>;
        using ConstPtr_t = std::shared_ptr<const ISignalRelay>;

        virtual ~ISignalRelay();
        virtual TypeInfo getSignature() const = 0;
        virtual bool hasSlots() const = 0;

      protected:
        friend class ISlot;
        friend class ISignal;
        template <class T, class E>
        friend class TSignal;
        template <class T>
        friend class TSlot;

        virtual bool connect(ISlot& slot) = 0;
        virtual bool connect(ISignal& signal) = 0;
        virtual bool disconnect(const ISlot& slot) = 0;
        virtual bool disconnect(const ISignal& signal) = 0;
    };
} // namespace mo
#endif // MO_SIGNALS_ISIGNAL_RELAY_HPP