#ifndef MO_PARAMS_IBUFFER_HPP
#define MO_PARAMS_IBUFFER_HPP

#include "MetaObject/core/detail/Enums.hpp"
#include "MetaObject/core/detail/Time.hpp"
#include "MetaObject/detail/Export.hpp"
#include "MetaObject/params/IPublisher.hpp"
#include "MetaObject/params/ISubscriber.hpp"
namespace mo
{
    namespace buffer
    {
        class MO_EXPORTS IBuffer : virtual public ISubscriber, virtual public IPublisher
        {
          public:
            static std::shared_ptr<IBuffer> create(BufferFlags type);

            virtual ~IBuffer();
            virtual void setFrameBufferCapacity(const uint64_t size) = 0;
            virtual void setTimePaddingCapacity(const Duration& time) = 0;
            virtual boost::optional<uint64_t> getFrameBufferCapacity() const = 0;
            virtual boost::optional<Duration> getTimePaddingCapacity() const = 0;

            virtual std::vector<Header> getAvailableHeaders() const = 0;

            // These are not const accessors because I may need to lock a mutex inside of them.
            virtual uint64_t getSize() const = 0;
            virtual uint64_t clear() = 0;
            virtual bool getTimestampRange(mo::OptionalTime& start, mo::OptionalTime& end) = 0;
            virtual bool getFrameNumberRange(uint64_t& start, uint64_t& end) = 0;
            virtual BufferFlags getBufferType() const = 0;

            virtual ConnectionPtr_t registerUpdateNotifier(ISlot&) = 0;
            virtual ConnectionPtr_t registerUpdateNotifier(const ISignalRelay::Ptr_t& relay) = 0;
        };
    } // namespace buffer
} // namespace mo
#endif // MO_PARAMS_IBUFFER_HPP