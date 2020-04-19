#pragma once
#include "IParam.hpp"
#include "MetaObject/detail/Export.hpp"
#include <MetaObject/core/detail/forward.hpp>

namespace mo
{
    struct MO_EXPORTS IPublisher : virtual IParam
    {
        virtual ~IPublisher();

        virtual bool providesOutput(const TypeInfo type) const = 0;
        virtual std::vector<TypeInfo> getOutputTypes() const = 0;
        virtual std::vector<Header> getAvailableHeaders() const = 0;
        virtual boost::optional<Header> getNewestHeader() const = 0;

        // Fetch the current data from the publisher at the desired header
        // If desired is nullptr, then retrieve the current data with no attempts to match the timestamp
        // if stream is nullptr, do not do any synchronization
        virtual IDataContainerConstPtr_t getData(const Header* desired = nullptr, IAsyncStream* = nullptr) = 0;

        virtual uint32_t getNumSubscribers() const = 0;
        virtual void setAllocator(Allocator::Ptr_t alloc) = 0;
    };
} // namespace mo
