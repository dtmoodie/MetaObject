#pragma once
#include <MetaObject/core/detail/Time.hpp>
#include <memory>

namespace mo
{
    struct Context;
    struct ICoordinateSystem;
    using ICoordinateSystemPtr_t = std::shared_ptr<ICoordinateSystem>;
    using ICoordinateSystemConstPtr_t = std::shared_ptr<const ICoordinateSystem>;

    struct Header
    {
        mo::OptionalTime_t         timestamp;
        uint64_t                   frame_number = std::numeric_limits<uint64_t>::max();

        Context*                   ctx;
        ICoordinateSystemPtr_t  coordinate_system;
    };
}
