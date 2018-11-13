#pragma once
#include <MetaObject/core/detail/Time.hpp>
#include <memory>

namespace mo
{
    struct IContext;
    struct ICoordinateSystem;
    using ICoordinateSystemPtr_t = std::shared_ptr<ICoordinateSystem>;
    using ICoordinateSystemConstPtr_t = std::shared_ptr<const ICoordinateSystem>;

    struct MO_EXPORTS Header
    {
        Header();
        Header(const Time& ts);
        Header(const Duration& d);
        Header(const uint64_t fn);
        Header(Header&&) = default;
        Header(const Header&) = default;
        Header& operator=(const Header&) = default;
        Header& operator=(Header&&) = default;

        mo::OptionalTime timestamp;
        FrameNumber frame_number;

        IContext* ctx;
        ICoordinateSystemPtr_t coordinate_system;

        bool operator==(const Header& other) const;
        bool operator!=(const Header& other) const;
        bool operator>(const Header& other) const;
        bool operator<(const Header& other) const;
    };
}
