#pragma once
#include <MetaObject/core/detail/Time.hpp>
#include <memory>

namespace mo
{
    struct Context;
    struct ICoordinateSystem;
    using ICoordinateSystemPtr_t = std::shared_ptr<ICoordinateSystem>;
    using ICoordinateSystemConstPtr_t = std::shared_ptr<const ICoordinateSystem>;

    struct MO_EXPORTS Header
    {
        Header();
        Header(const mo::Time& ts);
        Header(const uint64_t fn);
        Header(Header&&) = default;
        Header(const Header&) = default;
        Header& operator=(const Header&) = default;
        Header& operator=(Header&&) = default;

        mo::OptionalTime timestamp;
        uint64_t frame_number;

        Context* ctx;
        ICoordinateSystemPtr_t coordinate_system;

        bool operator==(const Header& other);
        bool operator!=(const Header& other);
        bool operator>(const Header& other);
        bool operator<(const Header& other);
    };
}
