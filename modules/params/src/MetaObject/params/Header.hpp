#pragma once
#include <MetaObject/core/detail/Time.hpp>
#include <ct/reflect.hpp>
#include <ct/reflect_macros.hpp>

#include <memory>

namespace mo
{
    struct IAsyncStream;
    struct ICoordinateSystem;
    using ICoordinateSystemPtr_t = std::shared_ptr<ICoordinateSystem>;
    using ICoordinateSystemConstPtr_t = std::shared_ptr<const ICoordinateSystem>;

    struct MO_EXPORTS Header
    {
        Header(mo::OptionalTime ts = mo::OptionalTime(), FrameNumber fn = FrameNumber());
        Header(const Time& ts);
        Header(const Duration& d);
        Header(FrameNumber fn);
        Header(uint64_t fn);
        Header(Header&&) = default;
        Header(const Header&) = default;
        Header& operator=(const Header&) = default;
        Header& operator=(Header&&) = default;
        ~Header() = default;

        mo::OptionalTime timestamp;
        FrameNumber frame_number;

        IAsyncStream* stream;
        ICoordinateSystemPtr_t coordinate_system;

        bool operator==(const Header& other) const;
        bool operator!=(const Header& other) const;
        bool operator>(const Header& other) const;
        bool operator<(const Header& other) const;
    };
}

namespace ct
{
    REFLECT_BEGIN(mo::Header)
        PUBLIC_ACCESS(timestamp)
        PUBLIC_ACCESS(frame_number)
    REFLECT_END;
}
