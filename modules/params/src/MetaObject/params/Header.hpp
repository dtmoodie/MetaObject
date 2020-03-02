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

    struct MO_EXPORTS SourceId
    {
        bool operator==(const SourceId& other) const;
        bool operator!=(const SourceId& other) const;
        uint64_t id = 0;
    };

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
        SourceId source_id;

        bool operator==(const Header& other) const;
        bool operator!=(const Header& other) const;
        bool operator>(const Header& other) const;
        bool operator<(const Header& other) const;
    };
} // namespace mo

namespace ct
{
    REFLECT_BEGIN(mo::Header)
        PUBLIC_ACCESS(timestamp)
        PUBLIC_ACCESS(frame_number)
        PUBLIC_ACCESS(source_id)
    REFLECT_END;
} // namespace ct
