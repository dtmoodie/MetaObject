#include "Header.hpp"
#include <ct/reflect_traits.hpp>

namespace mo
{
    bool SourceId::operator==(const SourceId& other) const
    {
        return other.id == this->id;
    }

    bool SourceId::operator!=(const SourceId& other) const
    {
        return other.id != this->id;
    }

    Header::Header(mo::OptionalTime ts, FrameNumber fn)
        : timestamp(std::move(ts))
        , frame_number(fn)
    {
    }

    Header::Header(const mo::Time& ts)
        : timestamp(ts)
    {
    }

    Header::Header(const Duration& d)
        : Header(Time(d))
    {
    }

    Header::Header(FrameNumber fn)
        : frame_number(fn)
    {
    }

    Header::Header(uint64_t fn)
        : frame_number(fn)
    {
    }

    bool Header::operator==(const Header& other) const
    {
        if (other.source_id != this->source_id)
        {
            return false;
        }
        if (timestamp && other.timestamp)
        {
            return *timestamp == *other.timestamp;
        }

        return frame_number == other.frame_number;
    }

    bool Header::operator!=(const Header& other) const
    {
        return !(*this == other);
    }

    bool Header::operator>(const Header& other) const
    {
        if (timestamp && other.timestamp)
        {
            return *timestamp > *other.timestamp;
        }
        return frame_number > other.frame_number;
    }

    bool Header::operator<(const Header& other) const
    {
        if (timestamp && other.timestamp)
        {
            return *timestamp < *other.timestamp;
        }
        return frame_number < other.frame_number;
    }

    static_assert(ct::IsReflected<Header>::value, "");
} // namespace mo
