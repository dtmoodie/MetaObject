#include "Header.hpp"

namespace mo
{
    Header::Header()
        : ctx(nullptr)
    {
    }

    Header::Header(const mo::Time& ts)
        : timestamp(ts)
        , ctx(nullptr)
    {
    }
    Header::Header(const uint64_t fn)
        : frame_number(fn)
        , ctx(nullptr)
    {
    }

    bool Header::operator==(const Header& other) const
    {
        if (timestamp && other.timestamp)
        {
            return *timestamp == *other.timestamp;
        }
        else
        {
            return frame_number == other.frame_number;
        }
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
        else
        {
            return frame_number > other.frame_number;
        }
    }

    bool Header::operator<(const Header& other) const
    {
        if (timestamp && other.timestamp)
        {
            return *timestamp < *other.timestamp;
        }
        else
        {
            return frame_number < other.frame_number;
        }
    }
}
