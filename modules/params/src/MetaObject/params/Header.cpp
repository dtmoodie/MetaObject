#include "Header.hpp"

namespace mo
{
     bool Header::operator==(const Header& other)
     {
        if(timestamp && other.timestamp)
        {
            return *timestamp == *other.timestamp;
        }else
        {
            return frame_number == other.frame_number;
        }
     }
    bool Header::operator!=(const Header& other)
    {
        return !(*this == other);
    }

    bool Header::operator>(const Header& other)
    {
        if(timestamp && other.timestamp)
        {
            return *timestamp > *other.timestamp;
        }else
        {
            return frame_number > other.frame_number;
        }
    }

    bool Header::operator<(const Header& other)
    {
        if(timestamp && other.timestamp)
        {
            return *timestamp < *other.timestamp;
        }else
        {
            return frame_number < other.frame_number;
        }
    }
}
