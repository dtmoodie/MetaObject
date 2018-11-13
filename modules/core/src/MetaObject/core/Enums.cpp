#include "MetaObject/core/detail/Enums.hpp"
#include "MetaObject/logging/logging.hpp"
#include <map>
#include <vector>

#define TYPE_NAME_HELPER(name)                                                                                         \
    {                                                                                                                  \
        ParamFlags::name##_e, #name                                                                                    \
    }

namespace mo
{

    namespace
    {
        static const std::vector<std::pair<ParamFlags, std::string>> type_flag_map = {TYPE_NAME_HELPER(None),
                                                                                      TYPE_NAME_HELPER(Input),
                                                                                      TYPE_NAME_HELPER(Output),
                                                                                      TYPE_NAME_HELPER(State),
                                                                                      TYPE_NAME_HELPER(Control),
                                                                                      TYPE_NAME_HELPER(Buffer),
                                                                                      TYPE_NAME_HELPER(Optional),
                                                                                      TYPE_NAME_HELPER(Unstamped),
                                                                                      TYPE_NAME_HELPER(Source),
                                                                                      TYPE_NAME_HELPER(Sync),
                                                                                      TYPE_NAME_HELPER(RequestBuffered),
                                                                                      TYPE_NAME_HELPER(Dynamic),
                                                                                      TYPE_NAME_HELPER(Source)};

        static std::map<const IContext*, std::map<const IContext*, BufferFlags>> connection_map;
        static BufferFlags default_connection_type = BLOCKING_STREAM_BUFFER;
    }

    std::string paramFlagsToString(EnumClassBitset<ParamFlags> type)
    {
        std::string output;
        for (const auto& itr : type_flag_map)
        {
            if (type.test(itr.first))
            {
                if (output.empty())
                {
                    output = itr.second;
                }
                else
                {
                    output += "|" + itr.second;
                }
            }
        }
        return output;
    }

    EnumClassBitset<ParamFlags> stringToParamFlags(const std::string& str)
    {
        std::string rest = str;
        EnumClassBitset<ParamFlags> output;
        auto pos = rest.find('|');
        while (pos != std::string::npos)
        {
            std::string substr = rest.substr(0, pos);
            rest = rest.substr(pos + 1);
            for (const auto& itr : type_flag_map)
            {
                if (substr == itr.second)
                    // output = ParamFlags(itr.first | output);
                    output.flip(itr.first);
            }
            pos = rest.find('|');
        }
        for (const auto& itr : type_flag_map)
        {
            if (rest == itr.second)
                // output = ParamFlags(itr.first | output);
                output.flip(itr.first);
        }
        return output;
    }

    std::string BufferFlagsToString(BufferFlags flags)
    {
        switch (flags)
        {
        case CIRCULAR_BUFFER:
            return "CircularBuffer";
        case MAP_BUFFER:
            return "Map";
        case STREAM_BUFFER:
            return "StreamBuffer";
        case BLOCKING_STREAM_BUFFER:
            return "BlockingStreamBuffer";
        case DROPPING_STREAM_BUFFER:
            return "DroppingStreamBuffer";
        case NEAREST_NEIGHBOR_BUFFER:
            return "NNStreamBuffer";
        case QUEUE_BUFFER:
            return "Queue";
        case BLOCKING_QUEUE_BUFFER:
            return "BlockingQueue";
        case DROPPING_QUEUE_BUFFER:
            return "DroppingQueue";
        }
        return "";
    }

    BufferFlags stringToBufferFlags(const std::string& str)
    {
        if (str == "CircularBuffer")
            return CIRCULAR_BUFFER;
        else if (str == "Map")
            return MAP_BUFFER;
        else if (str == "StreamBuffer")
            return STREAM_BUFFER;
        else if (str == "BlockingStreamBuffer")
            return BLOCKING_STREAM_BUFFER;
        else if (str == "NNStreamBuffer")
            return NEAREST_NEIGHBOR_BUFFER;
        THROW(debug) << "Invalid string " << str;
        return DEFAULT;
    }

    BufferFlags getDefaultBufferType(const IContext* source, const IContext* dest)
    {
        auto itr = connection_map.find(source);
        if (itr != connection_map.end())
        {
            auto itr2 = itr->second.find(dest);
            if (itr2 != itr->second.end())
            {
                return itr2->second;
            }
            itr2 = itr->second.find(nullptr);
            if (itr2 != itr->second.end())
            {
                return itr2->second;
            }
        }
        return default_connection_type;
    }

    void setDefaultBufferType(const IContext* source, const IContext* dest, BufferFlags type)
    {
        if (source == nullptr && dest == nullptr)
        {
            default_connection_type = type;
        }
        else
        {
            connection_map[source][dest] = type;
        }
    }
}
