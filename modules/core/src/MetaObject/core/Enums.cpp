#include "MetaObject/core/detail/Enums.hpp"
#include "MetaObject/logging/logging.hpp"
#include <map>
#include <vector>

using namespace mo;

#define TYPE_NAME_HELPER(name)                                                                                         \
    {                                                                                                                  \
        ParamFlags::name##_e, #name                                                                                    \
    }

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

std::string mo::paramFlagsToString(EnumClassBitset<ParamFlags> type)
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

EnumClassBitset<ParamFlags> mo::stringToParamFlags(const std::string& str)
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

std::string mo::paramTypeToString(BufferFlags flags)
{
    switch (flags)
    {
    case CircularBuffer_e:
        return "circularbuffer";
    case Map_e:
        return "map";
    case StreamBuffer_e:
        return "StreamBuffer";
    case BlockingStreamBuffer_e:
        return "BlockingStreamBuffer";
    case NNStreamBuffer_e:
        return "NNStreamBuffer";
    case Queue_e:
        return "Queue";
    case BlockingQueue_e:
        return "BlockingQueue";
    case DroppingQueue_e:
        return "DroppingQueue";
    case ForceDirectConnection_e:
        return "ForceDirectConnection";
    case ForceBufferedConnection_e:
        return "ForceBufferedConnection";
    }
    return "";
}

BufferFlags mo::stringToParamType(const std::string& str)
{
    if (str == "circularbuffer")
        return CircularBuffer_e;
    else if (str == "map")
        return Map_e;
    else if (str == "StreamBuffer")
        return StreamBuffer_e;
    else if (str == "BlockingStreamBuffer")
        return BlockingStreamBuffer_e;
    else if (str == "NNStreamBuffer")
        return NNStreamBuffer_e;
    THROW(debug) << "Invalid string " << str;
    return TParam_e;
}

static std::map<const Context*, std::map<const Context*, ParamType>> connection_map;
static ParamType default_connection_type = BlockingStreamBuffer_e;

ParamType mo::getDefaultBufferType(const Context* source, const Context* dest)
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

void mo::setDefaultBufferType(const Context* source, const Context* dest, ParamType type)
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
