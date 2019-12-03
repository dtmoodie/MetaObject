#include <MetaObject/core/detail/Enums.hpp>
#include <MetaObject/logging/logging.hpp>
#include <ct/enum.hpp>
#include <ct/reflect.hpp>

#include <map>

namespace mo
{
    namespace
    {
        static std::map<const IAsyncStream*, std::map<const IAsyncStream*, BufferFlags>> connection_map;
        static BufferFlags default_connection_type = BufferFlags::BLOCKING_STREAM_BUFFER;
    }

    static_assert(ct::Flags::CT_RESERVED_FLAG_BITS + 1 < 10, "");
    static_assert(ct::Flags::CT_RESERVED_FLAG_BITS + 1 < 10, "");

    std::string paramFlagsToString(ParamFlags type)
    {
        std::stringstream out;
        out << type;
        return std::move(out).str();
    }

    ParamFlags stringToParamFlags(const std::string& str)
    {
        return ct::bitsetFromString<ParamFlags>(str);
    }

    std::string bufferFlagsToString(BufferFlags flags)
    {
        std::stringstream ss;
        ss << flags;
        return std::move(ss).str();
    }

    BufferFlags stringToBufferFlags(const std::string& str)
    {
        auto flag = ct::fromString<BufferFlags>(str);
        if (!flag.success())
        {
            THROW(debug, "Invalid string {}", str);
        }
        return flag.value();
    }

    BufferFlags getDefaultBufferType(const IAsyncStream* source, const IAsyncStream* dest)
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

    void setDefaultBufferType(const IAsyncStream* source, const IAsyncStream* dest, BufferFlags type)
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
