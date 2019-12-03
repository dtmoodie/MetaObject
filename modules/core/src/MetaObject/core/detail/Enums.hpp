#ifndef MO_CORE_ENUMS_HPP
#define MO_CORE_ENUMS_HPP
#include <MetaObject/detail/Export.hpp>

#include <ct/flags.hpp>
#include <ct/reflect/MemberObjectPointer.hpp>

#include <bitset>
#include <string>
namespace mo
{
    struct IAsyncStream;

    BITSET_BEGIN(ParamFlags)
        ENUM_BITVALUE(kINPUT, 0)
        ENUM_BITVALUE(kOUTPUT, 1)
        ENUM_BITVALUE(kSTATE, 2)
        ENUM_BITVALUE(kCONTROL, 3)
        ENUM_BITVALUE(kBUFFER, 4)
        ENUM_BITVALUE(kOPTIONAL, 5)
        ENUM_BITVALUE(kDESYNCED, 6)
        ENUM_BITVALUE(kUNSTAMPED, 7)
        ENUM_BITVALUE(kSYNC, 8)
        ENUM_BITVALUE(kREQUIRE_BUFFERED, 9)
        ENUM_BITVALUE(kSOURCE, 10)
        ENUM_BITVALUE(kDYNAMIC, 11)
        ENUM_BITVALUE(kOWNS_MUTEX, 12)
    ENUM_END;

    // TODO figure out how to use the above ParamFlags instead
    BITSET_BEGIN(ParamReflectionFlags)
        ENUM_BITVALUE(kCONTROL, ct::Flags::CT_RESERVED_FLAG_BITS + 1)
        ENUM_BITVALUE(kSTATE, kCONTROL + 1)
        ENUM_BITVALUE(kSTATUS, kSTATE + 1)
        ENUM_BITVALUE(kINPUT, kSTATUS + 1)
        ENUM_BITVALUE(kOUTPUT, kINPUT + 1)
        ENUM_BITVALUE(kOPTIONAL, kOUTPUT + 1)
        ENUM_BITVALUE(kSOURCE, kOPTIONAL + 1)
        ENUM_BITVALUE(kSIGNAL, kSOURCE + 1)
        ENUM_BITVALUE(kSLOT, kSIGNAL + 1)
    ENUM_END;

    ENUM_BEGIN(UpdateFlags, int)
        ENUM_VALUE(kVALUE_UPDATED, 0)
        ENUM_VALUE(kINPUT_SET, 1)
        ENUM_VALUE(kINPUT_CLEARED, 2)
        ENUM_VALUE(kINPUT_UPDATED, 3)
        ENUM_VALUE(kBUFFER_UPDATED, 4)
    ENUM_END;

    MO_EXPORTS std::string paramFlagsToString(ParamFlags flags);
    MO_EXPORTS ParamFlags stringToParamFlags(const std::string& str);

    ENUM_BEGIN(BufferFlags, int)
        ENUM_VALUE(DEFAULT, 0)
        ENUM_VALUE(CIRCULAR_BUFFER, DEFAULT + 1)
        ENUM_VALUE(MAP_BUFFER, CIRCULAR_BUFFER + 1)
        ENUM_VALUE(STREAM_BUFFER, MAP_BUFFER + 1)
        ENUM_VALUE(BLOCKING_STREAM_BUFFER, STREAM_BUFFER + 1)
        ENUM_VALUE(DROPPING_STREAM_BUFFER, BLOCKING_STREAM_BUFFER + 1)
        ENUM_VALUE(NEAREST_NEIGHBOR_BUFFER, DROPPING_STREAM_BUFFER + 1)
        ENUM_VALUE(QUEUE_BUFFER, NEAREST_NEIGHBOR_BUFFER + 1)
        ENUM_VALUE(BLOCKING_QUEUE_BUFFER, QUEUE_BUFFER + 1)
        ENUM_VALUE(DROPPING_QUEUE_BUFFER, BLOCKING_QUEUE_BUFFER + 1)

        ENUM_BITVALUE(FORCE_BUFFERED, 10)
        ENUM_BITVALUE(FORCE_DIRECT, FORCE_BUFFERED + 1)
        ENUM_BITVALUE(SOURCE, FORCE_DIRECT + 1)
    ENUM_END;

    MO_EXPORTS std::string bufferFlagsToString(BufferFlags type);
    MO_EXPORTS BufferFlags stringToBufferFlags(const std::string& str);

    MO_EXPORTS BufferFlags getDefaultBufferType(const IAsyncStream* source, const IAsyncStream* dest);
    MO_EXPORTS void setDefaultBufferType(const IAsyncStream* source, const IAsyncStream* dest, BufferFlags type);
} // namespace mo

namespace std
{
    template <>
    struct underlying_type<mo::ParamFlags>
    {
        using type = ct::Flag_t;
    };
} // namespace std
#endif // MO_CORE_ENUMS_HPP