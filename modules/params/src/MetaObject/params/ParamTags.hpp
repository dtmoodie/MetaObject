#pragma once
#include <MetaObject/detail/Export.hpp>

#include "MetaObject/core/detail/Enums.hpp"
#include "MetaObject/params/NamedParam.hpp"
#include <MetaObject/core/detail/Time.hpp>

#include <memory>

namespace mo
{
    struct IAsyncStream;
    struct ICoordinateSystem;
    class IParam;

    /*MO_KEYWORD_INPUT(frame_number, uint64_t)
    MO_KEYWORD_INPUT(coordinate_system, const std::shared_ptr<ICoordinateSystem>)
    MO_KEYWORD_INPUT(stream, IAsyncStream*)
    MO_KEYWORD_INPUT(param_name, std::string)
    MO_KEYWORD_INPUT(tree_root, std::string)
    MO_KEYWORD_INPUT(param_flags, ParamFlags)*/

    namespace tags
    {
        struct Timestamp
        {
        };

        struct FrameNumber
        {
        };

        struct Stream
        {
        };
        struct Name
        {
        };
        struct Flags
        {
        };
        struct Param
        {
        };
    } // namespace tags
    namespace params
    {
        using Timestamp = TNamedParam<tags::Timestamp, Time>;
        using FrameNumber = TNamedParam<tags::FrameNumber, uint64_t>;
        using Stream = TNamedParam<tags::Stream, IAsyncStream*>;
        using Name = TNamedParam<tags::Name, std::string>;
        using Flags = TNamedParam<tags::Flags, ParamFlags>;
        using Param = TNamedParam<tags::Param, const IParam*>;

    }
    
    static constexpr TKeyword<params::Timestamp> timestamp;
    static constexpr TKeyword<params::FrameNumber> fn;
    static constexpr TKeyword<params::Stream> stream;
    static constexpr TKeyword<params::Name> name;
    static constexpr TKeyword<params::Flags> flags;
    static constexpr TKeyword<params::Param> param;

} // namespace mo