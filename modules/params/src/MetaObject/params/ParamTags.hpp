#pragma once
#include <MetaObject/detail/Export.hpp>

#include "MetaObject/core/detail/Enums.hpp"
#include "MetaObject/params/NamedParam.hpp"
#include <MetaObject/core/detail/Time.hpp>
#include <MetaObject/params/Header.hpp>

#include <memory>
#include <string>

namespace mo
{
    struct IAsyncStream;
    class IParam;
    struct ISubscriber;

    namespace tags
    {
        struct Timestamp
        {
            using value_type = Time;
            using storage_type = value_type;
            using pointer_type = storage_type*;
        };

        struct FrameNumber
        {
            using value_type = uint64_t;
            using storage_type = value_type;
            using pointer_type = storage_type*;
        };

        struct Stream
        {
            using value_type = IAsyncStream*;
            using storage_type = IAsyncStream*;
            using pointer_type = storage_type;
        };

        struct Name
        {
            using value_type = std::string;
            using storage_type = value_type;
            using pointer_type = storage_type*;
        };

        struct Flags
        {
            using value_type = ParamFlags;
            using storage_type = value_type;
            using pointer_type = storage_type*;
        };

        struct Param
        {
            using value_type = ISubscriber;
            using storage_type = const ISubscriber*;
            using pointer_type = storage_type;
        };

        struct Header
        {
            using value_type = mo::Header;
            using storage_type = value_type;
            using pointer_type = storage_type*;
        };

        struct Source
        {
            using value_type = SourceId;
            using storage_type = value_type;
            using pointer_type = storage_type*;
        };

        static constexpr TKeyword<tags::Timestamp> timestamp;
        static constexpr TKeyword<tags::FrameNumber> fn;
        static constexpr TKeyword<tags::Stream> stream;
        static constexpr TKeyword<tags::Name> name;
        static constexpr TKeyword<tags::Flags> flags;
        static constexpr TKeyword<tags::Param> param;
        static constexpr TKeyword<tags::Header> header;
        static constexpr TKeyword<tags::Source> source;

    } // namespace tags

    namespace tagged_values
    {
        using Timestamp = mo::TaggedValue<tags::Timestamp>;
        using FrameNumber = mo::TaggedValue<tags::FrameNumber>;
        using Stream = mo::TaggedValue<tags::Stream>;
        using Name = mo::TaggedValue<tags::Name>;
        using Flags = mo::TaggedValue<tags::Flags>;
        using Param = mo::TaggedValue<tags::Param>;
        using Header = mo::TaggedValue<tags::Header>;
        using Source = mo::TaggedValue<tags::Source>;
    } // namespace tagged_values

} // namespace mo
