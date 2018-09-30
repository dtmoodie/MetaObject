#pragma once
#include "MetaObject/params/NamedParam.hpp"
#include <MetaObject/core/detail/Time.hpp>
#include "MetaObject/core/detail/Enums.hpp"
#include <memory>

namespace mo
{
    class Context;
    class ICoordinateSystem;
    class IParam;

    MO_KEYWORD_INPUT(timestamp, mo::Time_t)
    MO_KEYWORD_INPUT(frame_number, uint64_t)
    MO_KEYWORD_INPUT(coordinate_system, const std::shared_ptr<ICoordinateSystem>)
    MO_KEYWORD_INPUT(context, Context*)
    MO_KEYWORD_INPUT(param_name, std::string)
    MO_KEYWORD_INPUT(tree_root, std::string)
    MO_KEYWORD_INPUT(param_flags, EnumClassBitset<ParamFlags>)

    namespace tag
    {
        struct param;
    } // namespace tag

    namespace kwargs
    {
        template <>
        struct TaggedArgument<tag::param> : public TaggedBase
        {
            typedef tag::param TagType;
            explicit TaggedArgument(const IParam& val) : arg(&val) {}

            const void* get() const { return arg; }

          protected:
            const void* arg;
        };

        template <>
        struct MO_EXPORTS TKeyword<tag::param>
        {
            static TKeyword instance;
            TaggedArgument<tag::param> operator=(const IParam& data);
        };

    } // namespace kwargs

    namespace tag
    {
        struct param
        {
            typedef IParam Type;
            typedef const Type& ConstRef;
            typedef Type& Ref;
            typedef ConstRef StorageType;
            typedef const void* VoidType;
            template <typename T>
            static constexpr bool AllowedType()
            {
                return std::is_same<Type, T>::value;
            }
            static VoidType GetPtr(const Type& arg) { return &arg; }
            template <class T>
            static VoidType GetPtr(const T& arg)
            {
                (void)arg;
                return nullptr;
            }
        }; // struct param
        static mo::kwargs::TKeyword<param>& _param = mo::kwargs::TKeyword<param>::instance;
    } // namespace tag
}
