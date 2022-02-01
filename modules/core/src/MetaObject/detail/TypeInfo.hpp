#ifndef MO_CORE_TYPE_INFO_HPP
#define MO_CORE_TYPE_INFO_HPP
#include "Export.hpp"

#include <ct/reflect.hpp>
#include <ct/reflect_macros.hpp>
#include <ct/typename.hpp>

#include <ostream>
#include <string>
#include <typeinfo>
namespace mo
{
    class MO_EXPORTS TypeInfo
    {
      public:
        template <class T>
        static TypeInfo create();

        TypeInfo();

        static const TypeInfo& Void();

        const std::type_info& get() const;
        bool before(const TypeInfo& rhs) const;
        std::string name() const;
        ct::StringView nameView() const;

        template <class T>
        bool isType() const;

        bool operator==(const std::type_info& rhs) const;
        bool operator!=(const std::type_info& rhs) const;

        bool operator==(const TypeInfo& rhs) const;
        bool operator!=(const TypeInfo& rhs) const;

        const std::type_info* ptr() const;

        std::size_t getHash() const;

      private:
        TypeInfo(const std::type_info&, ct::StringView name);
        const std::type_info* m_info;
        ct::StringView m_name;
        std::size_t m_hash = 0;
    };

    

    MO_EXPORTS bool operator<(const TypeInfo& lhs, const TypeInfo& rhs);

    

    MO_EXPORTS bool operator>(const TypeInfo& lhs, const TypeInfo& rhs);

    MO_EXPORTS bool operator<=(const TypeInfo& lhs, const TypeInfo& rhs);

    MO_EXPORTS bool operator>=(const TypeInfo& lhs, const TypeInfo& rhs);

    template <class T>
    bool TypeInfo::isType() const
    {
        return *this == TypeInfo::create<T>();
    }

    MO_EXPORTS std::ostream& operator<<(std::ostream& os, const TypeInfo& type);
} // namespace mo

namespace std
{
    template <>
    struct MO_EXPORTS hash<mo::TypeInfo>
    {
        using argument_type = mo::TypeInfo;
        using result_type = std::size_t;
        result_type operator()(argument_type const& s) const noexcept;
    };
} // namespace std

namespace ct
{
    REFLECT_BEGIN(mo::TypeInfo)
        PROPERTY(name, &mo::TypeInfo::name)
    REFLECT_END;
} // namespace ct


namespace mo
{
    template <class T>
    TypeInfo TypeInfo::create()
    {
        TypeInfo type(typeid(T), ct::GetName<T>::getName());
        static std::size_t g_hash = std::hash<mo::TypeInfo>{}(type);
        type.m_hash = g_hash;
        return type;
    }
}
#endif // MO_CORE_TYPE_INFO_HPP