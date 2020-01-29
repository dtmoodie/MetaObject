#pragma once
#include "Export.hpp"
#include <ostream>
#include <string>
#include <typeinfo>
namespace mo
{
    class MO_EXPORTS TypeInfo
    {
      public:
        template <class T>
        static TypeInfo create()
        {
            return TypeInfo(typeid(T));
        }

        TypeInfo();
        TypeInfo(const std::type_info&);

        static const TypeInfo& Void();

        const std::type_info& get() const;
        bool before(const TypeInfo& rhs) const;
        std::string name() const;

        template <class T>
        bool isType() const;

        bool operator==(const std::type_info& rhs);
        bool operator!=(const std::type_info& rhs);

        const std::type_info* ptr() const;

      private:
        const std::type_info* pInfo_;
    };

    MO_EXPORTS bool operator==(const TypeInfo& lhs, const TypeInfo& rhs);

    MO_EXPORTS bool operator<(const TypeInfo& lhs, const TypeInfo& rhs);

    MO_EXPORTS bool operator!=(const TypeInfo& lhs, const TypeInfo& rhs);

    MO_EXPORTS bool operator>(const TypeInfo& lhs, const TypeInfo& rhs);

    MO_EXPORTS bool operator<=(const TypeInfo& lhs, const TypeInfo& rhs);

    MO_EXPORTS bool operator>=(const TypeInfo& lhs, const TypeInfo& rhs);

    template <class T>
    bool TypeInfo::isType() const
    {
        return *this == TypeInfo(typeid(T));
    }

    MO_EXPORTS std::ostream& operator<<(std::ostream& os, const TypeInfo& type);
}

namespace std
{
    template <>
    struct MO_EXPORTS hash<mo::TypeInfo>
    {
        using argument_type = mo::TypeInfo;
        using result_type = std::size_t;
        result_type operator()(argument_type const& s) const noexcept;
    };
}
