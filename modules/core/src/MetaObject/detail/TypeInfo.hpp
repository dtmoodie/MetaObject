#pragma once
#include "Export.hpp"
#include <string>
#include <typeinfo>

namespace mo
{
    class MO_EXPORTS TypeInfo
    {
      public:
        TypeInfo();
        TypeInfo(const std::type_info&);
        template <class T>
        TypeInfo(const T& obj);

        const std::type_info& get() const;
        bool before(const TypeInfo& rhs) const;
        std::string name() const;

        template <class T>
        bool isType() const;

        bool operator==(const std::type_info& rhs);
        bool operator!=(const std::type_info& rhs);

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
    TypeInfo::TypeInfo(const T& obj) : TypeInfo(typeid(obj))
    {
    }

    template <class T>
    bool TypeInfo::isType() const
    {
        return *this == TypeInfo(typeid(T));
    }
}
