#include "TypeInfo.hpp"
// Needed to demangle GCC's name mangling.
#ifndef _MSC_VER
#include <cstdlib>
#include <cxxabi.h>
#include <memory>
#endif
#include <MetaObject/core/TypeTable.hpp>
#include <cassert>

namespace mo
{

    const TypeInfo& TypeInfo::Void()
    {
        static const TypeInfo g_info(typeid(void), ct::GetName<void>::getName());
        return g_info;
    }
    TypeInfo::TypeInfo()
    {
        class Void
        {
        };
        m_info = &typeid(Void);
        assert(m_info);
    }

    TypeInfo::TypeInfo(const std::type_info& ti, ct::StringView name)
        : m_info(&ti)
        , m_name(name)
    {
        assert(m_info);
    }

    bool TypeInfo::before(const TypeInfo& rhs) const
    {
        assert(m_info);
        return m_info->before(*rhs.m_info) != 0;
    }

    const std::type_info& TypeInfo::get() const
    {
        assert(m_info);
        return *m_info;
    }

    std::string TypeInfo::name() const
    {
        if (!m_name.empty())
        {
            return m_name;
        }
        assert(m_info);
#ifdef _MSC_VER
        return pInfo_->name();
#else
        int status = -4; // some arbitrary value to eliminate the compiler warning

        // enable c++11 by passing the flag -std=c++11 to g++
        std::unique_ptr<char, void (*)(void*)> res{abi::__cxa_demangle(m_info->name(), nullptr, nullptr, &status),
                                                   std::free};

        return (status == 0) ? res.get() : m_info->name();
#endif
    }

    ct::StringView TypeInfo::nameView() const
    {
        return m_name;
    }

    bool TypeInfo::operator==(const std::type_info& rhs) const
    {
        return this->m_info == &rhs;
    }

    bool TypeInfo::operator!=(const std::type_info& rhs) const
    {
        return !(*this == rhs);
    }

    const std::type_info* TypeInfo::ptr() const
    {
        return m_info;
    }

    bool TypeInfo::operator==(const TypeInfo& rhs) const
    {
        if (!m_name.empty())
        {
            return m_name == rhs.m_name;
        }
        return strcmp(m_info->name(), rhs.m_info->name()) == 0;
    }

    bool TypeInfo::operator!=(const TypeInfo& rhs) const
    {
        return !((*this) == rhs);
    }

    std::size_t TypeInfo::getHash() const
    {
        return m_hash;
    }

    bool operator<(const TypeInfo& lhs, const TypeInfo& rhs)
    {
        return lhs.before(rhs);
    }

    bool operator>(const TypeInfo& lhs, const TypeInfo& rhs)
    {
        return rhs < lhs;
    }

    bool operator<=(const TypeInfo& lhs, const TypeInfo& rhs)
    {
        return !(lhs > rhs);
    }

    bool operator>=(const TypeInfo& lhs, const TypeInfo& rhs)
    {
        return !(lhs < rhs);
    }

    std::ostream& operator<<(std::ostream& os, const TypeInfo& type)
    {
        os << TypeTable::instance()->typeToName(type);
        return os;
    }

} // namespace mo

namespace std
{

    hash<mo::TypeInfo>::result_type hash<mo::TypeInfo>::operator()(argument_type const& s) const noexcept
    {
        const size_t output = s.getHash();
        if (output != 0)
        {
            return output;
        }
        return ct::crc32(s.nameView());
    }
} // namespace std
