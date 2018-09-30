#include "TypeInfo.hpp"
// Needed to demangle GCC's name mangling.
#ifndef _MSC_VER
#include <cstdlib>
#include <cxxabi.h>
#include <memory>
#endif
#include <cassert>
namespace mo
{
    TypeInfo::TypeInfo()
    {
        class Void
        {
        };
        pInfo_ = &typeid(Void);
        assert(pInfo_);
    }

    TypeInfo::TypeInfo(const std::type_info& ti) : pInfo_(&ti) { assert(pInfo_); }

    bool TypeInfo::before(const TypeInfo& rhs) const
    {
        assert(pInfo_);
        return pInfo_->before(*rhs.pInfo_) != 0;
    }

    const std::type_info& TypeInfo::get() const
    {
        assert(pInfo_);
        return *pInfo_;
    }

    std::string TypeInfo::name() const
    {
        assert(pInfo_);
#ifdef _MSC_VER
        return pInfo_->name();
#else
        int status = -4; // some arbitrary value to eliminate the compiler warning

        // enable c++11 by passing the flag -std=c++11 to g++
        std::unique_ptr<char, void (*)(void*)> res{abi::__cxa_demangle(pInfo_->name(), NULL, NULL, &status), std::free};

        return (status == 0) ? res.get() : pInfo_->name();
#endif
    }

    bool TypeInfo::operator==(const std::type_info& rhs) { return this->pInfo_ == &rhs; }
    bool TypeInfo::operator!=(const std::type_info& rhs) { return !(*this == rhs); }

    const std::type_info* TypeInfo::ptr() const { return pInfo_; }

    bool operator==(const TypeInfo& lhs, const TypeInfo& rhs) { return (lhs.get() == rhs.get()) != 0; }

    bool operator<(const TypeInfo& lhs, const TypeInfo& rhs) { return lhs.before(rhs); }

    bool operator!=(const TypeInfo& lhs, const TypeInfo& rhs) { return !(lhs == rhs); }

    bool operator>(const TypeInfo& lhs, const TypeInfo& rhs) { return rhs < lhs; }

    bool operator<=(const TypeInfo& lhs, const TypeInfo& rhs) { return !(lhs > rhs); }

    bool operator>=(const TypeInfo& lhs, const TypeInfo& rhs) { return !(lhs < rhs); }

} // namespace mo

namespace std
{

    hash<mo::TypeInfo>::result_type hash<mo::TypeInfo>::operator()(argument_type const& s) const noexcept
    {
        return std::hash<const std::type_info*>{}(s.ptr());
    }
}
