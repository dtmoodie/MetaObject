/*
Copyright (c) 2015 Daniel Moodie.
All rights reserved.

Redistribution and use in source and binary forms are permitted
provided that the above copyright notice and this paragraph are
duplicated in all such forms and that any documentation,
advertising materials, and other materials related to such
distribution and use acknowledge that the software was developed
by the Daniel Moodie. The name of
Daniel Moodie may not be used to endorse or promote products derived
from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

https://github.com/dtmoodie/MetaObject
*/
#pragma once
#include "MetaObject/types.hpp"
#include <boost/filesystem/path.hpp>
#include <ct/reflect.hpp>
#include <string>
#include <vector>

namespace mo
{
    struct MO_EXPORTS ReadFile : boost::filesystem::path
    {
        ReadFile(const std::string& str = "");
    };

    struct MO_EXPORTS WriteFile : boost::filesystem::path
    {
        WriteFile(const std::string& file = "");
    };

    struct MO_EXPORTS ReadDirectory : boost::filesystem::path
    {
        ReadDirectory(const std::string& path = "");
    };

    struct MO_EXPORTS WriteDirectory : boost::filesystem::path
    {
        WriteDirectory(const std::string& str = "");
    };

    struct MO_EXPORTS EnumParam
    {
        EnumParam(const EnumParam&) = default;
        EnumParam(const std::initializer_list<std::pair<const char*, int>>& values);
        EnumParam(const std::initializer_list<const char*>& string, const std::initializer_list<int>& values);
        EnumParam();

        void setValue(const std::initializer_list<const char*>& string, const std::initializer_list<int>& values);

        void addEnum(int value, const ::std::string& enumeration);
        int getValue() const;
        std::string getEnum() const;
        std::vector<std::string> enumerations;
        std::vector<int> values;
        int current_selection = 0;
    };
} // namespace mo

namespace ct
{
    REFLECT_BEGIN(mo::EnumParam)
        PUBLIC_ACCESS(enumerations)
        PUBLIC_ACCESS(values)
        PUBLIC_ACCESS(current_selection)
    REFLECT_END;

    template <class T>
    struct ReflectImpl<
        T,
        ct::EnableIf<ct::VariadicTypedef<mo::ReadFile, mo::ReadDirectory, mo::WriteFile, mo::WriteDirectory>::
                         template contains<T>()>>
    {
        REFLECT_STUB
            static constexpr decltype(ct::makeMemberPropertyPointer(
                "path",
                ct::selectConstMemberFunctionPointer<boost::filesystem::path, const std::string&>(
                    &boost::filesystem::path::string),
                ct::selectMemberFunctionPointer<boost::filesystem::path, boost::filesystem::path&, const std::string&>(
                    &boost::filesystem::path::operator=)))
            getPtr(const ct::Indexer<__COUNTER__ - REFLECT_COUNT_BEGIN>)
            {
                return ct::makeMemberPropertyPointer(
                    "path",
                    ct::selectConstMemberFunctionPointer<boost::filesystem::path, const std::string&>(
                        &boost::filesystem::path::string),
                    ct::selectMemberFunctionPointer<boost::filesystem::path,
                                                    boost::filesystem::path&,
                                                    const std::string&>(&boost::filesystem::path::operator=));
            }
        REFLECT_INTERNAL_END;
    };
} // namespace ct
