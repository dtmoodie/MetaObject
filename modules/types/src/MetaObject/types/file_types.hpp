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
        ReadFile& operator=(const std::string& str);
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

    MO_EXPORTS int findNextFileIndex(const std::string& dir, std::string extension, const std::string& stem);

    struct MO_EXPORTS AppendDirectory : public WriteDirectory
    {
        AppendDirectory(const std::string& path, const std::string& file_prefix, const std::string& ext);

        std::string nextFilename() const;
        int nextFileIndex() const;

        AppendDirectory& operator=(const std::string&);

        template <class AR>
        std::string save_minimal(AR&) const
        {
            std::string out = string() + "/" + m_file_prefix + "." + m_extension;
            return out;
        }
        template <class AR>
        void load_minimal(const AR&, const std::string& val)
        {
            const auto pos0 = val.find_last_of('/');
            MO_ASSERT_NE(pos0, std::string::npos);
            const auto pos1 = val.find_last_of('.');
            static_cast<WriteDirectory&>(*this) = val.substr(0, pos0);
            m_file_prefix = val.substr(pos0, pos1 - pos0);
            m_extension = val.substr(pos1);
        }

      private:
        std::string m_file_prefix;
        std::string m_extension;
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

    REFLECT_BEGIN(mo::AppendDirectory)
        MEMBER_FUNCTION(nextFilename)
        MEMBER_FUNCTION(nextFileIndex)
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
