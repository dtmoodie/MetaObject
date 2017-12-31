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
#include <MetaObject/detail/Export.hpp>
#include <vector>
#include <string>
#include <boost/filesystem/path.hpp>

namespace mo
{
    struct MO_EXPORTS ReadFile : public boost::filesystem::path
    {
        ReadFile(const std::string& str = "");

        template<class AR>
        void save(AR& ar) const
        {
            ar(string());
        }

        template<class AR>
        void load(AR& ar)
        {
            std::string value;
            ar(value);
            *this = value;
        }
    };

    struct MO_EXPORTS WriteFile : public boost::filesystem::path
    {
        WriteFile(const std::string& file = "");


        template<class AR>
        void save(AR& ar) const
        {
            ar(string());
        }

        template<class AR>
        void load(AR& ar)
        {
            std::string value;
            ar(value);
            *this = value;
        }
    };

    struct MO_EXPORTS ReadDirectory : public boost::filesystem::path
    {
        ReadDirectory(const std::string& path = "");


        template<class AR>
        void save(AR& ar) const
        {
            ar(string());
        }

        template<class AR>
        void load(AR& ar)
        {
            std::string value;
            ar(value);
            *this = value;
        }
    };

    struct MO_EXPORTS WriteDirectory : public boost::filesystem::path
    {
        WriteDirectory(const std::string& str = "");


        template<class AR>
        void save(AR& ar) const
        {
            ar(string());
        }

        template<class AR>
        void load(AR& ar)
        {
            std::string value;
            ar(value);
            *this = value;
        }
    };

    class MO_EXPORTS EnumParam
    {
    public:
        EnumParam(const EnumParam&) = default;
        EnumParam(const std::initializer_list<std::pair<const char*, int>>& values);
        EnumParam();

        void SetValue(const std::initializer_list<const char*>& string, const std::initializer_list<int>& values);

        void addEnum(int value, const ::std::string& enumeration);
        int getValue() const;
        std::string getEnum() const;

        template<typename T>
        void serialize(T& ar);

        std::vector<std::string> enumerations;
        std::vector<int>         values;
        int currentSelection;
    };
}
