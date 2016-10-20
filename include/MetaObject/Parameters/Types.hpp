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

https://github.com/dtmoodie/parameters
*/
#pragma once

#include <vector>
#include <string>
#include <boost/filesystem/path.hpp>


namespace mo
{
    struct ReadFile : public boost::filesystem::path
    {
        ReadFile(const std::string& str = "") : boost::filesystem::path(str){}
    };
    struct WriteFile : public boost::filesystem::path
    {
        WriteFile(const std::string& file = "") : boost::filesystem::path(file){}
    };
    struct ReadDirectory : public boost::filesystem::path
    {
        ReadDirectory(const boost::filesystem::path& path = "") : boost::filesystem::path(path){}
    };
    struct WriteDirectory : public boost::filesystem::path
    {
        WriteDirectory(const std::string& str = "") : boost::filesystem::path(str){}
    };

    class EnumParameter
    {
    public:
        EnumParameter(const EnumParameter&) = default;
        EnumParameter(const std::initializer_list<std::pair<const char*, int>>& values)
        {
            enumerations.clear();
            this->values.clear();
            for (auto itr = values.begin(); itr != values.end(); ++itr)
            {
                enumerations.emplace_back(itr->first);
                this->values.emplace_back(itr->second);
            }
        }
        EnumParameter()
        {
            currentSelection = 0;
        }
        EnumParameter& operator=(const std::initializer_list<std::pair<const char*, int>>& values)
        {
            enumerations.clear();
            this->values.clear();
            for(auto itr = values.begin(); itr != values.end(); ++itr)
            {
                enumerations.emplace_back(itr->first);
                this->values.emplace_back(itr->second);
            }
            return *this;
        }


        void SetValue(std::initializer_list<char*> string, std::initializer_list<int> values)
        {
            auto iItr = values.begin();
            auto nItr = string.begin();
            enumerations.clear();
            this->values.clear();
            for( ; iItr != values.end() && nItr != string.begin(); ++iItr, ++nItr)
            {
                enumerations.push_back(*nItr);
                this->values.push_back(*iItr);
            }
        }

        void addEnum(int value, const ::std::string& enumeration)
        {
            enumerations.push_back(enumeration);
            values.push_back(value);
        }
        int getValue()
        {
            if (values.empty() || currentSelection >= values.size())
            {
                throw std::range_error("values.empty() || currentSelection >= values.size()");
            }
            return values[currentSelection];
        }
        std::string getEnum()
        {
            if (currentSelection >= values.size())
            {
                throw std::range_error("currentSelection >= values.size()");
            }
            return enumerations[currentSelection];
        }

        std::vector<std::string> enumerations;
        std::vector<int>         values;
        int currentSelection;
    };
}
