#include <MetaObject/params/Types.hpp>
using namespace mo;

ReadFile::ReadFile(const std::string& str) : boost::filesystem::path(str)
{
}

WriteFile::WriteFile(const std::string& file) : boost::filesystem::path(file)
{
}

ReadDirectory::ReadDirectory(const std::string& path) : boost::filesystem::path(path)
{
}

WriteDirectory::WriteDirectory(const std::string& str) : boost::filesystem::path(str)
{
}

EnumParam::EnumParam(const std::initializer_list<std::pair<const char*, int>>& values)
{
    enumerations.clear();
    this->values.clear();
    for (auto itr = values.begin(); itr != values.end(); ++itr)
    {
        enumerations.emplace_back(itr->first);
        this->values.emplace_back(itr->second);
    }
}

EnumParam::EnumParam()
{
    currentSelection = 0;
}

void EnumParam::SetValue(const std::initializer_list<const char*>& string, const std::initializer_list<int>& values)
{
    auto iItr = values.begin();
    auto nItr = string.begin();
    enumerations.clear();
    this->values.clear();
    for (; iItr != values.end() && nItr != string.end(); ++iItr, ++nItr)
    {
        enumerations.push_back(*nItr);
        this->values.push_back(*iItr);
    }
}

void EnumParam::addEnum(int value, const ::std::string& enumeration)
{
    enumerations.push_back(enumeration);
    values.push_back(value);
}

int EnumParam::getValue() const
{
    if (values.empty() || currentSelection >= values.size())
    {
        throw std::range_error("values.empty() || currentSelection >= values.size()");
    }
    return values[currentSelection];
}

std::string EnumParam::getEnum() const
{
    if (currentSelection >= values.size())
    {
        throw std::range_error("currentSelection >= values.size()");
    }
    return enumerations[currentSelection];
}
