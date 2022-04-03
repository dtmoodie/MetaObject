#include <MetaObject/types/file_types.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <iomanip>
namespace mo
{
    ReadFile::ReadFile(const std::string& str)
        : boost::filesystem::path(str)
    {
    }

    ReadFile& ReadFile::operator=(const std::string& str)
    {
        boost::filesystem::path::operator=(str);
        return *this;
    }

    WriteFile::WriteFile(const std::string& file)
        : boost::filesystem::path(file)
    {
    }

    ReadDirectory::ReadDirectory(const std::string& path)
        : boost::filesystem::path(path)
    {
    }

    WriteDirectory::WriteDirectory(const std::string& str)
        : boost::filesystem::path(str)
    {
        if (!boost::filesystem::exists(*this))
        {
            boost::filesystem::create_directories(*this);
        }
    }

    WriteDirectory& WriteDirectory::operator=(const std::string& str)
    {
        static_cast<boost::filesystem::path&>(*this) = str;
        if (!boost::filesystem::exists(*this))
        {
            boost::filesystem::create_directories(*this);
        }
        return *this;
    }

    AppendDirectory::AppendDirectory(const std::string& path, const std::string& file_prefix, const std::string& ext)
        : WriteDirectory(path)
        , m_file_prefix(file_prefix)
        , m_extension(ext)
    {
    }

    AppendDirectory& AppendDirectory::operator=(const std::string& str)
    {
        static_cast<WriteDirectory&>(*this) = str;
        return *this;
    }

    int findNextFileIndex(const std::string& dir, std::string extension, const std::string& stem_)
    {
        boost::filesystem::path path(dir);
        boost::filesystem::directory_iterator end;
        std::transform(extension.begin(), extension.end(), extension.begin(), tolower);
        int frame_count = 0;
        for (boost::filesystem::directory_iterator itr(path); itr != end; ++itr)
        {
            std::string ext = itr->path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), tolower);
            if (ext == extension)
            {
                const std::string& stem = itr->path().stem().string();
                if (stem.find(stem_) == 0)
                {
                    auto start = stem.find_last_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_");
                    auto end = stem.find_last_of("0123456789");
                    if (stem.size() && end != std::string::npos)
                    {
                        int idx =
                            boost::lexical_cast<int>(stem.substr(start == std::string::npos ? 0 : start + 1, end));
                        frame_count = std::max(frame_count, idx + 1);
                    }
                }
            }
        }
        return frame_count;
    }

    int AppendDirectory::nextFileIndex() const
    {
        return findNextFileIndex(string(), m_extension, m_file_prefix);
    }

    std::string AppendDirectory::nextFilename() const
    {
        int idx = nextFileIndex();
        std::stringstream ss;
        ss << string() << "/" << m_file_prefix << '-';
        ss << std::setw(6) << std::setfill('0') << idx;
        ss << '.' << m_extension;

        return std::move(ss).str();
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

    EnumParam::EnumParam(const std::initializer_list<const char*>& string, const std::initializer_list<int>& values)
    {
        setValue(string, values);
    }

    EnumParam::EnumParam()
    {
        current_selection = 0;
    }

    void EnumParam::setValue(const std::initializer_list<const char*>& string, const std::initializer_list<int>& values)
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
        if (values.empty() || current_selection >= values.size())
        {
            throw std::range_error("values.empty() || currentSelection >= values.size()");
        }
        return values[current_selection];
    }

    std::string EnumParam::getEnum() const
    {
        if (current_selection >= values.size())
        {
            throw std::range_error("currentSelection >= values.size()");
        }
        return enumerations[current_selection];
    }

} // namespace mo
