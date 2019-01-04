#include "BinarySaver.hpp"

namespace mo
{

    BinarySaver::BinarySaver(std::ostream& in)
        : m_os(in)
    {
    }


    template <class T>
    ISaveVisitor& BinarySaver::saveBinary(const T* ptr, const std::string&, const size_t cnt)
    {
        const size_t size = sizeof(T) * cnt;
        m_os.write(reinterpret_cast<const char*>(ptr), static_cast<std::streamsize>(size));
        return *this;
    }

    ISaveVisitor& BinarySaver::operator()(const bool* ptr, const std::string& name, const size_t cnt)
    {
        return saveBinary(ptr, name, cnt);
    }

    ISaveVisitor& BinarySaver::operator()(const char* ptr, const std::string& name, const size_t cnt)
    {
        return saveBinary(ptr, name, cnt);
    }

    ISaveVisitor& BinarySaver::operator()(const int8_t* ptr, const std::string& name, const size_t cnt)
    {
        return saveBinary(ptr, name, cnt);
    }

    ISaveVisitor& BinarySaver::operator()(const uint8_t* ptr, const std::string& name, const size_t cnt)
    {
        return saveBinary(ptr, name, cnt);
    }

    ISaveVisitor& BinarySaver::operator()(const int16_t* ptr, const std::string& name, const size_t cnt)
    {
        return saveBinary(ptr, name, cnt);
    }

    ISaveVisitor& BinarySaver::operator()(const uint16_t* ptr, const std::string& name, const size_t cnt)
    {
        return saveBinary(ptr, name, cnt);
    }

    ISaveVisitor& BinarySaver::operator()(const int32_t* ptr, const std::string& name, const size_t cnt)
    {
        return saveBinary(ptr, name, cnt);
    }

    ISaveVisitor& BinarySaver::operator()(const uint32_t* ptr, const std::string& name, const size_t cnt)
    {
        return saveBinary(ptr, name, cnt);
    }

    ISaveVisitor& BinarySaver::operator()(const int64_t* ptr, const std::string& name, const size_t cnt)
    {
        return saveBinary(ptr, name, cnt);
    }

    ISaveVisitor& BinarySaver::operator()(const uint64_t* ptr, const std::string& name, const size_t cnt)
    {
        return saveBinary(ptr, name, cnt);
    }


    ISaveVisitor& BinarySaver::operator()(const long long* ptr, const std::string& name, const size_t cnt)
    {
        return saveBinary(ptr, name, cnt);
    }

    ISaveVisitor& BinarySaver::operator()(const unsigned long long* ptr, const std::string& name, const size_t cnt)
    {
        return saveBinary(ptr, name, cnt);
    }

    ISaveVisitor& BinarySaver::operator()(const float* ptr, const std::string& name, const size_t cnt)
    {
        return saveBinary(ptr, name, cnt);
    }

    ISaveVisitor& BinarySaver::operator()(const double* ptr, const std::string& name, const size_t cnt)
    {
        return saveBinary(ptr, name, cnt);
    }

    ISaveVisitor& BinarySaver::operator()(const void* ptr, const std::string& name, const size_t cnt)
    {
        return saveBinary(static_cast<const char*>(ptr), name, cnt);
    }

    ISaveVisitor& BinarySaver::operator()(ISaveStructTraits* val, const std::string& name)
    {

        const auto cnt = val->count();

        if(val->triviallySerializable())
        {
            auto ptr = val->ptr();
            const auto sz = val->size() * cnt;
            saveBinary(static_cast<const char*>(ptr),name,  sz);
        }else
        {
            for(auto i = 0; i < cnt; ++i)
            {
                SaveCache::operator()(val, name);
                val->increment();
            }
        }

        return *this;
    }

    ISaveVisitor& BinarySaver::operator()( ISaveContainerTraits* val, const std::string&)
    {
        uint64_t num_vals = val->getSize();
        saveBinary(&num_vals);
        val->save(this);
        return *this;
    }

    VisitorTraits BinarySaver::traits() const
    {
        VisitorTraits out;
        out.reader = false;
        out.supports_named_access = false;
        return out;
    }
}
