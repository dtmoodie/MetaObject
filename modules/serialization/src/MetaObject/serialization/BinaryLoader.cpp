#include "BinaryLoader.hpp"

namespace mo
{

    BinaryLoader::BinaryLoader(std::istream& in)
        : m_is(in)
    {
    }

    template <class T>
    ILoadVisitor& BinaryLoader::loadBinary(T* ptr, const size_t cnt)
    {
        m_is.read(static_cast<char*>(static_cast<void*>(ptr)), std::streamsize(cnt * sizeof(T)));
        return *this;
    }

    ILoadVisitor& BinaryLoader::loadBinary(void* ptr, const size_t cnt)
    {
        m_is.read(static_cast<char*>(static_cast<void*>(ptr)), std::streamsize(cnt));
        return *this;
    }

    ILoadVisitor& BinaryLoader::operator()(bool* ptr, const std::string&, const size_t cnt)
    {
        return loadBinary(ptr, cnt);
    }

    ILoadVisitor& BinaryLoader::operator()(char* ptr, const std::string&, const size_t cnt)
    {
        return loadBinary(ptr, cnt);
    }

    ILoadVisitor& BinaryLoader::operator()(int8_t* ptr, const std::string&, const size_t cnt)
    {
        return loadBinary(ptr, cnt);
    }

    ILoadVisitor& BinaryLoader::operator()(uint8_t* ptr, const std::string&, const size_t cnt)
    {
        return loadBinary(ptr, cnt);
    }

    ILoadVisitor& BinaryLoader::operator()(int16_t* ptr, const std::string&, const size_t cnt)
    {
        return loadBinary(ptr, cnt);
    }

    ILoadVisitor& BinaryLoader::operator()(uint16_t* ptr, const std::string&, const size_t cnt)
    {
        return loadBinary(ptr, cnt);
    }

    ILoadVisitor& BinaryLoader::operator()(int32_t* ptr, const std::string&, const size_t cnt)
    {
        return loadBinary(ptr, cnt);
    }

    ILoadVisitor& BinaryLoader::operator()(uint32_t* ptr, const std::string&, const size_t cnt)
    {
        return loadBinary(ptr, cnt);
    }

    ILoadVisitor& BinaryLoader::operator()(int64_t* ptr, const std::string&, const size_t cnt)
    {
        return loadBinary(ptr, cnt);
    }

    ILoadVisitor& BinaryLoader::operator()(uint64_t* ptr, const std::string&, const size_t cnt)
    {
        return loadBinary(ptr, cnt);
    }
#ifdef ENVIRONMENT64
    ILoadVisitor& BinaryLoader::operator()(long long* ptr, const std::string&, const size_t cnt)
    {
        return loadBinary(ptr, cnt);
    }

    ILoadVisitor& BinaryLoader::operator()(unsigned long long* ptr, const std::string&, const size_t cnt)
    {
        return loadBinary(ptr, cnt);
    }
#else
    ILoadVisitor& BinaryLoader::operator()(long int* ptr, const std::string&, const size_t cnt)
    {
        return loadBinary(ptr, cnt);
    }

    ILoadVisitor& BinaryLoader::operator()(unsigned long int* ptr, const std::string&, const size_t cnt)
    {
        return loadBinary(ptr, cnt);
    }
#endif
    ILoadVisitor& BinaryLoader::operator()(float* ptr, const std::string&, const size_t cnt)
    {
        return loadBinary(ptr, cnt);
    }

    ILoadVisitor& BinaryLoader::operator()(double* ptr, const std::string&, const size_t cnt)
    {
        return loadBinary(ptr, cnt);
    }

    ILoadVisitor& BinaryLoader::operator()(void* ptr, const std::string&, const size_t cnt)
    {
        return loadBinary(static_cast<char*>(ptr), cnt);
    }

    ILoadVisitor& BinaryLoader::operator()(ILoadStructTraits* val, const std::string& name)
    {
        const auto cnt = val->count();
        if (val->triviallySerializable())
        {
            auto ptr = val->ptr();
            const auto sz = val->size() * cnt;
            loadBinary(ptr, sz);
        }
        else
        {
            for (size_t i = 0; i < cnt; ++i)
            {
                LoadCache::operator()(val, name);
                val->increment();
            }
        }

        return *this;
    }

    ILoadVisitor& BinaryLoader::operator()(ILoadContainerTraits* val, const std::string&)
    {
        uint64_t size = 0;
        loadBinary(&size);
        val->setSize(size);
        val->load(this);
        return *this;
    }

    VisitorTraits BinaryLoader::traits() const
    {
        VisitorTraits out;
        out.reader = true;
        out.supports_named_access = false;
        return out;
    }

    std::string BinaryLoader::getCurrentElementName() const
    {
        return "";
    }
}
