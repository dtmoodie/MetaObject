#include "BinaryLoader.hpp"
#include <ct/types/TArrayView.hpp>

namespace mo
{

    BinaryLoader::BinaryLoader(std::istream& in, bool cereal_compat)
        : m_is(in)
        , m_cereal_compat(cereal_compat)
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
#ifndef _MSC_VER
    ILoadVisitor& BinaryLoader::operator()(long long* ptr, const std::string&, const size_t cnt)
    {
        return loadBinary(ptr, cnt);
    }

    ILoadVisitor& BinaryLoader::operator()(unsigned long long* ptr, const std::string&, const size_t cnt)
    {
        return loadBinary(ptr, cnt);
    }
#endif
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

    ILoadVisitor& BinaryLoader::operator()(const IStructTraits* val, void* inst, const std::string& name, size_t cnt)
    {
        if (val->triviallySerializable() && !m_cereal_compat)
        {
            auto ptr = inst;
            const auto sz = val->size() * cnt;
            loadBinary(ptr, sz);
        }
        else
        {
            auto ptr = ct::ptrCast<uint8_t>(inst);
            for (size_t i = 0; i < cnt; ++i)
            {
                LoadCache::operator()(val, ptr, name, 1);
                ptr += val->size();
            }
        }

        return *this;
    }

    ILoadVisitor& BinaryLoader::operator()(const IContainerTraits* val, void* inst, const std::string& name, size_t cnt)
    {
        uint64_t size = 0;
        loadBinary(&size);
        val->setContainerSize(size, inst);
        m_current_size = size;
        val->load(*this, inst, name, cnt);
        m_current_size = 0;
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

    size_t BinaryLoader::getCurrentContainerSize() const
    {
        return m_current_size;
    }

    std::shared_ptr<Allocator> BinaryLoader::getAllocator() const
    {
        return m_allocator;
    }

    void BinaryLoader::setAllocator(std::shared_ptr<Allocator> alloc)
    {
        m_allocator = std::move(alloc);
    }
} // namespace mo
