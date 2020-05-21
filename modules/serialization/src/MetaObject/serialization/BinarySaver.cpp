#include "BinarySaver.hpp"
#include <ct/types/TArrayView.hpp>
namespace mo
{

    BinarySaver::BinarySaver(std::ostream& in, bool cereal_compat)
        : m_os(in)
        , m_cereal_compat(cereal_compat)
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

#ifdef ENVIRONMENT64

#ifndef _MSC_VER
    ISaveVisitor& BinarySaver::operator()(const long long* ptr, const std::string& name, const size_t cnt)
    {
        return saveBinary(ptr, name, cnt);
    }

    ISaveVisitor& BinarySaver::operator()(const unsigned long long* ptr, const std::string& name, const size_t cnt)
    {
        return saveBinary(ptr, name, cnt);
    }
#endif

#else
    ISaveVisitor& BinarySaver::operator()(const long int* ptr, const std::string& name, const size_t cnt)
    {
        return saveBinary(ptr, name, cnt);
    }

    ISaveVisitor& BinarySaver::operator()(const unsigned long int* ptr, const std::string& name, const size_t cnt)
    {
        return saveBinary(ptr, name, cnt);
    }
#endif

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

    ISaveVisitor& BinarySaver::operator()(IStructTraits* trait, const void* val, const std::string& name, size_t cnt)
    {
        if (trait->triviallySerializable() && !m_cereal_compat)
        {
            auto ptr = val;
            const auto sz = trait->size() * cnt;
            saveBinary(static_cast<const char*>(ptr), name, sz);
        }
        else
        {
            auto ptr = ct::ptrCast<const uint8_t>(val);
            for (size_t i = 0; i < cnt; ++i)
            {
                SaveCache::operator()(trait, ct::ptrCast<const void>(ptr), name, 1);
                ptr += trait->size();
            }
        }

        return *this;
    }

    ISaveVisitor& BinarySaver::operator()(IContainerTraits* val, const void* inst, const std::string& name, size_t cnt)
    {
        uint64_t num_vals = val->getContainerSize(inst);
        saveBinary(&num_vals);
        val->save(*this, inst, name, cnt);
        return *this;
    }

    VisitorTraits BinarySaver::traits() const
    {
        VisitorTraits out;
        out.reader = false;
        out.supports_named_access = false;
        return out;
    }

    std::shared_ptr<Allocator> BinarySaver::getAllocator() const
    {
        return m_allocator;
    }

    void BinarySaver::setAllocator(std::shared_ptr<Allocator> alloc)
    {
        m_allocator = std::move(alloc);
    }
} // namespace mo
