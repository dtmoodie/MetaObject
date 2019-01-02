#include "DynamicVisitor.hpp"

namespace mo
{

    std::string ITraits::getName() const
    {
        const auto t = type();
        return t.name();
    }

    CacheDataContainer::~CacheDataContainer()
    {
    }

    IDynamicVisitor::~IDynamicVisitor()
    {
    }

    StaticVisitor::~StaticVisitor()
    {

    }

    ReadCache::ReadCache(const std::shared_ptr<Cache_t>& cache,
                         const std::shared_ptr<SerializedPointerMap_t>& pointer_map)
    {
        setSerializedPointers(pointer_map);
        setCache(cache);
    }

    IReadVisitor& ReadCache::operator()(IStructTraits* val, const std::string&)
    {
        const auto traits = this->traits();
        // writing data out,
        void* ptr = val->ptr();
        const uint64_t id = uint64_t(ptr);
        (*m_serialized_pointers)[val->type()][id] = ptr;

        val->visit(this);

        if (traits.reader)
        {
            // reading data in
            // void* ptr = val->ptr();
        }
        else
        {
            // writing data out
        }
        return *this;
    }

    std::unique_ptr<CacheDataContainer>& ReadCache::accessCache(const std::string& name, const uint64_t id)
    {
        return (*m_cache)[name][id];
    }

    std::shared_ptr<ReadCache::SerializedPointerMap_t> ReadCache::getSerializedPointers()
    {
        return m_serialized_pointers;
    }

    void ReadCache::setCache(const std::shared_ptr<Cache_t>& cache)
    {
        m_cache = cache;
    }

    void ReadCache::setSerializedPointers(const std::shared_ptr<SerializedPointerMap_t>& pointer_map)
    {
        m_serialized_pointers = pointer_map;
    }

    std::shared_ptr<ReadCache::Cache_t> ReadCache::getCache()
    {
        return m_cache;
    }

    void ReadCache::setSerializedPointer(const TypeInfo type, const uint64_t id, void* ptr)
    {
        (*m_serialized_pointers)[type][id] = ptr;
    }

    void* ReadCache::getPointer(const TypeInfo type, const uint64_t id)
    {
        auto itr1 = m_serialized_pointers->find(type);
        if (itr1 != m_serialized_pointers->end())
        {
            auto itr2 = itr1->second.find(id);
            if (itr2 != itr1->second.end())
            {
                return itr2->second;
            }
        }

        return nullptr;
    }

    //////////////////////// write cache
    WriteCache::WriteCache(const std::shared_ptr<Cache_t>& cache,
                           const std::shared_ptr<SerializedPointerMap_t>& pointer_map)
    {
        setSerializedPointers(pointer_map);
        setCache(cache);
    }

    IWriteVisitor& WriteCache::operator()(const IStructTraits* val, const std::string&)
    {
        const auto traits = this->traits();
        // writing data out,
        const void* ptr = val->ptr();
        const uint64_t id = uint64_t(ptr);
        (*m_serialized_pointers)[val->type()][id] = ptr;

        val->visit(this);

        if (traits.reader)
        {
            // reading data in
            // void* ptr = val->ptr();
        }
        else
        {
            // writing data out
        }
        return *this;
    }

    std::unique_ptr<CacheDataContainer>& WriteCache::accessCache(const std::string& name, const uint64_t id)
    {
        return (*m_cache)[name][id];
    }

    std::shared_ptr<WriteCache::SerializedPointerMap_t> WriteCache::getSerializedPointers()
    {
        return m_serialized_pointers;
    }

    void WriteCache::setCache(const std::shared_ptr<Cache_t>& cache)
    {
        m_cache = cache;
    }

    void WriteCache::setSerializedPointers(const std::shared_ptr<SerializedPointerMap_t>& pointer_map)
    {
        m_serialized_pointers = pointer_map;
    }

    std::shared_ptr<ReadCache::Cache_t> WriteCache::getCache()
    {
        return m_cache;
    }

    void WriteCache::setSerializedPointer(const TypeInfo type, const uint64_t id, const void* ptr)
    {
        (*m_serialized_pointers)[type][id] = ptr;
    }

    const void* WriteCache::getPointer(const TypeInfo type, const uint64_t id)
    {
        auto itr1 = m_serialized_pointers->find(type);
        if (itr1 != m_serialized_pointers->end())
        {
            auto itr2 = itr1->second.find(id);
            if (itr2 != itr1->second.end())
            {
                return const_cast<void*>(itr2->second);
            }
        }

        return nullptr;
    }
}
