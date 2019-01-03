#include "DynamicVisitor.hpp"

namespace mo
{

    ITraits::~ITraits()
    {
    }

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

    LoadCache::LoadCache(const std::shared_ptr<Cache_t>& cache,
                         const std::shared_ptr<SerializedPointerMap_t>& pointer_map)
    {
        setSerializedPointers(pointer_map);
        setCache(cache);
    }

    ILoadVisitor& LoadCache::operator()(ILoadStructTraits* val, const std::string&)
    {
        const auto traits = this->traits();
        // writing data out,
        void* ptr = val->ptr();
        const auto id = uint64_t(ptr);
        (*m_serialized_pointers)[val->type()][id] = ptr;

        val->load(static_cast<ILoadVisitor*>(this));

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

    std::unique_ptr<CacheDataContainer>& LoadCache::accessCache(const std::string& name, const uint64_t id)
    {
        return (*m_cache)[name][id];
    }

    std::shared_ptr<LoadCache::SerializedPointerMap_t> LoadCache::getSerializedPointers()
    {
        return m_serialized_pointers;
    }

    void LoadCache::setCache(const std::shared_ptr<Cache_t>& cache)
    {
        m_cache = cache;
    }

    void LoadCache::setSerializedPointers(const std::shared_ptr<SerializedPointerMap_t>& pointer_map)
    {
        m_serialized_pointers = pointer_map;
    }

    std::shared_ptr<LoadCache::Cache_t> LoadCache::getCache()
    {
        return m_cache;
    }

    void LoadCache::setSerializedPointer(const TypeInfo type, const uint64_t id, void* ptr)
    {
        (*m_serialized_pointers)[type][id] = ptr;
    }

    void* LoadCache::getPointer(const TypeInfo type, const uint64_t id)
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
    SaveCache::SaveCache(const std::shared_ptr<Cache_t>& cache,
                           const std::shared_ptr<SerializedPointerMap_t>& pointer_map)
    {
        setSerializedPointers(pointer_map);
        setCache(cache);
    }

    ISaveVisitor& SaveCache::operator()(ISaveStructTraits* val, const std::string&)
    {
        const auto traits = this->traits();
        // writing data out,
        const void* ptr = val->ptr();
        const auto id = uint64_t(ptr);
        (*m_serialized_pointers)[val->type()][id] = ptr;

        val->save(this);

        return *this;
    }

    std::unique_ptr<CacheDataContainer>& SaveCache::accessCache(const std::string& name, const uint64_t id)
    {
        return (*m_cache)[name][id];
    }

    std::shared_ptr<SaveCache::SerializedPointerMap_t> SaveCache::getSerializedPointers()
    {
        return m_serialized_pointers;
    }

    void SaveCache::setCache(const std::shared_ptr<Cache_t>& cache)
    {
        m_cache = cache;
    }

    void SaveCache::setSerializedPointers(const std::shared_ptr<SerializedPointerMap_t>& pointer_map)
    {
        m_serialized_pointers = pointer_map;
    }

    std::shared_ptr<LoadCache::Cache_t> SaveCache::getCache()
    {
        return m_cache;
    }

    void SaveCache::setSerializedPointer(const TypeInfo type, const uint64_t id, const void* ptr)
    {
        (*m_serialized_pointers)[type][id] = ptr;
    }

    const void* SaveCache::getPointer(const TypeInfo type, const uint64_t id)
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
