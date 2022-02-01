#include "DynamicVisitor.hpp"

namespace mo
{

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

    ILoadVisitor& LoadCache::operator()(const IStructTraits* val, void* ptr, const std::string& name, size_t cnt)
    {
        //const auto traits = this->traits();
        // writing data out,
        // const auto id = uint64_t(ptr);
        //(*m_serialized_pointers)[val->type()][id] = ptr;

        val->load(*this, ptr, name, cnt);
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

    void LoadCache::setSerializedPointer(const TypeInfo type, const uint32_t id, void* ptr)
    {
        (*m_serialized_pointers)[type][id] = ptr;
    }

    void* LoadCache::getPointer(const TypeInfo type, const uint32_t id)
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

    ISaveVisitor& SaveCache::operator()(const IStructTraits* val, const void* ptr, const std::string& name, size_t cnt)
    {
        // const auto traits = this->traits();
        // writing data out,
        // const void* ptr = val->ptr();
        // const auto id = getPointerId(val->type(), ptr);
        //(*m_serialized_pointers)[val->type()][id] = ptr;

        val->save(*this, ptr, name, cnt);

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

    uint32_t SaveCache::getPointerId(TypeInfo type, const void* ptr)
    {
        if (ptr == nullptr)
        {
            return 0;
        }
        auto& type_map = (*m_serialized_pointers)[type];
        for (const auto& ptr_itr : type_map)
        {
            if (ptr_itr.second == ptr)
            {
                auto id = ptr_itr.first;
                id = id & (~0x80000000);
                return id;
            }
        }
        auto id = m_serialized_pointer_id++;
        id = id | 0x80000000;
        return id;
    }

    std::shared_ptr<LoadCache::Cache_t> SaveCache::getCache()
    {
        return m_cache;
    }

    void SaveCache::setSerializedPointer(const TypeInfo type, const uint32_t id, const void* ptr)
    {
        (*m_serialized_pointers)[type][id] = ptr;
    }

    const void* SaveCache::getPointer(const TypeInfo type, const uint32_t id)
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
} // namespace mo
