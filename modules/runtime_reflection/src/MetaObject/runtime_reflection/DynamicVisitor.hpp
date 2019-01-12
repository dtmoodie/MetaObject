#pragma once
#include "IDynamicVisitor.hpp"
#include <memory>
#include <unordered_map>
namespace mo
{
    struct LoadCache : virtual public ILoadVisitor
    {
      public:
        using Cache_t =
            std::unordered_map<std::string, std::unordered_map<uint64_t, std::unique_ptr<CacheDataContainer>>>;
        using SerializedPointerMap_t = std::unordered_map<TypeInfo, std::unordered_map<uint64_t, void*>>;

        LoadCache(
            const std::shared_ptr<Cache_t>& cache = std::make_shared<Cache_t>(),
            const std::shared_ptr<SerializedPointerMap_t>& pointer_map = std::make_shared<SerializedPointerMap_t>());

        ILoadVisitor& operator()(ILoadStructTraits* val, const std::string& name = "") override;

        std::shared_ptr<Cache_t> getCache();
        void setCache(const std::shared_ptr<Cache_t>& cache);

        void setSerializedPointers(const std::shared_ptr<SerializedPointerMap_t>& pointer_map);
        std::shared_ptr<SerializedPointerMap_t> getSerializedPointers();

      protected:
        std::unique_ptr<CacheDataContainer>& accessCache(const std::string& name,
                                                                 const uint64_t id = 0) override;
        void* getPointer(const TypeInfo type, const uint64_t id) override;
        void setSerializedPointer(const TypeInfo type, const uint64_t id, void* ptr) override;

      private:
        std::shared_ptr<SerializedPointerMap_t> m_serialized_pointers;
        std::shared_ptr<Cache_t> m_cache;
    };

    class SaveCache : virtual public ISaveVisitor
    {
      public:
        using Cache_t =
            std::unordered_map<std::string, std::unordered_map<uint64_t, std::unique_ptr<CacheDataContainer>>>;
        using SerializedPointerMap_t = std::unordered_map<TypeInfo, std::unordered_map<uint64_t, const void*>>;

        SaveCache(
            const std::shared_ptr<Cache_t>& cache = std::make_shared<Cache_t>(),
            const std::shared_ptr<SerializedPointerMap_t>& pointer_map = std::make_shared<SerializedPointerMap_t>());

        std::shared_ptr<Cache_t> getCache();
        void setCache(const std::shared_ptr<Cache_t>& cache);

        void setSerializedPointers(const std::shared_ptr<SerializedPointerMap_t>& pointer_map);
        std::shared_ptr<SerializedPointerMap_t> getSerializedPointers();

        ISaveVisitor& operator()(ISaveStructTraits* val, const std::string& name = "") override;

      protected:
        std::unique_ptr<CacheDataContainer>& accessCache(const std::string& name,
                                                                 const uint64_t id = 0) override;
        const void* getPointer(const TypeInfo type, const uint64_t id) override;
        void setSerializedPointer(const TypeInfo type, const uint64_t id, const void* ptr) override;

      private:
        std::shared_ptr<SerializedPointerMap_t> m_serialized_pointers;
        std::shared_ptr<Cache_t> m_cache;
    };
}
