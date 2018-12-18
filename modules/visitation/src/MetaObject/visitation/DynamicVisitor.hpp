#pragma once
#include "IDynamicVisitor.hpp"
#include <memory>
#include <unordered_map>
namespace mo
{
    struct ReadCache : virtual public IReadVisitor
    {
      public:
        using Cache_t =
            std::unordered_map<std::string, std::unordered_map<uint64_t, std::unique_ptr<CacheDataContainer>>>;
        using SerializedPointerMap_t = std::unordered_map<TypeInfo, std::unordered_map<uint64_t, void*>>;

        ReadCache(
            const std::shared_ptr<Cache_t>& cache = std::make_shared<Cache_t>(),
            const std::shared_ptr<SerializedPointerMap_t>& pointer_map = std::make_shared<SerializedPointerMap_t>());

        virtual IReadVisitor& operator()(IStructTraits* val, const std::string& name = "") override;

        std::shared_ptr<Cache_t> getCache();
        void setCache(const std::shared_ptr<Cache_t>& cache);

        void setSerializedPointers(const std::shared_ptr<SerializedPointerMap_t>& pointer_map);
        std::shared_ptr<SerializedPointerMap_t> getSerializedPointers();

      protected:
        virtual std::unique_ptr<CacheDataContainer>& accessCache(const std::string& name,
                                                                 const uint64_t id = 0) override;
        virtual void* getPointer(const TypeInfo type, const uint64_t id) override;
        virtual void setSerializedPointer(const TypeInfo type, const uint64_t id, void* ptr) override;

      private:
        std::shared_ptr<SerializedPointerMap_t> m_serialized_pointers;
        std::shared_ptr<Cache_t> m_cache;
    };

    class WriteCache : virtual public IWriteVisitor
    {
      public:
        using Cache_t =
            std::unordered_map<std::string, std::unordered_map<uint64_t, std::unique_ptr<CacheDataContainer>>>;
        using SerializedPointerMap_t = std::unordered_map<TypeInfo, std::unordered_map<uint64_t, const void*>>;

        WriteCache(
            const std::shared_ptr<Cache_t>& cache = std::make_shared<Cache_t>(),
            const std::shared_ptr<SerializedPointerMap_t>& pointer_map = std::make_shared<SerializedPointerMap_t>());

        std::shared_ptr<Cache_t> getCache();
        void setCache(const std::shared_ptr<Cache_t>& cache);

        void setSerializedPointers(const std::shared_ptr<SerializedPointerMap_t>& pointer_map);
        std::shared_ptr<SerializedPointerMap_t> getSerializedPointers();

        virtual IWriteVisitor& operator()(const IStructTraits* val, const std::string& name = "") override;

      protected:
        virtual std::unique_ptr<CacheDataContainer>& accessCache(const std::string& name,
                                                                 const uint64_t id = 0) override;
        virtual const void* getPointer(const TypeInfo type, const uint64_t id) override;
        virtual void setSerializedPointer(const TypeInfo type, const uint64_t id, const void* ptr) override;

      private:
        std::shared_ptr<SerializedPointerMap_t> m_serialized_pointers;
        std::shared_ptr<Cache_t> m_cache;
    };
}
