#ifndef MO_VISITATION_HASH_VISITOR_HPP
#define MO_VISITATION_HASH_VISITOR_HPP
#include "IDynamicVisitor.hpp"

namespace mo
{
    struct HashVisitor: public IWriteVisitor
    {
        size_t generateObjecthash(const IStructTraits* traits);

        VisitorTraits traits() const override;

        IWriteVisitor& operator()(const bool* val, const std::string& name = "", const size_t cnt = 1) override;
        IWriteVisitor& operator()(const char* val, const std::string& name = "", const size_t cnt = 1) override;
        IWriteVisitor& operator()(const int8_t* val, const std::string& name = "", const size_t cnt = 1) override;
        IWriteVisitor& operator()(const uint8_t* val, const std::string& name = "", const size_t cnt = 1) override;
        IWriteVisitor& operator()(const int16_t* val, const std::string& name = "", const size_t cnt = 1) override;
        IWriteVisitor& operator()(const uint16_t* val, const std::string& name = "", const size_t cnt = 1) override;
        IWriteVisitor& operator()(const int32_t* val, const std::string& name = "", const size_t cnt = 1) override;
        IWriteVisitor& operator()(const uint32_t* val, const std::string& name = "", const size_t cnt = 1) override;
        IWriteVisitor& operator()(const int64_t* val, const std::string& name = "", const size_t cnt = 1) override;
        IWriteVisitor& operator()(const uint64_t* val, const std::string& name = "", const size_t cnt = 1) override;

        IWriteVisitor& operator()(const long long* val, const std::string& name = "", const size_t cnt = 1) override;
        IWriteVisitor& operator()(const unsigned long long* val, const std::string& name = "", const size_t cnt = 1) override;
        IWriteVisitor& operator()(const float* val, const std::string& name = "", const size_t cnt = 1) override;
        IWriteVisitor& operator()(const double* val, const std::string& name = "", const size_t cnt = 1) override;
        IWriteVisitor& operator()(const void* binary, const std::string& name = "", const size_t bytes = 1) override;


        IWriteVisitor& operator()(const IStructTraits* val, const std::string& name = "") override;
        IWriteVisitor& operator()(const IContainerTraits* val, const std::string& name = "") override;
    protected:

        std::unique_ptr<CacheDataContainer>& accessCache(const std::string& name, const uint64_t id = 0) override;

        template<class T>
        void hash(const std::string& name);

        const void* getPointer(const TypeInfo type, const uint64_t id) override;
        void setSerializedPointer(const TypeInfo type, const uint64_t id, const void* ptr) override;

        std::unique_ptr<CacheDataContainer> m_ptr;
        size_t m_hash;
    };
}

#endif // MO_VISITATION_HASH_VISITOR_HPP
