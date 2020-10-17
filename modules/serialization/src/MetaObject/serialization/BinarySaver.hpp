#pragma once
#include <MetaObject/runtime_reflection/DynamicVisitor.hpp>
#include <iostream>
#include <unordered_map>

namespace mo
{
    class MO_EXPORTS BinarySaver : public SaveCache
    {
      public:
        BinarySaver(std::ostream& in, bool cereal_compat = false);

        ISaveVisitor& operator()(const bool* ptr, const std::string& name, const size_t cnt) override;
        ISaveVisitor& operator()(const char* ptr, const std::string& name, const size_t cnt) override;
        ISaveVisitor& operator()(const int8_t* ptr, const std::string& name, const size_t cnt) override;
        ISaveVisitor& operator()(const uint8_t* ptr, const std::string& name, const size_t cnt) override;
        ISaveVisitor& operator()(const int16_t* ptr, const std::string& name, const size_t cnt) override;
        ISaveVisitor& operator()(const uint16_t* ptr, const std::string& name, const size_t cnt) override;
        ISaveVisitor& operator()(const int32_t* ptr, const std::string& name, const size_t cnt) override;
        ISaveVisitor& operator()(const uint32_t* ptr, const std::string& name, const size_t cnt) override;
        ISaveVisitor& operator()(const int64_t* ptr, const std::string& name, const size_t cnt) override;
        ISaveVisitor& operator()(const uint64_t* ptr, const std::string& name, const size_t cnt) override;
#ifdef ENVIRONMENT64
#ifndef _MSC_VER
        ISaveVisitor& operator()(const long long* val, const std::string& name = "", const size_t cnt = 1) override;
        ISaveVisitor&
        operator()(const unsigned long long* val, const std::string& name = "", const size_t cnt = 1) override;
#endif
#else
        ISaveVisitor& operator()(const long int* val, const std::string& name = "", const size_t cnt = 1) override;
        ISaveVisitor&
        operator()(const unsigned long int* val, const std::string& name = "", const size_t cnt = 1) override;
#endif
        ISaveVisitor& operator()(const float* val, const std::string& name, const size_t cnt) override;
        ISaveVisitor& operator()(const double*, const std::string&, const size_t) override;
        ISaveVisitor& operator()(const void*, const std::string&, const size_t) override;
        ISaveVisitor&
        operator()(const IStructTraits* val, const void* inst, const std::string& name = "", size_t cnt = 1) override;
        ISaveVisitor& operator()(const IContainerTraits* val,
                                 const void* inst,
                                 const std::string& name = "",
                                 size_t cnt = 1) override;

        template <class T>
        ISaveVisitor& operator()(const T* val, const std::string& name = "", size_t cnt = 1)
        {
            return ISaveVisitor::operator()(val, name, cnt);
        }

        VisitorTraits traits() const override;
        std::shared_ptr<Allocator> getAllocator() const override;
        void setAllocator(std::shared_ptr<Allocator>) override;

      private:
        template <class T>
        ISaveVisitor& saveBinary(const T* ptr, const std::string& name = "", const size_t cnt = 1);
        std::ostream& m_os;
        std::shared_ptr<Allocator> m_allocator;
        bool m_cereal_compat;
    };
} // namespace mo
