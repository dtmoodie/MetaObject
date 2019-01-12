#pragma once
#include <MetaObject/runtime_reflection/DynamicVisitor.hpp>
#include <iostream>
#include <unordered_map>

namespace mo
{
    class BinarySaver : public SaveCache
    {
      public:
        BinarySaver(std::ostream& in);

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

        ISaveVisitor& operator()(const long long* val, const std::string& name = "", const size_t cnt = 1) override;
        ISaveVisitor& operator()(const unsigned long long* val, const std::string& name = "", const size_t cnt = 1) override;
        ISaveVisitor& operator()(const float* val, const std::string& name, const size_t cnt) override;
        ISaveVisitor& operator()(const double*, const std::string&, const size_t) override;
        ISaveVisitor& operator()(const void*, const std::string&, const size_t) override;
        ISaveVisitor& operator()(ISaveStructTraits* val, const std::string& name = "") override;
        ISaveVisitor& operator()(ISaveContainerTraits* val, const std::string& name = "") override;

        VisitorTraits traits() const override;

      private:
        template <class T>
        ISaveVisitor& saveBinary(const T* ptr, const std::string& name = "", const size_t cnt = 1);
        std::ostream& m_os;
    };
}
