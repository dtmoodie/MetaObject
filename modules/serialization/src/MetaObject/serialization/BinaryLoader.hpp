#pragma once
#include <MetaObject/visitation/DynamicVisitor.hpp>
#include <iostream>
#include <unordered_map>

namespace mo
{
    class BinaryLoader : public LoadCache
    {
      public:
        BinaryLoader(std::istream& in);

        ILoadVisitor& operator()(bool* ptr, const std::string& name, const size_t cnt) override;
        ILoadVisitor& operator()(char* ptr, const std::string& name, const size_t cnt) override;
        ILoadVisitor& operator()(int8_t* ptr, const std::string& name, const size_t cnt) override;
        ILoadVisitor& operator()(uint8_t* ptr, const std::string& name, const size_t cnt) override;
        ILoadVisitor& operator()(int16_t* ptr, const std::string& name, const size_t cnt) override;
        ILoadVisitor& operator()(uint16_t* ptr, const std::string& name, const size_t cnt) override;
        ILoadVisitor& operator()(int32_t* ptr, const std::string& name, const size_t cnt) override;
        ILoadVisitor& operator()(uint32_t* ptr, const std::string& name, const size_t cnt) override;
        ILoadVisitor& operator()(int64_t* ptr, const std::string& name, const size_t cnt) override;
        ILoadVisitor& operator()(uint64_t* ptr, const std::string& name, const size_t cnt) override;
        ILoadVisitor& operator()(long long* val, const std::string& name, const size_t cnt) override;
        ILoadVisitor& operator()(unsigned long long* val, const std::string& name, const size_t cnt) override;
        ILoadVisitor& operator()(float* val, const std::string& name, const size_t cnt) override;
        ILoadVisitor& operator()(double*, const std::string&, const size_t cnt) override;
        ILoadVisitor& operator()(void*, const std::string&, const size_t cnt) override;
        ILoadVisitor& operator()(ILoadStructTraits* val, const std::string& name = "") override;
        ILoadVisitor& operator()(ILoadContainerTraits* val, const std::string& name = "") override;

        VisitorTraits traits() const override;
        std::string getCurrentElementName() const override;

      private:
        template <class T>
        ILoadVisitor& loadBinary(T* ptr, size_t cnt = 1);
        ILoadVisitor& loadBinary(void* ptr, size_t cnt = 1);

        std::istream& m_is;
    };
}
