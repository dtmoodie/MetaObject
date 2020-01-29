#pragma once
#include <MetaObject/runtime_reflection/DynamicVisitor.hpp>
#include <iostream>
#include <unordered_map>

namespace mo
{
    class MO_EXPORTS BinaryLoader : public LoadCache
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
#ifdef ENVIRONMENT64
#ifndef _MSC_VER
        ILoadVisitor& operator()(long long* val, const std::string& name, const size_t cnt) override;
        ILoadVisitor& operator()(unsigned long long* val, const std::string& name, const size_t cnt) override;
#endif
#else
        ILoadVisitor& operator()(long int* val, const std::string& name, const size_t cnt) override;
        ILoadVisitor& operator()(unsigned long int* val, const std::string& name, const size_t cnt) override;
#endif
        ILoadVisitor& operator()(float* val, const std::string& name, const size_t cnt) override;
        ILoadVisitor& operator()(double*, const std::string&, const size_t cnt) override;
        ILoadVisitor& operator()(void*, const std::string&, const size_t cnt) override;
        ILoadVisitor& operator()(IStructTraits* val, void* inst, const std::string& name = "", size_t cnt = 1) override;
        ILoadVisitor&
        operator()(IContainerTraits* val, void* inst, const std::string& name = "", size_t cnt = 1) override;

        VisitorTraits traits() const override;
        std::string getCurrentElementName() const override;
        size_t getCurrentContainerSize() const override;

      private:
        template <class T>
        ILoadVisitor& loadBinary(T* ptr, size_t cnt = 1);
        ILoadVisitor& loadBinary(void* ptr, size_t cnt = 1);

        std::istream& m_is;
        size_t m_current_size = 0;
    };
} // namespace mo
