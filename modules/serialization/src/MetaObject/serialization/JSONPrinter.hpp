#pragma once

#include "MetaObject/runtime_reflection/DynamicVisitor.hpp"

#include <cereal/archives/json.hpp>
#include <iostream>
#include <unordered_map>
namespace mo
{
    struct MO_EXPORTS JSONSaver : public SaveCache
    {
        JSONSaver(std::ostream& os);
        ISaveVisitor& operator()(const bool* val, const std::string& name = "", const size_t cnt = 1) override;
        ISaveVisitor& operator()(const char* val, const std::string& name = "", const size_t cnt = 1) override;
        ISaveVisitor& operator()(const int8_t*, const std::string& name = "", const size_t cnt = 1) override;
        ISaveVisitor& operator()(const uint8_t*, const std::string& name = "", const size_t cnt = 1) override;
        ISaveVisitor& operator()(const int16_t*, const std::string& name = "", const size_t cnt = 1) override;
        ISaveVisitor& operator()(const uint16_t*, const std::string& name = "", const size_t cnt = 1) override;
        ISaveVisitor& operator()(const int32_t*, const std::string& name = "", const size_t cnt = 1) override;
        ISaveVisitor& operator()(const uint32_t*, const std::string& name = "", const size_t cnt = 1) override;
        ISaveVisitor& operator()(const int64_t*, const std::string& name = "", const size_t cnt = 1) override;
        ISaveVisitor& operator()(const uint64_t*, const std::string& name = "", const size_t cnt = 1) override;
#ifdef ENVIRONMENT64

#ifndef _MSC_VER
        ISaveVisitor& operator()(const long long* val, const std::string& name, const size_t cnt) override;
        ISaveVisitor& operator()(const unsigned long long* val, const std::string& name, const size_t cnt) override;
#endif
#else
        ISaveVisitor& operator()(const long int* val, const std::string& name, const size_t cnt) override;
        ISaveVisitor& operator()(const unsigned long int* val, const std::string& name, const size_t cnt) override;
#endif
        ISaveVisitor& operator()(const float* val, const std::string& name, const size_t cnt) override;
        ISaveVisitor& operator()(const double*, const std::string&, const size_t) override;
        ISaveVisitor& operator()(const void*, const std::string&, const size_t) override;
        ISaveVisitor&
        operator()(IStructTraits* val, const void* inst, const std::string& name = "", size_t cnt = 1) override;
        ISaveVisitor&
        operator()(IContainerTraits* val, const void* inst, const std::string& name = "", size_t cnt = 1) override;

        VisitorTraits traits() const override;

        template <class T>
        ISaveVisitor& operator()(T* val, const std::string& name = "", size_t cnt = 1)
        {
            return ISaveVisitor::operator()(val, name, cnt);
        }

        std::shared_ptr<Allocator> getAllocator() const override;
        void setAllocator(std::shared_ptr<Allocator>) override;

      private:
        template <class T>
        ISaveVisitor& writePod(const T* ptr, const std::string& name, const size_t cnt);

        cereal::JSONOutputArchive m_ar;
        std::shared_ptr<Allocator> m_allocator;
    };

    struct MO_EXPORTS JSONLoader : public LoadCache
    {
        JSONLoader(std::istream& os);
        ILoadVisitor& operator()(bool* val, const std::string& name = "", const size_t cnt = 1) override;
        ILoadVisitor& operator()(char* val, const std::string& name = "", const size_t cnt = 1) override;
        ILoadVisitor& operator()(int8_t*, const std::string& name = "", const size_t cnt = 1) override;
        ILoadVisitor& operator()(uint8_t*, const std::string& name = "", const size_t cnt = 1) override;
        ILoadVisitor& operator()(int16_t*, const std::string& name = "", const size_t cnt = 1) override;
        ILoadVisitor& operator()(uint16_t*, const std::string& name = "", const size_t cnt = 1) override;
        ILoadVisitor& operator()(int32_t*, const std::string& name = "", const size_t cnt = 1) override;
        ILoadVisitor& operator()(uint32_t*, const std::string& name = "", const size_t cnt = 1) override;
        ILoadVisitor& operator()(int64_t*, const std::string& name = "", const size_t cnt = 1) override;
        ILoadVisitor& operator()(uint64_t*, const std::string& name = "", const size_t cnt = 1) override;
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
        ILoadVisitor& operator()(double*, const std::string&, const size_t) override;
        ILoadVisitor& operator()(void*, const std::string&, const size_t) override;
        ILoadVisitor& operator()(IStructTraits* val, void* inst, const std::string& name = "", size_t cnt = 1) override;
        ILoadVisitor&
        operator()(IContainerTraits* val, void* inst, const std::string& name = "", size_t cnt = 1) override;

        template <class T>
        ILoadVisitor& operator()(T* val, const std::string& name = "", size_t cnt = 1)
        {
            return ILoadVisitor::operator()(val, name, cnt);
        }

        VisitorTraits traits() const override;

        std::string getCurrentElementName() const override;
        size_t getCurrentContainerSize() const override;

        std::shared_ptr<Allocator> getAllocator() const override;
        void setAllocator(std::shared_ptr<Allocator>) override;

      private:
        template <class T>
        ILoadVisitor& readPod(T* ptr, const std::string& name, const size_t cnt);

        cereal::JSONInputArchive m_ar;
        std::string m_last_read_name;
        size_t m_current_size;
        std::shared_ptr<Allocator> m_allocator;
    };
} // namespace mo
