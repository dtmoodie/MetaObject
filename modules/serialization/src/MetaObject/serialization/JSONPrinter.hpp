#pragma once

#include "MetaObject/runtime_reflection/DynamicVisitor.hpp"

#include <cereal/archives/json.hpp>
#include <iostream>
#include <unordered_map>
namespace mo
{
    struct JSONSaver : public SaveCache
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
        ISaveVisitor& operator()(const long long* val, const std::string& name, const size_t cnt) override;
        ISaveVisitor& operator()(const unsigned long long* val, const std::string& name, const size_t cnt) override;
#else
        ISaveVisitor& operator()(const long int* val, const std::string& name, const size_t cnt) override;
        ISaveVisitor& operator()(const unsigned long int* val, const std::string& name, const size_t cnt) override;
#endif
        ISaveVisitor& operator()(const float* val, const std::string& name, const size_t cnt) override;
        ISaveVisitor& operator()(const double*, const std::string&, const size_t) override;
        ISaveVisitor& operator()(const void*, const std::string&, const size_t) override;
        ISaveVisitor& operator()(ISaveStructTraits* val, const std::string& name = "") override;
        ISaveVisitor& operator()(ISaveContainerTraits* val, const std::string& name = "") override;

        VisitorTraits traits() const override;

      private:
        template <class T>
        ISaveVisitor& writePod(const T* ptr, const std::string& name, const size_t cnt);

        cereal::JSONOutputArchive m_ar;
    };

    struct JSONLoader : public LoadCache
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
        ILoadVisitor& operator()(long long* val, const std::string& name, const size_t cnt) override;
        ILoadVisitor& operator()(unsigned long long* val, const std::string& name, const size_t cnt) override;
#else
        ILoadVisitor& operator()(long int* val, const std::string& name, const size_t cnt) override;
        ILoadVisitor& operator()(unsigned long int* val, const std::string& name, const size_t cnt) override;
#endif
        ILoadVisitor& operator()(float* val, const std::string& name, const size_t cnt) override;
        ILoadVisitor& operator()(double*, const std::string&, const size_t) override;
        ILoadVisitor& operator()(void*, const std::string&, const size_t) override;
        ILoadVisitor& operator()(ILoadStructTraits* val, const std::string& name = "") override;
        ILoadVisitor& operator()(ILoadContainerTraits* val, const std::string& name = "") override;

        VisitorTraits traits() const override;

        std::string getCurrentElementName() const override;

      private:
        template <class T>
        ILoadVisitor& readPod(T* ptr, const std::string& name, const size_t cnt);

        cereal::JSONInputArchive m_ar;
        std::string m_last_read_name;
    };
}
