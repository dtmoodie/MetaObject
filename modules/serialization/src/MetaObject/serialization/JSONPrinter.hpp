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
        virtual ISaveVisitor& operator()(const bool* val, const std::string& name = "", const size_t cnt = 1) override;
        virtual ISaveVisitor& operator()(const char* val, const std::string& name = "", const size_t cnt = 1) override;
        virtual ISaveVisitor& operator()(const int8_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual ISaveVisitor& operator()(const uint8_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual ISaveVisitor& operator()(const int16_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual ISaveVisitor& operator()(const uint16_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual ISaveVisitor& operator()(const int32_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual ISaveVisitor& operator()(const uint32_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual ISaveVisitor& operator()(const int64_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual ISaveVisitor& operator()(const uint64_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual ISaveVisitor& operator()(const long long* val, const std::string& name, const size_t cnt) override;
        virtual ISaveVisitor& operator()(const unsigned long long* val, const std::string& name, const size_t cnt) override;
        virtual ISaveVisitor& operator()(const float* val, const std::string& name, const size_t cnt) override;
        virtual ISaveVisitor& operator()(const double*, const std::string&, const size_t) override;
        virtual ISaveVisitor& operator()(const void*, const std::string&, const size_t) override;
        virtual ISaveVisitor& operator()(ISaveStructTraits* val, const std::string& name = "") override;
        virtual ISaveVisitor& operator()(ISaveContainerTraits* val, const std::string& name = "") override;

        virtual VisitorTraits traits() const override;

      private:
        template <class T>
        ISaveVisitor& writePod(const T* ptr, const std::string& name, const size_t cnt);

        cereal::JSONOutputArchive m_ar;
    };

    struct JSONLoader : public LoadCache
    {
        JSONLoader(std::istream& os);
        virtual ILoadVisitor& operator()(bool* val, const std::string& name = "", const size_t cnt = 1) override;
        virtual ILoadVisitor& operator()(char* val, const std::string& name = "", const size_t cnt = 1) override;
        virtual ILoadVisitor& operator()(int8_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual ILoadVisitor& operator()(uint8_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual ILoadVisitor& operator()(int16_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual ILoadVisitor& operator()(uint16_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual ILoadVisitor& operator()(int32_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual ILoadVisitor& operator()(uint32_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual ILoadVisitor& operator()(int64_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual ILoadVisitor& operator()(uint64_t*, const std::string& name = "", const size_t cnt = 1) override;

        virtual ILoadVisitor& operator()(long long* val, const std::string& name, const size_t cnt) override;
        virtual ILoadVisitor& operator()(unsigned long long* val, const std::string& name, const size_t cnt) override;
        virtual ILoadVisitor& operator()(float* val, const std::string& name, const size_t cnt) override;
        virtual ILoadVisitor& operator()(double*, const std::string&, const size_t) override;
        virtual ILoadVisitor& operator()(void*, const std::string&, const size_t) override;
        virtual ILoadVisitor& operator()(ILoadStructTraits* val, const std::string& name = "") override;
        virtual ILoadVisitor& operator()(ILoadContainerTraits* val, const std::string& name = "") override;

        virtual VisitorTraits traits() const override;

        virtual std::string getCurrentElementName() const override;

      private:
        template <class T>
        ILoadVisitor& readPod(T* ptr, const std::string& name, const size_t cnt);

        cereal::JSONInputArchive m_ar;
        std::string m_last_read_name;
    };
}
