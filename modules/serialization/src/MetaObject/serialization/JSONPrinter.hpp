#pragma once

#include "MetaObject/visitation/DynamicVisitor.hpp"

#include <cereal/archives/json.hpp>
#include <iostream>
#include <unordered_map>
namespace mo
{
    struct JSONWriter : public WriteCache
    {
        JSONWriter(std::ostream& os);
        virtual IWriteVisitor& operator()(const char* val, const std::string& name = "", const size_t cnt = 1) override;
        virtual IWriteVisitor& operator()(const int8_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual IWriteVisitor& operator()(const uint8_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual IWriteVisitor& operator()(const int16_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual IWriteVisitor& operator()(const uint16_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual IWriteVisitor& operator()(const int32_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual IWriteVisitor& operator()(const uint32_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual IWriteVisitor& operator()(const int64_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual IWriteVisitor& operator()(const uint64_t*, const std::string& name = "", const size_t cnt = 1) override;

        virtual IWriteVisitor& operator()(const float* val, const std::string& name, const size_t cnt) override;
        virtual IWriteVisitor& operator()(const double*, const std::string&, const size_t) override;
        virtual IWriteVisitor& operator()(const void*, const std::string&, const size_t) override;
        virtual IWriteVisitor& operator()(const IStructTraits* val, const std::string& name = "") override;
        virtual IWriteVisitor& operator()(const IContainerTraits* val, const std::string& name = "") override;

        virtual VisitorTraits traits() const override;

      private:
        template <class T>
        IWriteVisitor& writePod(const T* ptr, const std::string& name, const size_t cnt);

        cereal::JSONOutputArchive m_ar;
    };

    struct JSONReader : public ReadCache
    {
        JSONReader(std::istream& os);
        virtual IReadVisitor& operator()(char* val, const std::string& name = "", const size_t cnt = 1) override;
        virtual IReadVisitor& operator()(int8_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual IReadVisitor& operator()(uint8_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual IReadVisitor& operator()(int16_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual IReadVisitor& operator()(uint16_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual IReadVisitor& operator()(int32_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual IReadVisitor& operator()(uint32_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual IReadVisitor& operator()(int64_t*, const std::string& name = "", const size_t cnt = 1) override;
        virtual IReadVisitor& operator()(uint64_t*, const std::string& name = "", const size_t cnt = 1) override;

        virtual IReadVisitor& operator()(float* val, const std::string& name, const size_t cnt) override;
        virtual IReadVisitor& operator()(double*, const std::string&, const size_t) override;
        virtual IReadVisitor& operator()(void*, const std::string&, const size_t) override;
        virtual IReadVisitor& operator()(IStructTraits* val, const std::string& name = "") override;
        virtual IReadVisitor& operator()(IContainerTraits* val, const std::string& name = "") override;

        virtual VisitorTraits traits() const override;

        virtual std::string getCurrentElementName() const override;

      private:
        template <class T>
        IReadVisitor& readPod(T* ptr, const std::string& name, const size_t cnt);

        cereal::JSONInputArchive m_ar;
        std::string m_last_read_name;
    };
}
