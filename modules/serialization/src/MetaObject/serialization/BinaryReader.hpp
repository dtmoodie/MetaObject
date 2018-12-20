#pragma once
#include <MetaObject/visitation/DynamicVisitor.hpp>
#include <iostream>
#include <unordered_map>

namespace mo
{
    class BinaryReader : public ReadCache
    {
      public:
        BinaryReader(std::istream& in);
        virtual ~BinaryReader() override;
        virtual IReadVisitor& operator()(bool* ptr, const std::string& name, const size_t cnt) override;
        virtual IReadVisitor& operator()(char* ptr, const std::string& name, const size_t cnt) override;
        virtual IReadVisitor& operator()(int8_t* ptr, const std::string& name, const size_t cnt) override;
        virtual IReadVisitor& operator()(uint8_t* ptr, const std::string& name, const size_t cnt) override;
        virtual IReadVisitor& operator()(int16_t* ptr, const std::string& name, const size_t cnt) override;
        virtual IReadVisitor& operator()(uint16_t* ptr, const std::string& name, const size_t cnt) override;
        virtual IReadVisitor& operator()(int32_t* ptr, const std::string& name, const size_t cnt) override;
        virtual IReadVisitor& operator()(uint32_t* ptr, const std::string& name, const size_t cnt) override;
        virtual IReadVisitor& operator()(int64_t* ptr, const std::string& name, const size_t cnt) override;
        virtual IReadVisitor& operator()(uint64_t* ptr, const std::string& name, const size_t cnt) override;

        virtual IReadVisitor& operator()(float* val, const std::string& name, const size_t cnt) override;
        virtual IReadVisitor& operator()(double*, const std::string&, const size_t cnt) override;
        virtual IReadVisitor& operator()(void*, const std::string&, const size_t cnt) override;
        virtual IReadVisitor& operator()(IStructTraits* val, const std::string& name = "") override;
        virtual IReadVisitor& operator()(IContainerTraits* val, const std::string& name = "") override;

        virtual VisitorTraits traits() const override;
        virtual std::string getCurrentElementName() const override;

      private:
        template <class T>
        IReadVisitor& readBinary(T* ptr, const size_t cnt = 1);

        std::istream& m_is;
    };
}
