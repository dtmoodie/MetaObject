#pragma once
#include "../DynamicVisitor.hpp"
#include <string>

namespace mo
{
    template <>
    struct TTraits<std::string, void> : public IContainerTraits
    {
        using base = IContainerTraits;

        TTraits(std::string* ptr, const std::string* const_ptr);
        virtual void visit(IReadVisitor* visitor) override;
        virtual void visit(IWriteVisitor* visitor) const override;
        virtual TypeInfo keyType() const override;
        virtual TypeInfo valueType() const override;

        virtual TypeInfo type() const override;
        virtual bool isContinuous() const override;
        virtual bool podValues() const override;
        virtual bool podKeys() const override;
        virtual size_t getSize() const override;
        virtual void setSize(const size_t num) override;
        virtual std::string getName() const override;

      private:
        std::string* m_ptr;
        const std::string* m_const_ptr;
        size_t num_to_read = 0;
    };
}
