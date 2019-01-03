#pragma once
#include <MetaObject/detail/TypeInfo.hpp>
#include <memory>

namespace cereal
{
    class BinaryInputArchive;
    class BinaryOutputArchive;
}

namespace mo
{
    using BinaryInputVisitor = cereal::BinaryInputArchive;
    using BinaryOutputVisitor = cereal::BinaryOutputArchive;

    struct ILoadVisitor;
    struct ISaveVisitor;
    struct Header;

    struct IDataContainer : public std::enable_shared_from_this<IDataContainer>
    {
        using Ptr = std::shared_ptr<IDataContainer>;
        using ConstPtr = std::shared_ptr<const IDataContainer>;

        virtual ~IDataContainer();
        virtual TypeInfo getType() const = 0;

        virtual void visit(ILoadVisitor&) = 0;
        virtual void visit(ISaveVisitor&) const = 0;
        virtual void visit(BinaryInputVisitor& ar) = 0;
        virtual void visit(BinaryOutputVisitor& ar) const = 0;

        virtual const Header& getHeader() const = 0;
    };
}
