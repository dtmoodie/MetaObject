#pragma once
#include <MetaObject/detail/TypeInfo.hpp>
#include <memory>

namespace cereal
{
    class BinaryInputArchive;
    class BinaryOutputArchive;
} // namespace cereal

namespace mo
{
    using BinaryInputVisitor = cereal::BinaryInputArchive;
    using BinaryOutputVisitor = cereal::BinaryOutputArchive;

    struct ILoadVisitor;
    struct ISaveVisitor;
    struct StaticVisitor;
    struct Header;

    struct MO_EXPORTS IDataContainer : public std::enable_shared_from_this<IDataContainer>
    {
        using Ptr_t = std::shared_ptr<IDataContainer>;
        using ConstPtr_t = std::shared_ptr<const IDataContainer>;

        IDataContainer() = default;
        IDataContainer(const IDataContainer&) = default;
        IDataContainer(IDataContainer&&) noexcept = default;
        IDataContainer& operator=(const IDataContainer&) = default;
        IDataContainer& operator=(IDataContainer&&) noexcept = default;
        virtual ~IDataContainer();

        virtual TypeInfo getType() const = 0;

        virtual void load(ILoadVisitor&) = 0;
        virtual void save(ISaveVisitor&) const = 0;
        virtual void load(BinaryInputVisitor& ar) = 0;
        virtual void save(BinaryOutputVisitor& ar) const = 0;
        virtual void visit(StaticVisitor&) const = 0;

        virtual const Header& getHeader() const = 0;
    };
} // namespace mo
