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
    struct IAsyncStream;

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

        virtual void load(ILoadVisitor&, const std::string& name) = 0;
        virtual void save(ISaveVisitor&, const std::string& name) const = 0;
        virtual void visit(StaticVisitor&) const = 0;

        virtual const Header& getHeader() const = 0;

        virtual void record(IAsyncStream& src) const = 0;
        virtual void sync(IAsyncStream& dest) const = 0;

        template <class T>
        T* ptr();
        template <class T>
        const T* ptr() const;
    };
} // namespace mo

namespace ct
{
    REFLECT_BEGIN(mo::IDataContainer)
        MEMBER_FUNCTION(save)
        MEMBER_FUNCTION(load)
        MEMBER_FUNCTION(record)
        MEMBER_FUNCTION(sync)
    REFLECT_END;
} // namespace ct