#ifndef MO_VISITOR_TRAITS_TIME_HPP
#define MO_VISITOR_TRAITS_TIME_HPP
#include "../StructTraits.hpp"
#include <MetaObject/core/detail/Time.hpp>
#include <MetaObject/runtime_reflection/IDynamicVisitor.hpp>
namespace mo
{
    template <class T>
    struct TTraits<boost::optional<T>, 4> : StructBase<boost::optional<T>>
    {
        void load(ILoadVisitor& visitor, void* inst, const std::string&, size_t) const override
        {
            boost::optional<T>& ref = this->ref(inst);
            bool valid = false;
            visitor(&valid, "valid");
            if (valid)
            {
                T val;
                visitor(&val, "value");
                ref = std::move(val);
            }
        }

        void save(ISaveVisitor& visitor, const void* inst, const std::string&, size_t) const override
        {
            const boost::optional<T>& ref = this->ref(inst);
            const bool valid = static_cast<bool>(ref);
            visitor(&valid, "valid");
            if (valid)
            {
                visitor(static_cast<const T*>(&*ref), "value");
            }
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            visitor.template visit<bool>("valid");
            visitor.template visit<T>("value");
        }

        uint32_t getNumMembers() const override
        {
            return 1;
        }

        bool getMember(
            void* inst, void** member, const IStructTraits** trait, uint32_t idx, std::string* name) const override
        {
            if (idx == 0)
            {
                boost::optional<T>& ref = this->ref(inst);
                if (!ref.is_initialized())
                {
                    ref = T();
                }
                *member = static_cast<void*>(&ref.value());
                static const TTraits<T> mem_trait;
                *trait = &mem_trait;
                return true;
            }
            return false;
        }

        bool getMember(const void* inst,
                       const void** member,
                       const IStructTraits** trait,
                       uint32_t idx,
                       std::string* name) const override
        {
            if (idx == 0)
            {
                const boost::optional<T>& ref = this->ref(inst);
                if (!ref.is_initialized())
                {
                    return false;
                }
                *member = static_cast<const void*>(&ref.value());
                static const TTraits<T> mem_trait;
                *trait = &mem_trait;
                return true;
            }
            return false;
        }
    };

    template <>
    struct MO_EXPORTS TTraits<Time, 4> : StructBase<Time>
    {
        void load(ILoadVisitor& visitor, void* inst, const std::string&, size_t) const override;

        void save(ISaveVisitor& visitor, const void* inst, const std::string&, size_t) const override;

        void visit(StaticVisitor& visitor, const std::string&) const override;
    };

} // namespace mo

#endif // MO_VISITOR_TRAITS_TIME_HPP
