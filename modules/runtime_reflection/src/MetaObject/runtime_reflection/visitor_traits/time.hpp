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
            auto& ref = this->ref(inst);
            bool valid = false;
            visitor(&valid, "valid");
            if (valid)
            {
                T val;
                visitor(&val, "value");
                ref = val;
            }
        }

        void save(ISaveVisitor& visitor, const void* inst, const std::string&, size_t) const override
        {
            const auto& ref = this->ref(inst);
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
