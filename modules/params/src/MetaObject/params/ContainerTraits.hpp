#ifndef MO_PARAMS_CONTAINERTRAITS_HPP
#define MO_PARAMS_CONTAINERTRAITS_HPP
#include "ParamAllocator.hpp"

namespace mo
{
    template <class T>
    struct TDataContainer;

    template <class T, class E = void>
    struct ContainerTraits
    {
        using type = T;
        static constexpr const bool CONST = false;

        ContainerTraits(ParamAllocator::Ptr_t)
        {
        }

        ContainerTraits()
        {
        }

        template <class... ARGS>
        type create(ARGS&&... args)
        {
            return type(std::forward<ARGS>(args)...);
        }

        template <class... ARGS>
        static type staticCreate(const ParamAllocator::Ptr_t&, ARGS&&... args)
        {
            return type(std::forward<ARGS>(args)...);
        }

        ParamAllocator::Ptr_t getAllocator() const
        {
            return {};
        }

        void record(IAsyncStream& src) const
        {
        }

        void sync(IAsyncStream& dst) const
        {
        }
    };
} // namespace mo

#include <MetaObject/params/traits/Containers.hpp>

#endif // MO_PARAMS_CONTAINERTRAITS_HPP