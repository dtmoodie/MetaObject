#ifndef MO_PARAMS_TRAITS_CONTAINERS_HPP
#define MO_PARAMS_TRAITS_CONTAINERS_HPP
#include <MetaObject/params/TStlAllocator.hpp>

namespace mo
{
    template <class T, class E>
    struct ContainerTraits;

    template <class T, class A>
    struct ContainerTraits<std::vector<T, A>, ct::EnableIf<std::is_trivially_copyable<T>::value>>
    {
        using type = std::vector<T, TStlAllocator<T>>;
        static constexpr const bool CONST = false;

        ContainerTraits(ParamAllocator::Ptr_t alloc)
            : m_allocator(std::move(alloc))
        {
        }

        ContainerTraits()
        {
        }

        template <class... ARGS>
        type create(ARGS&&... args)
        {
            return type(std::forward<ARGS>(args)..., TStlAllocator<T>(m_allocator));
        }

        template <class... ARGS>
        static type staticCreate(const ParamAllocator::Ptr_t& alloc, ARGS&&... args)
        {
            return type(std::forward<ARGS>(args)..., TStlAllocator<T>(alloc));
        }

        ParamAllocator::Ptr_t getAllocator() const
        {
            return m_allocator;
        }

        void record(IAsyncStream& src) const
        {
        }

        void sync(IAsyncStream& dst) const
        {
        }

      private:
        ParamAllocator::Ptr_t m_allocator;
    };

    template <class T>
    struct ContainerTraits<ct::TArrayView<const T>>
    {
        using type = ct::TArrayView<const T>;
        static constexpr const bool CONST = true;

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

#endif // MO_PARAMS_TRAITS_CONTAINERS_HPP