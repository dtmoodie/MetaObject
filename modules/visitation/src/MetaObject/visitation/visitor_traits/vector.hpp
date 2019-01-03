#pragma once
#include "../DynamicVisitor.hpp"
#include "../ContainerTraits.hpp"

namespace mo
{
    template<class T, class A>
    struct IsContinuous<std::vector<T, A>>
    {
        static constexpr const bool value = true;
    };

    template<class T, class A>
    struct Visit<std::vector<T, A>>
    {
        static ILoadVisitor& load(ILoadVisitor& visitor, std::vector<T, A>* val, const std::string& name, const size_t)
        {
            visitor(val->data(), name, val->size());
            return visitor;
        }

        static ISaveVisitor& save(ISaveVisitor& visitor, const std::vector<T, A>* val, const std::string& name, const size_t)
        {
            visitor(val->data(), name, val->size());
            return visitor;
        }
    };

    template<class T, class A>
    struct TTraits<std::vector<T, A>, void>: public ContainerBase<std::vector<T, A>>
    {
        TTraits(std::vector<T, A>* ptr):ContainerBase<std::vector<T, A>>(ptr){}
    };

    template<class T, class A>
    struct TTraits<const std::vector<T, A>, void>: public ContainerBase<const std::vector<T, A>>
    {
        TTraits(const std::vector<T, A>* ptr):ContainerBase<const std::vector<T, A>>(ptr){}

    };
}
