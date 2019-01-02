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
        static IReadVisitor& read(IReadVisitor& visitor, std::vector<T, A>* val, const std::string& name, const size_t cnt)
        {
            if (IsPrimitive<T>::value)
            {
                visitor(val->data(), name, val->size());
            }
            else
            {
                for(auto& v : *val)
                {
                    visitor(&v);
                }
            }
            return visitor;
        }

        static IWriteVisitor& write(IWriteVisitor& visitor, const std::vector<T, A>* val, const std::string& name, const size_t cnt)
        {
            if (IsPrimitive<T>::value)
            {
                visitor(val->data(), name, val->size());
            }
            else
            {
                for(const auto& v : *val)
                {
                    visitor(&v);
                }
            }
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
