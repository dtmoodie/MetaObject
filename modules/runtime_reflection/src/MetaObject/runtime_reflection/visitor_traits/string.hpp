#pragma once
#include "../DynamicVisitor.hpp"
#include "../ContainerTraits.hpp"

#include <string>

namespace mo
{
    template<>
    struct IsContinuous<std::string>
    {
        static constexpr const bool value = true;
    };

    template<>
    struct TTraits<std::string, void>: public ContainerBase<std::string>
    {
        TTraits(std::string* ptr);
    };

    template<>
    struct TTraits<const std::string, void>: public ContainerBase<const std::string>
    {
        TTraits(const std::string* ptr);
    };

    template<>
    struct Visit<std::string>
    {
        static ILoadVisitor& load(ILoadVisitor&, std::string* str, const std::string& name, const size_t cnt);
        static ISaveVisitor& save(ISaveVisitor&, const std::string* str, const std::string& name, const size_t cnt);
    };
}
