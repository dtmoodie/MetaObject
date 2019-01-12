#include "string.hpp"

namespace mo
{
    TTraits<std::string, void>::TTraits(std::string* ptr)
        : ContainerBase<std::string>(ptr)
    {}

    TTraits<const std::string, void>::TTraits(const std::string* ptr)
        : ContainerBase<const std::string>(ptr)
    {}

    ILoadVisitor& Visit<std::string>::load(ILoadVisitor& visitor, std::string* str, const std::string& name, const size_t )
    {
        visitor(&(*str)[0], name, str->size());
    }

    ISaveVisitor& Visit<std::string>::save(ISaveVisitor& visitor, const std::string* str, const std::string& name, const size_t )
    {
        visitor(&(*str)[0], name, str->size());
    }
}
