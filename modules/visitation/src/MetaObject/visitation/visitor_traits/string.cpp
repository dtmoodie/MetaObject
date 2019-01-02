#include "string.hpp"

namespace mo
{
    TTraits<std::string, void>::TTraits(std::string* ptr):ContainerBase<std::string>(ptr){}
    TTraits<const std::string, void>::TTraits(const std::string* ptr):ContainerBase<const std::string>(ptr){}
    IReadVisitor& Visit<std::string>::read(IReadVisitor& visitor, std::string* str, const std::string& name, const size_t cnt)
    {
        visitor(&(*str)[0], name, str->size());
    }

    IWriteVisitor& Visit<std::string>::write(IWriteVisitor& visitor, const std::string* str, const std::string& name, const size_t cnt)
    {
        visitor(&(*str)[0], name, str->size());
    }
}
