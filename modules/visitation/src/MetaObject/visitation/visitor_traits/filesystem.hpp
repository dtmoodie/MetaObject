#ifndef MO_VISITATION_FILESYSTEM_HPP
#define MO_VISITATION_FILESYSTEM_HPP

#include <MetaObject/visitation/IDynamicVisitor.hpp>
#include <MetaObject/types/file_types.hpp>

namespace mo
{
    template<>
    IWriteVisitor& visit(IWriteVisitor&, const ReadFile*, const std::string& name, const size_t);

    template<>
    IWriteVisitor& visit(IWriteVisitor&, const WriteFile*, const std::string& name, const size_t);

    template<>
    IWriteVisitor& visit(IWriteVisitor&, const ReadDirectory*, const std::string& name, const size_t);

    template<>
    IWriteVisitor& visit(IWriteVisitor&, const WriteDirectory*, const std::string& name, const size_t);
}

#endif // MO_VISITATION_FILESYSTEM_HPP

