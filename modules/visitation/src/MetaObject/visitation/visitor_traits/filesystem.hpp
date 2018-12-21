#ifndef MO_VISITATION_FILESYSTEM_HPP
#define MO_VISITATION_FILESYSTEM_HPP

#include <MetaObject/visitation/IDynamicVisitor.hpp>
#include <MetaObject/types/file_types.hpp>

namespace mo
{
    IReadVisitor& read(IReadVisitor&, ReadFile*, const std::string& name, const size_t);
    IReadVisitor& read(IReadVisitor&, WriteFile*, const std::string& name, const size_t);
    IReadVisitor& read(IReadVisitor&, ReadDirectory*, const std::string& name, const size_t);
    IReadVisitor& read(IReadVisitor&, WriteDirectory*, const std::string& name, const size_t);

    IWriteVisitor& write(IWriteVisitor&, const ReadFile*, const std::string& name, const size_t);
    IWriteVisitor& write(IWriteVisitor&, const WriteFile*, const std::string& name, const size_t);
    IWriteVisitor& write(IWriteVisitor&, const ReadDirectory*, const std::string& name, const size_t);
    IWriteVisitor& write(IWriteVisitor&, const WriteDirectory*, const std::string& name, const size_t);
}

#endif // MO_VISITATION_FILESYSTEM_HPP

