#ifndef MO_VISITATION_FILESYSTEM_HPP
#define MO_VISITATION_FILESYSTEM_HPP

#include <MetaObject/visitation/IDynamicVisitor.hpp>
#include <MetaObject/types/file_types.hpp>

namespace mo
{
    template<>
    struct Visit<ReadFile>
    {
        static IReadVisitor& read(IReadVisitor&, ReadFile*, const std::string& name, const size_t);
        static IWriteVisitor& write(IWriteVisitor&, const ReadFile*, const std::string& name, const size_t);
    };

    template<>
    struct Visit<WriteFile>
    {
        static IReadVisitor& read(IReadVisitor&, WriteFile*, const std::string& name, const size_t);
        static IWriteVisitor& write(IWriteVisitor&, const WriteFile*, const std::string& name, const size_t);
    };

    template<>
    struct Visit<ReadDirectory>
    {
        static IReadVisitor& read(IReadVisitor&, ReadDirectory*, const std::string& name, const size_t);
        static IWriteVisitor& write(IWriteVisitor&, const ReadDirectory*, const std::string& name, const size_t);
    };

    template<>
    struct Visit<WriteDirectory>
    {
        static IReadVisitor& read(IReadVisitor&, WriteDirectory*, const std::string& name, const size_t);
        static IWriteVisitor& write(IWriteVisitor&, const WriteDirectory*, const std::string& name, const size_t);
    };
}

#endif // MO_VISITATION_FILESYSTEM_HPP

