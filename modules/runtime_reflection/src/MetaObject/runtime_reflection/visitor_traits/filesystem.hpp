#ifndef MO_VISITATION_FILESYSTEM_HPP
#define MO_VISITATION_FILESYSTEM_HPP

#include <MetaObject/types/file_types.hpp>
#include <MetaObject/runtime_reflection/IDynamicVisitor.hpp>

namespace mo
{
    template <>
    struct Visit<ReadFile>
    {
        static ILoadVisitor& load(ILoadVisitor&, ReadFile*, const std::string& name, const size_t);
        static ISaveVisitor& save(ISaveVisitor&, const ReadFile*, const std::string& name, const size_t);
        static StaticVisitor& visit(StaticVisitor&, const std::string& name, const size_t);
    };

    template <>
    struct Visit<WriteFile>
    {
        static ILoadVisitor& load(ILoadVisitor&, WriteFile*, const std::string& name, const size_t);
        static ISaveVisitor& save(ISaveVisitor&, const WriteFile*, const std::string& name, const size_t);
        static StaticVisitor& visit(StaticVisitor&, const std::string& name, const size_t);
    };

    template <>
    struct Visit<ReadDirectory>
    {
        static ILoadVisitor& load(ILoadVisitor&, ReadDirectory*, const std::string& name, const size_t);
        static ISaveVisitor& save(ISaveVisitor&, const ReadDirectory*, const std::string& name, const size_t);
        static StaticVisitor& visit(StaticVisitor&, const std::string& name, const size_t);
    };

    template <>
    struct Visit<WriteDirectory>
    {
        static ILoadVisitor& load(ILoadVisitor&, WriteDirectory*, const std::string& name, const size_t);
        static ISaveVisitor& save(ISaveVisitor&, const WriteDirectory*, const std::string& name, const size_t);
        static StaticVisitor& visit(StaticVisitor&, const std::string& name, const size_t);
    };
}

#endif // MO_VISITATION_FILESYSTEM_HPP
