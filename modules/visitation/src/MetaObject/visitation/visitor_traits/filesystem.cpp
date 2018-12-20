#include "filesystem.hpp"
#include <MetaObject/logging/logging.hpp>
namespace mo
{
    template <>
    IWriteVisitor& visit(IWriteVisitor& visitor, const ReadFile* file, const std::string& name, const size_t cnt)
    {
        MO_ASSERT_EQ(cnt, 1);
        const auto string = file->string();
        visitor(&string, name, 1);
        return visitor;
    }

    template <>
    IWriteVisitor& visit(IWriteVisitor& visitor, const WriteFile* file, const std::string& name, const size_t cnt)
    {
        MO_ASSERT_EQ(cnt, 1);
        const auto string = file->string();
        visitor(&string, name, 1);
        return visitor;
    }

    template <>
    IWriteVisitor& visit(IWriteVisitor& visitor, const ReadDirectory* dir, const std::string& name, const size_t cnt)
    {
        MO_ASSERT_EQ(cnt, 1);
        const auto string = dir->string();
        visitor(&string, name, 1);
        return visitor;
    }

    template <>
    IWriteVisitor& visit(IWriteVisitor& visitor, const WriteDirectory* dir, const std::string& name, const size_t cnt)
    {
        MO_ASSERT_EQ(cnt, 1);
        const auto string = dir->string();
        visitor(&string, name, 1);
        return visitor;
    }
}
