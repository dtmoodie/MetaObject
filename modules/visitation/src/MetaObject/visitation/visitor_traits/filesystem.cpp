#include "filesystem.hpp"
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/visitation/visitor_traits/string.hpp>

namespace mo
{
    namespace impl
    {
        template<class T>
        IWriteVisitor& write(IWriteVisitor& visitor, const T* file, const std::string& name, const size_t cnt)
        {
            MO_ASSERT_EQ(cnt, 1);
            const auto string = file->string();
            visitor(&string, name, 1);
            return visitor;
        }

        template<class T>
        IReadVisitor& read(IReadVisitor& visitor, T* file, const std::string& name, const size_t cnt)
        {
            MO_ASSERT_EQ(cnt, 1);
            std::string path;
            visitor(&path, name, 1);
            *file = path;
            return visitor;
        }
    }

    IReadVisitor& Visit<ReadFile>::read(IReadVisitor& visitor, ReadFile* dir, const std::string& name, const size_t cnt)
    {
        return impl::read(visitor, dir, name, cnt);
    }

    IReadVisitor& Visit<WriteFile>::read(IReadVisitor& visitor, WriteFile* dir, const std::string& name, const size_t cnt)
    {
        return impl::read(visitor, dir, name, cnt);
    }

    IReadVisitor& Visit<ReadDirectory>::read(IReadVisitor& visitor, ReadDirectory* dir, const std::string& name, const size_t cnt)
    {
        return impl::read(visitor, dir, name, cnt);
    }

    IReadVisitor& Visit<WriteDirectory>::read(IReadVisitor& visitor, WriteDirectory* dir, const std::string& name, const size_t cnt)
    {
        return impl::read(visitor, dir, name, cnt);
    }

    IWriteVisitor& Visit<ReadFile>::write(IWriteVisitor& visitor, const ReadFile* file, const std::string& name, const size_t cnt)
    {
        return impl::write(visitor, file, name, cnt);
    }

    IWriteVisitor& Visit<WriteFile>::write(IWriteVisitor& visitor, const WriteFile* file, const std::string& name, const size_t cnt)
    {
        return impl::write(visitor, file, name, cnt);
    }

    IWriteVisitor& Visit<ReadDirectory>::write(IWriteVisitor& visitor, const ReadDirectory* dir, const std::string& name, const size_t cnt)
    {
        return impl::write(visitor, dir, name, cnt);
    }

    IWriteVisitor& Visit<WriteDirectory>::write(IWriteVisitor& visitor, const WriteDirectory* dir, const std::string& name, const size_t cnt)
    {
        return impl::write(visitor, dir, name, cnt);
    }
}
