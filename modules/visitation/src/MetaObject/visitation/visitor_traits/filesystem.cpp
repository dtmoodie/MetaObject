#include "filesystem.hpp"
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/visitation/visitor_traits/string.hpp>

namespace mo
{
    namespace impl
    {
        template<class T>
        ISaveVisitor& save(ISaveVisitor& visitor, const T* file, const std::string& name, const size_t cnt)
        {
            MO_ASSERT_EQ(cnt, 1);
            const auto& string = file->string();
            visitor(&string, name, 1);
            return visitor;
        }

        template<class T>
        ILoadVisitor& load(ILoadVisitor& visitor, T* file, const std::string& name, const size_t cnt)
        {
            MO_ASSERT_EQ(cnt, 1);
            std::string path;
            visitor(&path, name, 1);
            *file = path;
            return visitor;
        }
    }

    ILoadVisitor& Visit<ReadFile>::load(ILoadVisitor& visitor, ReadFile* dir, const std::string& name, const size_t cnt)
    {
        return impl::load(visitor, dir, name, cnt);
    }

    ILoadVisitor& Visit<WriteFile>::load(ILoadVisitor& visitor, WriteFile* dir, const std::string& name, const size_t cnt)
    {
        return impl::load(visitor, dir, name, cnt);
    }

    ILoadVisitor& Visit<ReadDirectory>::load(ILoadVisitor& visitor, ReadDirectory* dir, const std::string& name, const size_t cnt)
    {
        return impl::load(visitor, dir, name, cnt);
    }

    ILoadVisitor& Visit<WriteDirectory>::load(ILoadVisitor& visitor, WriteDirectory* dir, const std::string& name, const size_t cnt)
    {
        return impl::load(visitor, dir, name, cnt);
    }

    ISaveVisitor& Visit<ReadFile>::save(ISaveVisitor& visitor, const ReadFile* file, const std::string& name, const size_t cnt)
    {
        return impl::save(visitor, file, name, cnt);
    }

    ISaveVisitor& Visit<WriteFile>::save(ISaveVisitor& visitor, const WriteFile* file, const std::string& name, const size_t cnt)
    {
        return impl::save(visitor, file, name, cnt);
    }

    ISaveVisitor& Visit<ReadDirectory>::save(ISaveVisitor& visitor, const ReadDirectory* dir, const std::string& name, const size_t cnt)
    {
        return impl::save(visitor, dir, name, cnt);
    }

    ISaveVisitor& Visit<WriteDirectory>::save(ISaveVisitor& visitor, const WriteDirectory* dir, const std::string& name, const size_t cnt)
    {
        return impl::save(visitor, dir, name, cnt);
    }
}
