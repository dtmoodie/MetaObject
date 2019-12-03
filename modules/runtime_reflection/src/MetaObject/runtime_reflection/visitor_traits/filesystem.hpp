#ifndef MO_VISITATION_FILESYSTEM_HPP
#define MO_VISITATION_FILESYSTEM_HPP
#include "../ContainerTraits.hpp"

#include <ct/VariadicTypedef.hpp>
#include <ct/type_traits.hpp>

#include <MetaObject/runtime_reflection/IDynamicVisitor.hpp>
#include <MetaObject/types/file_types.hpp>

namespace mo
{
    namespace impl
    {
        template <class T>
        ISaveVisitor& save(ISaveVisitor& visitor, const T* file, const std::string& name, const size_t cnt)
        {
            MO_ASSERT_EQ(cnt, 1);
            const auto& string = file->string();
            visitor(&string, name, 1);
            return visitor;
        }

        template <class T>
        ILoadVisitor& load(ILoadVisitor& visitor, T* file, const std::string& name, const size_t cnt)
        {
            MO_ASSERT_EQ(cnt, 1);
            std::string path;
            visitor(&path, name, 1);
            *file = path;
            return visitor;
        }
    }



    template<class T>
    struct TTraits<T, 5, ct::EnableIf<ct::VariadicTypedef<ReadFile, WriteFile, ReadDirectory, WriteDirectory>::template count<T>() == 1>>: public ContainerBase<T, char>
    {

        ISaveVisitor& save(ISaveVisitor& visitor, const T* file, const std::string& name, const size_t cnt)
        {
            MO_ASSERT_EQ(cnt, 1);
            const auto& string = file->string();
            visitor(&string, name, 1);
            return visitor;
        }

        ILoadVisitor& load(ILoadVisitor& visitor, T* file, const std::string& name, const size_t cnt)
        {
            MO_ASSERT_EQ(cnt, 1);
            std::string path;
            visitor(&path, name, 1);
            *file = path;
            return visitor;
        }
    };

    template <>
    struct Visit<ReadFile>
    {
        static ILoadVisitor& load(ILoadVisitor&, ReadFile*, const std::string& name, size_t);
        static ISaveVisitor& save(ISaveVisitor&, const ReadFile*, const std::string& name, size_t);
        static StaticVisitor& visit(StaticVisitor&, const std::string& name, size_t);
    };

    template <>
    struct Visit<WriteFile>
    {
        static ILoadVisitor& load(ILoadVisitor&, WriteFile*, const std::string& name, size_t);
        static ISaveVisitor& save(ISaveVisitor&, const WriteFile*, const std::string& name, size_t);
        static StaticVisitor& visit(StaticVisitor&, const std::string& name, size_t);
    };

    template <>
    struct Visit<ReadDirectory>
    {
        static ILoadVisitor& load(ILoadVisitor&, ReadDirectory*, const std::string& name, size_t);
        static ISaveVisitor& save(ISaveVisitor&, const ReadDirectory*, const std::string& name, size_t);
        static StaticVisitor& visit(StaticVisitor&, const std::string& name, size_t);
    };

    template <>
    struct Visit<WriteDirectory>
    {
        static ILoadVisitor& load(ILoadVisitor&, WriteDirectory*, const std::string& name, size_t);
        static ISaveVisitor& save(ISaveVisitor&, const WriteDirectory*, const std::string& name, size_t);
        static StaticVisitor& visit(StaticVisitor&, const std::string& name, size_t);
    };
}

#endif // MO_VISITATION_FILESYSTEM_HPP
