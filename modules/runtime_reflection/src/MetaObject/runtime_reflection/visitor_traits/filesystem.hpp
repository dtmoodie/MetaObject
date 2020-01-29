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
    } // namespace impl

    /*template <class T>
    struct TTraits<
        T,
        5,
        ct::EnableIf<ct::VariadicTypedef<ReadFile, WriteFile, ReadDirectory, WriteDirectory>::template contains<T>()>>
        : public ContainerBase<T, char>
    {

        void save(ISaveVisitor& visitor, const void* file_, const std::string& name, size_t cnt) const override
        {
            const T* file = static_cast<const T*>(file_);
            MO_ASSERT_EQ(cnt, 1);
            const auto& string = file->string();
            visitor(&string, name, 1);
        }

        void load(ILoadVisitor& visitor, void* file_, const std::string& name, size_t cnt) const override
        {
            T* file = static_cast<T*>(file_);
            MO_ASSERT_EQ(cnt, 1);
            std::string path;
            visitor(&path, name, 1);
            *file = std::move(path);
        }

        void visit(StaticVisitor& visitor, const std::string&) const override
        {
            visitor.template visit<std::string>("path");
        }
    };*/
} // namespace mo

#endif // MO_VISITATION_FILESYSTEM_HPP
