#ifndef MO_CORE_OBJECT_CONSTRUCTOR_HPP
#define MO_CORE_OBJECT_CONSTRUCTOR_HPP
#include <ct/type_traits.hpp>

#include <memory>

namespace mo
{
    template <class T, class E = void>
    struct ObjectConstructor
    {
        using SharedPtr_t = std::shared_ptr<T>;
        using UniquePtr_t = std::unique_ptr<T>;

        template <class... ARGS>
        SharedPtr_t makeShared(ARGS&&... args) const
        {
            return std::make_shared<T>(std::forward<ARGS>(args)...);
        }

        template <class... ARGS>
        SharedPtr_t createShared(ARGS&&... args) const
        {
            return makeShared(std::forward<ARGS>(args)...);
        }

        template <class... ARGS>
        UniquePtr_t createUnique(ARGS&&... args) const
        {
            return UniquePtr_t(create(std::forward<ARGS>(args)...));
        }

        template <class... ARGS>
        T* create(ARGS&&... args) const
        {
            return new T(std::forward<ARGS>(args)...);
        }
    };

} // namespace mo

#endif // MO_CORE_OBJECT_CONSTRUCTOR_HPP
